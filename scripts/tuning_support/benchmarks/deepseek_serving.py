"""Matched DeepSeek 4.1 Flash 1017 serving screen; never starts a model.

Prepare prompts with the pinned native encoder. Raw decode uses exact queued
token-list counts and server monotonic times from the optional phase observer.
SSE pieces are never independently tokenized. Without telemetry C8 stays
unverified. Effective cold prefill includes request/first-output overhead.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks import serving
from silyx.adapters import deepseek_phase_observer as observer
from silyx.adapters.deepseek_exl3_prefill import MODEL, REVISION, verify_checkpoint

SERVED = 'deepseek-v4.1-flash-1017-tensorfold'
ENCODER_SHA = '3962b8f83cbebb4d0cad98d44a5020c170292e942cbbfd97cc2ddd0506e8f880'


def sha(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


class NativeFactory(serving.PromptFactory):
    def __init__(self, model, encoder):
        model, encoder = Path(model), Path(encoder)
        verify_checkpoint(model)
        if hashlib.sha256(encoder.read_bytes()).hexdigest() != ENCODER_SHA:
            raise ValueError('exact mounted 1017 native encoder required')
        spec = importlib.util.spec_from_file_location('deepseek_1017_native_encoder', encoder)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        from tokenizers import Tokenizer
        rust = Tokenizer.from_file(str(model/'tokenizer.json'))

        class Tokens:
            def encode(self, text):
                return rust.encode(text, add_special_tokens=False).ids
            def decode(self, ids):
                return rust.decode(ids, skip_special_tokens=False)
        self.tokenizer = Tokens()
        self.native = native
        self.files = {'encoding.py': ENCODER_SHA,
                      'tokenizer.json': hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest()}

    def render(self, content, thinking):
        return self.native.encode([{'role': 'user', 'content': content}], thinking=thinking == 'high', effort=75)


def make_plan(factory, *, repeats=2, output_tokens=64):
    if type(repeats) is not int or not 1 <= repeats <= 5 or type(output_tokens) is not int or not 8 <= output_tokens <= 128:
        raise ValueError('bounded screen requires repeats1..5 and output8..128')
    bursts = []
    for phase in ('decode', 'prefill'):
        for thinking in ('off', 'high'):
            for geometry in (1, 8) if phase == 'decode' else (1536, 3072):
                concurrency, inputs = (geometry, 128) if phase == 'decode' else (1, geometry)
                for repeat in range(repeats):
                    pairs = []
                    for stream in range(concurrency):
                        instruction = (serving.PROSE[stream]+' Give concrete details and continue until the response limit.'
                                       if phase == 'decode' else
                                       'Read the following context. Reply with OK only after reading it.')
                        content = f'Public DeepSeek 1017 screen {phase} {inputs} {stream}. '+instruction
                        prepared, prompt_sha, expected = factory.prepare(content, thinking, inputs)
                        ids = list(factory.tokenizer.encode(factory.render(prepared['messages'][0]['content'], thinking)))
                        item = dict(phase=phase, workload='prose' if phase == 'decode' else 'prefill',
                                    thinking=thinking, concurrency=concurrency, repeat=repeat, stream=stream,
                                    output_cap=output_tokens if phase == 'decode' else 1,
                                    expected_prompt_tokens=expected, prompt_sha256=prompt_sha,
                                    prompt_ids=ids, prompt_ids_sha256=sha(ids))
                        body = dict(model=SERVED, **prepared, max_tokens=item['output_cap'], temperature=0,
                                    stream=True, stream_options={'include_usage': True}, return_token_ids=True,
                                    chat_template_kwargs={'thinking': thinking == 'high',
                                                          'enable_thinking': thinking == 'high'},
                                    reasoning_effort='high' if thinking == 'high' else 'none')
                        pairs.append([item, body])
                    bursts.append(pairs)
    return dict(schema=1, kind='deepseek1017-native-serving-plan', target_model=MODEL, revision=REVISION,
                served_model=SERVED, tokenizer_files_sha256=factory.files, bursts=bursts)


def validate_plan(plan):
    if (plan.get('schema') != 1 or plan.get('kind') != 'deepseek1017-native-serving-plan'
            or plan.get('target_model') != MODEL or plan.get('revision') != REVISION
            or plan.get('served_model') != SERVED
            or plan.get('tokenizer_files_sha256', {}).get('encoding.py') != ENCODER_SHA):
        raise ValueError('exact native 1017 plan identity required')
    for pairs in plan['bursts']:
        for item, body in pairs:
            ids = item['prompt_ids']
            if (not ids or any(type(token) is not int or token < 0 for token in ids)
                    or len(ids) != item['expected_prompt_tokens'] or sha(ids) != item['prompt_ids_sha256']
                    or body['model'] != SERVED or body['max_tokens'] != item['output_cap']
                    or any(key in body for key in ('cache_salt', 'tf_policy', 'tf_copy'))):
                raise ValueError('invalid prompt/token or unsupported DeepSeek control')
    return plan


def checked_trace(row):
    stats = row['backend'].get('tensorfold', {})
    proof = stats.get('deepseek_phase', {})
    ids = stats.get('token_ids')
    events = proof.get('emissions', [])
    if (proof.get('schema') != 1 or proof.get('source_sha256') != observer.PINS
            or proof.get('observer_sha256') != hashlib.sha256(Path(observer.__file__).read_bytes()).hexdigest()
            or proof.get('clock') != 'server_time.monotonic' or proof.get('truncated') is not False
            or not isinstance(ids, list) or len(ids) != row['completion_tokens']
            or any(type(token) is not int for token in ids) or not events):
        raise ValueError('missing exact observer/token trace')
    total, prior = 0, -1.0
    for event in events:
        at, count = event['at_s'], event['count']
        if (not isinstance(at, (int, float)) or not math.isfinite(at) or at < prior or type(count) is not int or count < 1
                or event['phase'].get('executor_active') is not False
                or event['phase'].get('round_errors') != 0):
            raise ValueError('invalid exact token emission trace')
        total += count
        if event['emitted'] != total:
            raise ValueError('token trace offsets are not consecutive')
        prior = at
    if (total != len(ids) or proof['emitted_tokens'] != total
            or proof['end'].get('round_errors') != 0 or proof['end'].get('executor_active') is not False):
        raise ValueError('queue emissions and final output IDs differ')
    row['token_ids_sha256'] = sha(ids)
    return proof


def summarize(rows, before, after):
    protected = serving.isolation('tensorfold', before, after, rows)
    isolated = (protected['verified'] and protected['response_ids_unique']
                and all(value.get('ok') is True and not value.get('fatal') and not value.get('stalled')
                        for value in (before, after))
                and serving.counter_delta(before, after, 'errors') == 0)
    capped = all(row['cap_satisfied'] for row in rows)
    item = rows[0]
    result = {key: item[key] for key in ('phase', 'thinking', 'concurrency', 'repeat', 'output_cap')}
    result.update(outside_clients=protected, cap_satisfied=capped,
                  raw_decode_aggregate_tps=None, pure_decode_verified=False,
                  effective_cold_prefill_input_tps=None)
    first_phase, last_phase = before.get('deepseek_phase', {}), after.get('deepseek_phase', {})
    graph_deltas = {}
    for name in ('target_graphs', 'draft_graphs'):
        first_stats = first_phase.get(name, {}).get('stats', {})
        last_stats = last_phase.get(name, {}).get('stats', {})
        graph_deltas[name] = {key: serving.counter_delta(first_stats, last_stats, key)
                              for key in sorted(set(first_stats) | set(last_stats))}
    result['graph_stat_deltas'] = graph_deltas
    result['draft_graph_max_slots'] = last_phase.get('draft_graphs', {}).get('max_slots')
    result['speculative_draft_deltas'] = {
        key: serving.counter_delta(first_phase.get('speculative_draft', {}),
                                   last_phase.get('speculative_draft', {}), key)
        for key in sorted(set(first_phase.get('speculative_draft', {})) | set(last_phase.get('speculative_draft', {})))}
    try:
        proofs = [checked_trace(row) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        result['telemetry_error'] = str(error)
        return result
    # Server timestamps share one process clock. Exclude every token in the
    # boundary emission, including multiple accepted draft tokens in that put.
    start = max(proof['emissions'][0]['at_s'] for proof in proofs)
    end = max(proof['emissions'][-1]['at_s'] for proof in proofs)
    boundary = next(event for proof in proofs for event in proof['emissions'] if event['at_s'] == start)
    phase_keys = ('prefill_tokens', 'prefill_finalizations', 'prefill_rounds')
    signature = tuple(boundary['phase'][key] for key in phase_keys)
    later = [event for proof in proofs for event in proof['emissions'] if event['at_s'] > start]
    ends = [proof['end'] for proof in proofs]
    no_prefill = all(tuple(value[key] for key in phase_keys) == signature
                     for value in [event['phase'] for event in later]+ends)
    tokens = sum(event['count'] for event in later)
    pure = (isolated and capped and no_prefill and end > start and tokens > 0
            and all(proof['emissions'][-1]['at_s'] > start for proof in proofs))
    result.update(pure_decode_verified=pure,
                  decode_scope='batch aggregate suffix after latest first queue emission; exact counts, includes finishing tails',
                  raw_decode_aggregate_tps=tokens/(end-start) if pure and item['phase'] == 'decode' else None,
                  decode_tokens=tokens, decode_seconds=end-start,
                  prefill_unchanged_in_suffix=no_prefill,
                  ttft_p50_seconds=statistics.median(row['ttft_seconds'] for row in rows),
                  observed_client_interval_tps=[row['observed_stream_interval_tps'] for row in rows])
    if item['phase'] == 'prefill':
        a, b = before.get('deepseek_phase', {}), after.get('deepseek_phase', {})
        inputs = sum(row['prompt_tokens'] for row in rows)
        cold = (isolated and all(proof.get('session_store_disabled') is True for proof in proofs)
                and a.get('session_store_disabled') is True and b.get('session_store_disabled') is True
                and serving.counter_delta(before, after, 'cached_tokens_total') == 0
                and serving.counter_delta(a, b, 'prefill_tokens') == inputs-len(rows)
                and serving.counter_delta(a, b, 'prefill_finalizations') == len(rows)
                and all(row['backend']['tensorfold'].get('cached') == 0 for row in rows))
        wall = max(row['first_visible_s'] for row in rows)-min(row['request_started_s'] for row in rows)
        result.update(cold_prefill_verified=cold,
                      effective_cold_prefill_input_tps=inputs/wall if cold and wall > 0 else None,
                      prefill_scope='client request-to-first-visible-output, cap1; includes scheduling and first verify')
    return result


def run(plan, endpoint, out, *, baseline=None, warmup=False):
    validate_plan(plan)
    models = endpoint.get('/v1/models')
    if SERVED not in {value['id'] for value in models.get('data', [])}:
        raise RuntimeError('the exact 1017 TensorFold model is not loaded')
    reference = {}
    if baseline:
        reference = {serving.case_key(value): value for value in
                     (json.loads(line) for line in Path(baseline).read_text().splitlines())}
    summaries, warmed = [], set()
    with (out/'requests.jsonl').open('w') as records:
        for index, pairs in enumerate(plan['bursts']):
            items, bodies = zip(*pairs)
            geometry = tuple(items[0][key] for key in ('phase', 'thinking', 'concurrency', 'expected_prompt_tokens'))
            if warmup and geometry not in warmed:
                warm = serving.burst(endpoint, [{**item, 'output_cap': min(8, item['output_cap'])} for item in items],
                                     [{**body, 'max_tokens': min(8, body['max_tokens'])} for body in bodies])
                (out/f'warmup-{index:02d}.json').write_text(json.dumps(warm)+'\n')
                warmed.add(geometry)
            before = endpoint.get('/health')
            rows = serving.burst(endpoint, items, bodies)
            after = endpoint.get('/health')
            summary = summarize(rows, before, after)
            (out/f'health-{index:02d}.json').write_text(json.dumps(dict(before=before, after=after), indent=2)+'\n')
            for row in rows:
                previous = reference.get(serving.case_key(row))
                row['baseline_matched_prompt'] = (previous.get('prompt_ids_sha256') == row['prompt_ids_sha256']) if previous else None
                row['baseline_token_ids_match'] = (previous['backend'].get('tensorfold', {}).get('token_ids')
                                                  == row['backend'].get('tensorfold', {}).get('token_ids')) if previous else None
                row['baseline_text_match'] = previous['output_text_sha256'] == row['output_text_sha256'] if previous else None
                records.write(json.dumps(row)+'\n')
            records.flush()
            summaries.append(summary)
            (out/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
            print(json.dumps(summary), flush=True)
            if not summary['outside_clients']['verified']:
                raise RuntimeError('outside-client isolation failed; stop the screen')
            if not summary['cap_satisfied']:
                raise RuntimeError('output cap not reached; stop the screen')
            if summary.get('telemetry_error') or (items[0]['phase'] == 'decode' and not summary['pure_decode_verified']):
                raise RuntimeError('pure exact-token decode telemetry failed; stop the screen')
            if items[0]['phase'] == 'prefill' and not summary.get('cold_prefill_verified'):
                raise RuntimeError('cold-prefill proof failed; stop the screen')
            if baseline and any(row.get(key) is not True for row in rows
                                for key in ('baseline_matched_prompt', 'baseline_token_ids_match', 'baseline_text_match')):
                raise RuntimeError('matched baseline prompt/output changed; stop before further timing')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, help='reuse an already prepared native prompt plan')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--base', default='http://127.0.0.1:8888')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output directory must be new')
    if args.plan:
        plan = validate_plan(json.loads(args.plan.read_text()))
    else:
        if not args.tokenizer or not args.encoder:
            parser.error('preparation requires the exact local tokenizer and mounted native encoder')
        plan = make_plan(NativeFactory(args.tokenizer, args.encoder), repeats=args.repeats,
                         output_tokens=args.output_tokens)
    args.out.mkdir(parents=True)
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    metadata = dict(schema=1, engine='TensorFold', plan_sha256=sha(plan), status='prepared',
                    target_model=MODEL, revision=REVISION,
                    quality_scope='same native prompts and full output token/text comparison; no broad quality claim')
    (args.out/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps(dict(prepared_bursts=len(plan['bursts']), api_requests=0)))
        return
    try:
        run(plan, serving.Endpoint(args.base, os.environ.get('SILYX_API_KEY')), args.out,
            baseline=args.baseline, warmup=args.warmup)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata.update(status='failed', error_type=type(error).__name__, error=str(error)[:300])
        raise
    finally:
        (args.out/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
