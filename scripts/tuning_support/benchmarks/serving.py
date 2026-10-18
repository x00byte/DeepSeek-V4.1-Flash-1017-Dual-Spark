"""Matched streamed serving measurements. Never starts/stops a model service.

No model API is contacted with --prepare-only. Cold-prefill measurements require
a local tokenizer so the exact input length is known before timing. C8 intervals
are marked mixed/unverified unless backend metadata proves no interleaved prefill.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.parse import urlsplit
import uuid


SCHEMA = 1
PROSE = (
    "Explain how to design a distributed work queue with retries, cancellation and fairness.",
    "Explain tradeoffs between database snapshot isolation and serializable transactions.",
    "Explain how a streaming system handles late events, watermarks and recovery.",
    "Explain how to diagnose an intermittent network latency problem using controlled experiments.",
    "Explain how to design a file backup system with checksums, incremental copies and restores.",
    "Explain how to allocate a limited compute budget fairly among interactive and batch jobs.",
    "Explain the design of a bounded cache with eviction, admission and concurrent readers.",
    "Explain how to compare two engineering designs with uncertain costs and failure rates.",
)
STRUCTURED = (
    "Emit at least 200 JSON objects describing queue jobs with id, priority, retry_count and state.",
    "Emit at least 200 CSV rows describing database transactions with id, isolation, reads and writes.",
    "Emit at least 200 numbered test cases for a streaming event pipeline with input and expected output.",
    "Emit at least 200 JSON objects describing network probes with id, source, destination and latency_ms.",
    "Emit at least 200 CSV rows describing backup files with path, size, checksum and retention_days.",
    "Emit at least 200 numbered scheduling examples with job, resources, queue and start_time.",
    "Emit at least 200 JSON objects describing cache events with key, action, size and reason.",
    "Emit at least 200 numbered engineering risks with probability, impact and mitigation.",
)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values)*fraction)-1)] if values else None


class Endpoint:
    def __init__(self, base, api_key=None, timeout=600):
        url = urlsplit(base)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base must be an HTTP(S) endpoint without credentials or query")
        self.scheme, self.host, self.port = url.scheme, url.hostname, url.port
        self.prefix = url.path.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer "+api_key

    def connection(self):
        kind = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return kind(self.host, self.port, timeout=self.timeout)

    def get(self, path):
        connection = self.connection()
        try:
            connection.request("GET", self.prefix+path, headers=self.headers)
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"GET {path}: HTTP {response.status}")
            return json.loads(body)
        finally:
            connection.close()


class PromptFactory:
    def __init__(self, directory=None):
        self.tokenizer = None
        self.files = {}
        if directory:
            from silyx.runtime.tokenizer import HFTokenizer
            self.tokenizer = HFTokenizer(directory)
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
                path = Path(directory)/name
                if path.is_file():
                    self.files[name] = digest(path.read_bytes())

    def render(self, content, thinking):
        if self.tokenizer is None:
            return None
        return self.tokenizer.chat_template([{"role": "user", "content": content}], thinking=thinking == "high")

    def prepare(self, content, thinking, input_tokens=None):
        rendered = self.render(content, thinking)
        if rendered is None:
            if input_tokens is not None:
                raise ValueError("fixed input token counts require --tokenizer")
            return {"messages": [{"role": "user", "content": content}]}, digest(content), None
        count = len(self.tokenizer.encode(rendered))
        if input_tokens is not None:
            if count > input_tokens:
                raise ValueError(f"prompt needs {count} tokens; requested input length is {input_tokens}")
            if input_tokens-count > 24:
                # Vary context records instead of filling long prefills with a
                # single repeated token, which could bias expert/cache behavior.
                records = []
                for index in range(input_tokens):
                    value = int(digest(f"{content}:{index}")[:8], 16)
                    records.append(f"\nContext {index:04d}: job {value % 1000000:06d} in region {value % 7} "
                                   f"used {value % 53+1} units; retries={value % 4}, owner=team-{index % 8}.")
                    if len(records) >= max(2, (input_tokens-count)//12):
                        break
                filler = tuple(self.tokenizer.encode("".join(records)))
                take = min(len(filler), max(0, input_tokens-count-8))
                original = content
                while take:
                    content = original+self.tokenizer.decode(filler[:take])
                    rendered = self.render(content, thinking)
                    count = len(self.tokenizer.encode(rendered))
                    if count <= input_tokens:
                        break
                    take -= count-input_tokens+1
                if count > input_tokens:
                    content = original
                    rendered = self.render(content, thinking)
                    count = len(self.tokenizer.encode(rendered))
            if count < input_tokens:
                # Choose a one-token suffix with this exact template boundary.
                # No token estimates or truncation of the think-block tail.
                unit = next((value for value in (" x", " a", " 0", "\n")
                             if len(self.tokenizer.encode(self.render(content+value, thinking))) == count+1), None)
                if unit is None:
                    raise ValueError("tokenizer has no tested one-token padding suffix")
                content += unit*(input_tokens-count)
                rendered = self.render(content, thinking)
                count = len(self.tokenizer.encode(rendered))
                if count != input_tokens:
                    raise ValueError("padding did not preserve exact requested token length")
            if count != input_tokens:
                raise ValueError("prepared prompt does not match requested token length")
        # Use chat on both engines: TensorFold text completions intentionally
        # drop reasoning deltas, which would invalidate HIGH output timing.
        return {"messages": [{"role": "user", "content": content}]}, digest(rendered), count


def sse_events(response):
    lines = []
    while True:
        raw = response.readline(16*1024*1024)
        if not raw:
            if lines:
                data = b"\n".join(lines)
                if data != b"[DONE]":
                    yield json.loads(data)
            return
        if len(raw) >= 16*1024*1024:
            raise RuntimeError("SSE line exceeded safety limit")
        if raw in (b"\n", b"\r\n"):
            if not lines:
                continue
            data, lines = b"\n".join(lines), []
            if data == b"[DONE]":
                return
            yield json.loads(data)
        elif raw.startswith(b"data:"):
            lines.append(raw[5:].lstrip(b" ").rstrip(b"\r\n"))


def request(endpoint, item, body, barrier):
    connection = endpoint.connection()
    try:
        connected = time.perf_counter()
        connection.connect()
        connect_seconds = time.perf_counter()-connected
        barrier.wait()
        started = time.perf_counter()
        route = "/v1/completions" if "prompt" in body else "/v1/chat/completions"
        connection.request("POST", endpoint.prefix+route, json.dumps(body).encode(), endpoint.headers)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"completion HTTP {response.status}: {response.read(4096).decode(errors='replace')}")
        first = last = terminal = None
        usage, finish, response_id = None, None, None
        content, reasoning, combined = [], [], []
        extra, chunks = {}, 0
        for event in sse_events(response):
            now = time.perf_counter()
            if event.get("error"):
                raise RuntimeError(f"backend error: {event['error']}")
            identifier = event.get("id")
            if event.get("model") not in (None, body["model"]):
                raise RuntimeError("server replied with another model ID")
            if identifier:
                if response_id is not None and response_id != identifier:
                    raise RuntimeError("response ID changed within one stream")
                response_id = identifier
            if event.get("usage"):
                usage = event["usage"]
                terminal = now
            for key in ("silyx", "tensorfold", "timings", "stats", "speculative", "pass_economics"):
                if key in event:
                    extra[key] = event[key]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content", "") if "delta" in choice else choice.get("text", "")
                thought = delta.get("reasoning_content", delta.get("reasoning", ""))
                if text or thought:
                    first = now if first is None else first
                    last = now
                    chunks += 1
                    content.append(text)
                    reasoning.append(thought)
                    combined.append(thought+text)
                finish = choice.get("finish_reason") or finish
                if choice.get("finish_reason"):
                    terminal = now
        ended = time.perf_counter()
        if (not isinstance(usage, dict) or not response_id
                or any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens"))):
            raise RuntimeError("missing exact token usage or response ID; no estimated counts accepted")
        if item["expected_prompt_tokens"] is not None and usage["prompt_tokens"] != item["expected_prompt_tokens"]:
            raise RuntimeError("server prompt usage differs from exact locally prepared prompt")
        if first is None:
            if body["max_tokens"] > 1:
                raise RuntimeError("decode benchmark received no visible streamed text")
            first = last = terminal or ended
            first_scope = "terminal_fallback_no_visible_token"
        else:
            first_scope = "first_visible_content_or_reasoning"
        text = "".join(combined)
        row = {**item, "response_id": response_id, "prompt_tokens": usage["prompt_tokens"],
               "completion_tokens": usage["completion_tokens"], "finish_reason": finish,
               "connect_seconds": connect_seconds, "request_started_s": started,
               "first_visible_s": first, "last_visible_s": last, "ended_s": ended,
               "ttft_seconds": first-started, "ttft_scope": first_scope,
               "visible_decode_interval_seconds": last-first,
               "observed_stream_interval_tps": (usage["completion_tokens"]-1)/(last-first)
               if usage["completion_tokens"] > 1 and last > first else None,
               "chunks": chunks, "backend": extra, "output_text_sha256": digest(text),
               "output_text": text, "reasoning_text_separated": bool(any(reasoning)),
               "cap_satisfied": usage["completion_tokens"] == body["max_tokens"]}
        metrics = extra.get("silyx", {}).get("metrics", {})
        row["token_hash"] = metrics.get("token_sha256") or extra.get("tensorfold", {}).get("token_sha")
        row["token_hash_scheme"] = "sha256_decimal_ids_comma_separated" if row["token_hash"] else None
        return row
    except BaseException:
        barrier.abort()
        raise
    finally:
        connection.close()


def burst(endpoint, items, bodies):
    barrier = threading.Barrier(len(items)+1, timeout=min(endpoint.timeout, 30))
    with ThreadPoolExecutor(max_workers=len(items)) as workers:
        futures = [workers.submit(request, endpoint, item, body, barrier) for item, body in zip(items, bodies)]
        barrier.wait()  # Every TCP/TLS connection is ready before POST begins.
        return [future.result() for future in futures]


def counter_delta(before, after, name):
    a, b = before.get(name), after.get(name)
    return b-a if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b >= a else None


def isolation(engine, before, after, rows):
    if engine == "silyx":
        requests = counter_delta(before, after, "completed_requests")
        cancelled = counter_delta(before, after, "cancelled_requests")
        tokens = counter_delta(before, after, "generated_tokens")
        idle = all(h.get("active_requests") == 0 and h.get("pending_requests") == 0 for h in (before, after))
        exact_requests = requests == len(rows) and cancelled == 0
    else:
        requests = counter_delta(before, after, "requests_total")
        tokens = counter_delta(before, after, "completion_tokens_total")
        idle = all(h.get("requests_running") == 0 and not h.get("busy", False) for h in (before, after))
        exact_requests = requests == len(rows)
    stable = before.get("backend") == after.get("backend")
    verified = idle and exact_requests and stable and tokens == sum(row["completion_tokens"] for row in rows)
    return {"verified": verified, "idle_before_after": idle, "request_counter_delta": requests,
            "completion_token_counter_delta": tokens,
            "backend_identity_stable": stable,
            "response_ids_unique": len({row["response_id"] for row in rows}) == len(rows)}


def decode_purity(engine, rows, before, after, isolated):
    if not isolated:
        return False, "outside-client isolation or counters not verified"
    if len(rows) == 1:
        return True, "isolated single autoregressive request; own prefill precedes first token"
    if engine == "silyx":
        metrics = [r["backend"].get("silyx", {}).get("metrics", {}) for r in rows]
        prefills = [m.get("prefill_completed_batch") for m in metrics]
        decodes = [m.get("first_decode_batch") for m in metrics]
        if all(type(value) is int for value in prefills+decodes):
            if min(decodes) > max(prefills):
                return True, "server batch sequence proves every prompt completed before any decode step"
            return False, "server batch sequence proves prefill interleaved with decode"
        return False, "server lacks batch-sequence metadata; arrival barrier alone is insufficient proof"
    prefill = counter_delta(before, after, "prefill_active_tokens_total")
    if prefill == 0:
        return True, "TensorFold health counters confirm zero timed prefill tokens"
    return False, "TensorFold timed prefill work is nonzero or unreported"


def graph_warmup(engine, before, after):
    counts = []
    for health in (before, after):
        graphs = health.get("silyx_graphs", {}) if engine == "tensorfold" else {}
        value = graphs.get("cold_preparations") if isinstance(graphs, dict) else None
        counts.append(value if type(value) is int and value >= 0 else None)
    first, last = counts
    delta = last-first if first is not None and last is not None and last >= first else None
    return {"graph_cold_preparations_before": first, "graph_cold_preparations_after": last,
            "graph_cold_preparations_delta": delta,
            "graph_warmup_verified": delta == 0 if delta is not None else None,
            "graph_warmup_note": ("timed burst includes lazy graph preparation; screening only" if delta else
                                  "no lazy graph preparation observed during this timed burst") if delta is not None else
                                 "graph preparation telemetry unavailable or counters reset"}


def all_graphs_warm(flags):
    flags = list(flags)
    return False if any(flag is False for flag in flags) else True if flags and all(flag is True for flag in flags) else None


def summarize(engine, rows, before, after):
    cohort = rows[0]
    protected = isolation(engine, before, after, rows)
    pure, proof = decode_purity(engine, rows, before, after,
                                protected["verified"] and protected["response_ids_unique"])
    first, last = min(r["first_visible_s"] for r in rows), max(r["last_visible_s"] for r in rows)
    count = sum(max(0, r["completion_tokens"]-1) for r in rows)
    observed = count/(last-first) if count and last > first else None
    capped = all(row["cap_satisfied"] for row in rows)
    result = {key: cohort[key] for key in ("phase", "workload", "thinking", "concurrency", "repeat", "output_cap")}
    result.update(graph_warmup(engine, before, after))
    result.update({"outside_clients": protected, "cap_satisfied": capped,
                   "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                   "completion_tokens": sum(r["completion_tokens"] for r in rows),
                   "ttft_p50_seconds": quantile([r["ttft_seconds"] for r in rows], .5),
                   "ttft_p95_seconds": quantile([r["ttft_seconds"] for r in rows], .95),
                   "pure_decode_verified": pure, "decode_purity_evidence": proof,
                   "raw_decode_aggregate_tps": observed if pure and capped and cohort["phase"] == "decode" else None,
                   "service_window_mixed_or_unverified_tps": observed if not pure and cohort["phase"] == "decode" else None,
                   "observed_stream_interval_tps": [r["observed_stream_interval_tps"] for r in rows],
                   "raw_decode_observer": "client first/last visible SSE delta; usage minus one token per stream"})
    metrics = [row["backend"].get("silyx", {}).get("metrics", {}) for row in rows]
    timestamps = [(m.get("first_token_monotonic_s"), m.get("last_token_monotonic_s")) for m in metrics]
    if all(isinstance(value, (float, int)) for pair in timestamps for value in pair):
        span = max(pair[1] for pair in timestamps)-min(pair[0] for pair in timestamps)
        result["backend_observed_interval_tps"] = count/span if count and span > 0 else None
        result["backend_observed_interval_is_pure"] = pure
    if engine == "tensorfold":
        start, end = before.get("decode_active", {}), after.get("decode_active", {})
        seconds = sum(value["seconds"]-start.get(key, {}).get("seconds", 0) for key, value in end.items()) if end else None
        tokens = sum(value["tokens"]-start.get(key, {}).get("tokens", 0) for key, value in end.items()) if end else None
        result["backend_active_round_tps"] = tokens/seconds if seconds else None
        result["backend_active_round_note"] = "excludes scheduling gaps; distinct from client full-burst decode rate"
        drafted, accepted, rounds = (counter_delta(before, after, key)
                                    for key in ("drafted_total", "accepted_total", "rounds_total"))
        global_rounds = [counter_delta(start.get(key, {"rounds": 0}), value, "rounds") for key, value in end.items()]
        global_rounds = sum(global_rounds) if global_rounds and all(value is not None for value in global_rounds) else None
        verified_rows = drafted+rounds if drafted is not None and rounds is not None else None
        result.update({"drafted_tokens": drafted, "accepted_draft_tokens": accepted,
                       "request_decode_rounds": rounds, "global_decode_rounds": global_rounds,
                       "draft_acceptance_fraction": accepted/drafted if drafted and accepted is not None else None,
                       "verified_rows_total": verified_rows,
                       "mean_verified_rows_per_global_round": verified_rows/global_rounds if verified_rows is not None and global_rounds else None,
                       "mean_output_tokens_per_request_round": count/rounds if rounds else None,
                       "draft_counters_isolated": protected["verified"] and protected["response_ids_unique"]})
        reported = [row["backend"].get("tensorfold", {}) for row in rows]
        policies = [item.get("policy") for item in reported]
        copies = [item.get("copy_rounds") for item in reported]
        result.update({"tf_policy_requested": cohort.get("tf_policy"),
                       "tf_policies_reported": sorted({policy for policy in policies if isinstance(policy, str)}),
                       "tf_policy_honored": all(policy == cohort["tf_policy"] for policy in policies)
                       if cohort.get("tf_policy") is not None and all(isinstance(policy, str) for policy in policies) else None,
                       "copy_rounds_total": sum(copies) if all(type(value) is int for value in copies) else None})
    if cohort["phase"] == "prefill":
        if engine == "silyx":
            prefill_before = before.get("runtime_observations", {}).get("completed_prefill_input_tokens")
            prefill_after = after.get("runtime_observations", {}).get("completed_prefill_input_tokens")
            cold = (type(prefill_before) is int and type(prefill_after) is int
                    and prefill_after-prefill_before == sum(r["prompt_tokens"] for r in rows))
        else:
            cold = (counter_delta(before, after, "cached_tokens_total") == 0
                    and (counter_delta(before, after, "prefill_active_tokens_total") or 0)
                    >= sum(r["prompt_tokens"] for r in rows))
        wall = max(r["first_visible_s"] for r in rows)-min(r["request_started_s"] for r in rows)
        result["cold_prefill_verified"] = cold and protected["verified"]
        result["effective_prefill_input_tps_observed"] = sum(r["prompt_tokens"] for r in rows)/wall if wall > 0 else None
        result["effective_cold_prefill_input_tps"] = sum(r["prompt_tokens"] for r in rows)/wall if cold and protected["verified"] and wall > 0 else None
        result["prefill_scope"] = "client request-to-first-token window; includes scheduling and first output token, cap=1"
    return result


def health_schema(health):
    return {"declared_version": health.get("schema_version"), "backend": health.get("backend"),
            "keys_sha256": digest(json.dumps(sorted(health), separators=(",", ":")))}


def case_key(row):
    return tuple(row.get(k) for k in ("phase", "workload", "thinking", "concurrency", "repeat", "stream",
                                     "output_cap", "prompt_sha256"))


def aggregate_runs(summaries, requests):
    keys = ("phase", "workload", "thinking", "concurrency", "output_cap")
    groups = {}
    for summary in summaries:
        groups.setdefault(tuple(summary[k] for k in keys), []).append(summary)
    result = []
    for key, bursts in groups.items():
        rows = [row for row in requests if tuple(row[k] for k in keys) == key]
        raw = [burst["raw_decode_aggregate_tps"] for burst in bursts if burst.get("raw_decode_aggregate_tps") is not None]
        mixed = [burst["service_window_mixed_or_unverified_tps"] for burst in bursts
                 if burst.get("service_window_mixed_or_unverified_tps") is not None]
        cold = [burst["effective_cold_prefill_input_tps"] for burst in bursts
                if burst.get("effective_cold_prefill_input_tps") is not None]
        result.append({**dict(zip(keys, key)), "repeats": len(bursts), "requests": len(rows),
                       "graph_warmup_verified": all_graphs_warm(burst.get("graph_warmup_verified") for burst in bursts),
                       "graph_warmup_verified_repeats": sum(burst.get("graph_warmup_verified") is True for burst in bursts),
                       "graph_cold_preparation_repeats": sum(burst.get("graph_warmup_verified") is False for burst in bursts),
                       "graph_warmup_unknown_repeats": sum(burst.get("graph_warmup_verified") is None for burst in bursts),
                       "graph_cold_preparations_delta_by_repeat": [burst.get("graph_cold_preparations_delta") for burst in bursts],
                       "pure_capped_decode_repeats": len(raw),
                       "raw_decode_aggregate_tps_median": statistics.median(raw) if raw else None,
                       "raw_decode_aggregate_tps_best": max(raw) if raw else None,
                       "mixed_or_unverified_tps_median": statistics.median(mixed) if mixed else None,
                       "effective_cold_prefill_input_tps_median": statistics.median(cold) if cold else None,
                       "ttft_p50_seconds": quantile([row["ttft_seconds"] for row in rows], .5),
                       "ttft_p95_seconds": quantile([row["ttft_seconds"] for row in rows], .95),
                       "baseline_token_comparisons": sum(row.get("baseline_token_hash_match") is not None for row in rows),
                       "baseline_token_matches": sum(row.get("baseline_token_hash_match") is True for row in rows)})
    return result


def make_case(args, factory, *, phase, workload, thinking, concurrency, repeat, stream, run_id):
    nonce = digest(f"{args.suite_id}:{phase}:{workload}:{thinking}:{concurrency}:{repeat}:{stream}")[:12]
    if phase == "prefill":
        instruction = "Read the following benchmark context. Reply with OK only after reading it."
        inputs, cap = args.prefill_input_tokens, 1
    else:
        instruction = (PROSE if workload == "prose" else STRUCTURED)[(stream+repeat) % 8]
        instruction += " Give concrete details and continue until the response limit."
        inputs, cap = args.decode_input_tokens if factory.tokenizer else None, args.output_tokens
    content = f"Benchmark case {nonce}. {instruction}"
    prepared, prompt_hash, expected = factory.prepare(content, thinking, inputs)
    salt = f"silyx-serving:{run_id}:{phase}:{workload}:{thinking}:{concurrency}:{repeat}:{stream}"
    body = {"model": args.model, **prepared, "max_tokens": cap, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "cache_salt": salt}
    if args.engine == "silyx":
        body["thinking"] = thinking == "high"
        if args.silyx_metrics:
            body["stream_options"]["include_silyx_metrics"] = True
    else:
        body["chat_template_kwargs"] = {"thinking": thinking == "high", "enable_thinking": thinking == "high"}
        if thinking == "high":
            body["chat_template_kwargs"]["reasoning_effort"] = "high"
            body["reasoning_effort"] = "high"
        if getattr(args, "tf_policy", None) is not None:
            body["tf_policy"] = args.tf_policy
        if getattr(args, "no_copy", False):
            body["tf_copy"] = False
    item = {"phase": phase, "workload": workload, "thinking": thinking,
            "concurrency": concurrency, "repeat": repeat, "stream": stream, "output_cap": cap,
            "prompt_sha256": prompt_hash, "expected_prompt_tokens": expected,
            "prompt_hash_scope": "expected complete server chat prompt" if factory.tokenizer else "user content only",
            "cache_salt": salt}
    if args.engine == "tensorfold":
        item.update(tf_policy=getattr(args, "tf_policy", None), tf_copy=False if getattr(args, "no_copy", False) else None)
    return item, body


def prefix_seed_bodies(args, bodies):
    """Policy0 can resume a neural seed; its own seed lacks the drafter snapshot.

    Keep draft enabled: this pinned TensorFold version excludes draft:false
    requests from prefix reuse, which would mix prefill into C8 serial timing.
    """
    policy = getattr(args, "tf_seed_policy", None)
    if policy is None and getattr(args, "tf_policy", None) == "0":
        policy = "fnc7:0.3"
    return [{**body, "max_tokens": 1, **({"tf_policy": policy} if policy is not None else {})} for body in bodies]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("silyx", "tensorfold"))
    parser.add_argument("--base", default="http://127.0.0.1:11002")
    parser.add_argument("--model", required=True, help="exact /v1/models ID")
    parser.add_argument("--target-model", default="nvidia/GLM-5.3-Flash-FP4-Local")
    parser.add_argument("--model-revision")
    parser.add_argument("--backend-revision")
    parser.add_argument("--tokenizer", help="local model directory; required for fixed-token prefill")
    parser.add_argument("--concurrency", default="1,8")
    parser.add_argument("--thinking", default="off,high")
    parser.add_argument("--workloads", default="prose,structured")
    parser.add_argument("--phases", default="decode,prefill")
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--decode-input-tokens", type=int, default=128)
    parser.add_argument("--prefill-input-tokens", type=int)
    parser.add_argument("--quick", action="store_true", help="default cap32, repeat1, prefill256")
    parser.add_argument("--warm-prefix", action="store_true", help="TensorFold decode only; verify zero timed prefill")
    parser.add_argument("--tf-policy", help="TensorFold request policy, e.g. 0, f1, fnc3:0.3, fnc7:0.3 or fa")
    parser.add_argument("--tf-seed-policy", help="prefix seed policy; policy0 defaults to neural fnc7:0.3")
    parser.add_argument("--no-copy", action="store_true", help="TensorFold tf_copy:false; retain neural drafting and prefix reuse")
    parser.add_argument("--silyx-metrics", action="store_true", help="request opt-in Silyx batch/timing/hash extension")
    parser.add_argument("--baseline", type=Path, help="previous requests.jsonl for matched prompt/token-hash checks")
    parser.add_argument("--suite-id", default="glm53-serving-v1")
    parser.add_argument("--api-key-env", default="SILYX_API_KEY")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true", help="write the plan; contact no model API")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.output_tokens = args.output_tokens if args.output_tokens is not None else (32 if args.quick else 128)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.quick else 3)
    args.prefill_input_tokens = args.prefill_input_tokens if args.prefill_input_tokens is not None else (256 if args.quick else 2048)
    args.concurrency = [int(c) for c in args.concurrency.split(",")]
    args.thinking, args.workloads, args.phases = (getattr(args, key).split(",") for key in ("thinking", "workloads", "phases"))
    if (not args.concurrency or len(set(args.concurrency)) != len(args.concurrency)
            or any(c < 1 or c > 8 for c in args.concurrency)):
        parser.error("concurrency must contain distinct values 1–8")
    if (set(args.thinking)-{"off", "high"} or set(args.workloads)-{"prose", "structured"}
            or set(args.phases)-{"decode", "prefill"}):
        parser.error("invalid thinking, workload or phase")
    if min(args.output_tokens, args.repeats, args.prefill_input_tokens, args.decode_input_tokens) < 1:
        parser.error("token counts and repeats must be positive")
    if "decode" in args.phases and args.output_tokens < 2:
        parser.error("decode needs at least two output tokens")
    if "prefill" in args.phases and not args.tokenizer:
        parser.error("fixed-token prefill requires --tokenizer; or use --phases decode")
    if args.warm_prefix and args.engine != "tensorfold":
        parser.error("warm-prefix is currently supported only for TensorFold")
    if args.engine != "tensorfold" and (args.tf_policy is not None or args.tf_seed_policy is not None or args.no_copy):
        parser.error("TensorFold policy/copy controls require --engine tensorfold")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be new or empty")
    endpoint = Endpoint(args.base, os.environ.get(args.api_key_env), args.timeout)
    args.out.mkdir(parents=True, exist_ok=True)
    factory = PromptFactory(args.tokenizer)
    run_id = uuid.uuid4().hex
    plan = []
    for phase in args.phases:
        for workload in args.workloads if phase == "decode" else ("prefill",):
            for thinking in args.thinking:
                for concurrency in args.concurrency:
                    for repeat in range(args.repeats):
                        pairs = [make_case(args, factory, phase=phase, workload=workload, thinking=thinking,
                                           concurrency=concurrency, repeat=repeat, stream=i, run_id=run_id)
                                 for i in range(concurrency)]
                        plan.append(pairs)
    configuration = {key: value for key, value in vars(args).items() if key not in ("tokenizer", "baseline", "out")}
    metadata = {"schema_version": SCHEMA, "run_id": run_id, "configuration": configuration,
                "tokenizer_files_sha256": factory.files, "planned_bursts": len(plan),
                "planned_requests": sum(len(pairs) for pairs in plan),
                "decode_scope": "usage minus one per stream over first/last visible deltas; pure only with explicit evidence",
                "accuracy_scope": "response/token-hash reproducibility checks, not a general quality evaluation"}
    (args.out/"run.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (args.out/"plan.json").write_text(json.dumps([[item for item, _ in pairs] for pairs in plan], indent=2)+"\n")
    if args.prepare_only:
        print(json.dumps({"prepared": len(plan), "api_requests": 0}))
        return
    models = endpoint.get("/v1/models")
    if args.model not in {item["id"] for item in models.get("data", [])}:
        raise RuntimeError("requested model ID is not loaded; refusing to benchmark another model")
    initial = endpoint.get("/health")
    metadata["initial_health_schema"] = health_schema(initial)
    (args.out/"run.json").write_text(json.dumps(metadata, indent=2)+"\n")
    baseline = {}
    if args.baseline:
        for line in args.baseline.read_text().splitlines():
            row = json.loads(line)
            baseline[case_key(row)] = row
    summaries, all_rows = [], []
    with (args.out/"requests.jsonl").open("w") as records:
        for index, pairs in enumerate(plan):
            items, bodies = zip(*pairs)
            if args.warm_prefix and items[0]["phase"] == "decode":
                seed_items = [{**item, "output_cap": 1, "timed": False, "role": "prefix_seed"} for item in items]
                seeds = burst(endpoint, seed_items, prefix_seed_bodies(args, bodies))
                with (args.out/"prefix-seeds.jsonl").open("a") as output:
                    for row in seeds:
                        output.write(json.dumps(row)+"\n")
            before = endpoint.get("/health")
            rows = burst(endpoint, items, bodies)
            after = endpoint.get("/health")
            (args.out/f"health-{index:03d}.json").write_text(json.dumps(
                {"schema_version": SCHEMA, "before": before, "after": after,
                 "health_schema_before": health_schema(before), "health_schema_after": health_schema(after)}, indent=2)+"\n")
            for row in rows:
                reference = baseline.get(case_key(row))
                row["baseline_prompt_matched"] = reference is not None if baseline else None
                a, b = row.get("token_hash"), reference.get("token_hash") if reference else None
                row["baseline_token_hash_match"] = a[:min(len(a), len(b))] == b[:min(len(a), len(b))] if a and b else None
                records.write(json.dumps(row)+"\n")
            records.flush()
            all_rows.extend(rows)
            summary = summarize(args.engine, rows, before, after)
            summaries.append(summary)
            (args.out/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
            flat = [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                     for key, value in item.items()} for item in summaries]
            fields = list(dict.fromkeys(key for item in flat for key in item))
            with (args.out/"summary.csv").open("w") as output:
                writer = csv.DictWriter(output, fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(flat)
            print(json.dumps(summary), flush=True)
    (args.out/"aggregate.json").write_text(json.dumps(aggregate_runs(summaries, all_rows), indent=2)+"\n")
    (args.out/"complete.json").write_text(json.dumps(
        {"schema_version": SCHEMA, "completed_unix_s": time.time(), "bursts": len(summaries),
         "all_capped": all(item["cap_satisfied"] for item in summaries),
         "all_isolated": all(item["outside_clients"]["verified"] for item in summaries),
         "graph_warmup_verified": all_graphs_warm(item.get("graph_warmup_verified") for item in summaries),
         "graph_cold_preparation_bursts": sum(item.get("graph_warmup_verified") is False for item in summaries),
         "graph_warmup_unknown_bursts": sum(item.get("graph_warmup_verified") is None for item in summaries),
         "pure_decode_bursts": sum(item["phase"] == "decode" and item["pure_decode_verified"] for item in summaries)}, indent=2)+"\n")


if __name__ == "__main__":
    main()
