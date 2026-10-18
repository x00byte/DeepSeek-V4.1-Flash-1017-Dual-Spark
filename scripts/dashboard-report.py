"""Summarize separate dashboard runs only after every usage/isolation gate passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('complete suite required')
    repeats=manifest['repeats']
    groups={kind:[]for kind in manifest['separate_job_order']}
    total_requests=total_output=0
    for kind in groups:
        for repeat in range(repeats):
            result=json.loads((root/f'{kind}-r{repeat}.json').read_text())
            if result['job']['status']!='done'or not result['isolation']['verified']:
                raise ValueError('successful isolated jobs required')
            records=[json.loads(line)for line in (root/f'{kind}-r{repeat}-requests.jsonl').read_text().splitlines()]
            before,after=result['health_before'],result['health_after']
            if len(records)!=(4 if kind=='prefill'else 16):raise ValueError('request count mismatch')
            for record in records:
                for field in ('prompt','output_text'):
                    if hashlib.sha256(record[field].encode()).hexdigest()!=record[field+'_sha256']:
                        raise ValueError('request text digest mismatch')
                if record['error']or record['model']!=manifest['model']or record['thinking']is not False:
                    raise ValueError('request configuration or completion failed')
            if (after['requests_total']-before['requests_total']!=len(records)
                or after['completion_tokens_total']-before['completion_tokens_total']
                    !=sum(r['usage']['completion_tokens']for r in records)
                or after['prompt_tokens_total']-before['prompt_tokens_total']
                    !=sum(r['usage']['prompt_tokens']for r in records)
                or after['errors']!=before['errors']):
                raise ValueError('request/usage/error counters mismatch')
            groups[kind].append(result)
            total_requests+=result['isolation']['recorded_requests']
            total_output+=result['isolation']['recorded_output_tokens']
    decode=[]
    for kind in ('code','prose','structured'):
        row=dict(workload=kind,concurrency={})
        for concurrency in manifest['concurrencies']:
            values=[]
            for result in groups[kind]:
                value=next(v for v in result['job']['levels']if v['concurrency']==concurrency)
                if value['streamsFailed']or value['streamsOk']!=concurrency:
                    raise ValueError('all requested streams must complete')
                values.append(value)
            row['concurrency'][str(concurrency)]=dict(
                median_output_tps=statistics.median(v['aggregateDecodeTps']for v in values),
                min_output_tps=min(v['aggregateDecodeTps']for v in values),
                max_output_tps=max(v['aggregateDecodeTps']for v in values),
                median_mean_ttft_ms=statistics.median(v['meanTtftMs']for v in values),
                prompt_tokens=sorted({s['promptTokens']for v in values for s in v['perStream']}),
                output_tokens=sorted({s['tokens']for v in values for s in v['perStream']}))
        decode.append(row)
    prefill=[]
    cold_proofs=[]
    for result in groups['prefill']:
        before,after=result['health_before'],result['health_after']
        a,b=before.get('deepseek_phase',{}),after.get('deepseek_phase',{})
        count=sum(v['promptTokens']for v in result['job']['results'])
        cold=(a.get('session_store_disabled')is True and b.get('session_store_disabled')is True
              and after['cached_tokens_total']-before['cached_tokens_total']==0
              and b['prefill_tokens']-a['prefill_tokens']==count-4
              and b['prefill_finalizations']-a['prefill_finalizations']==4)
        cold_proofs.append(cold)
    if not all(cold_proofs):raise ValueError('cold-prefill counters failed')
    for target in manifest['prefill_targets']:
        values=[next(v for v in r['job']['results']if v['targetTokens']==target)for r in groups['prefill']]
        if any(v['error']or v['prefillTps']<=0 for v in values):raise ValueError('valid prefill measurements required')
        prefill.append(dict(target_tokens=target,actual_prompt_tokens=sorted({v['promptTokens']for v in values}),
            median_ttft_ms=statistics.median(v['ttftMs']for v in values),
            median_input_tps=statistics.median(v['prefillTps']for v in values),
            min_input_tps=min(v['prefillTps']for v in values),max_input_tps=max(v['prefillTps']for v in values)))
    return dict(schema=1,date=manifest['date'],model=manifest['model'],backend='TensorFold',
                thinking=False,temperature=0,output_cap=400,repeats=repeats,
                source_features_sha256=manifest['features_sha256'],separate_jobs=4*repeats,
                total_requests_including_warmup=total_requests,total_output_tokens_including_warmup=total_output,
                isolation_verified=True,cold_prefill_verified=True,decode=decode,prefill=prefill,
                decode_scope='dashboard observed aggregate: all completion tokens divided by earliest-first to latest-last visible SSE; may include interleaved prefill at concurrency>1',
                prefill_scope='C1 actual input tokens divided by request-to-first-visible-token latency, output cap1; unique prefix per size',
                limitations=[f'{repeats} passes on one loaded profile','400-token output cap; no sustained or quality claim',
                             'structured means counting1..200, without a schema constraint','prefill targets are approximate; actual counts are authoritative'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    summary=summarize(args.results)
    (args.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
