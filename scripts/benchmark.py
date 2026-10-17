"""Identical streamed C1..C8 workloads; records effective prefill, TTFT and decode.

Never switches models. Run once per engine after selecting it on the dashboard.
Effective prefill includes scheduling and first-token work; it is not a kernel-only measurement.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time
import urllib.request

def get_json(url):
    with urllib.request.urlopen(url,timeout=15) as response: return json.load(response)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--base',default='http://127.0.0.1:8888')
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--thinking',choices=['off','75'],default='off')
    ap.add_argument('--concurrency',default='1,2,3,4,5,6,7,8')
    ap.add_argument('--skip-warmup',action='store_true')
    ap.add_argument('--input-lines',type=int,default=192)
    ap.add_argument('--output-tokens',type=int,default=512)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tensorfold-log',type=Path)
    a=ap.parse_args()
    if a.repeats<1 or a.input_lines<1 or a.output_tokens<1: ap.error('Positive sizes required')
    levels=[int(x) for x in a.concurrency.split(',')]
    if not levels or any(x<1 or x>8 for x in levels) or len(levels)!=len(set(levels)): ap.error('Concurrency must contain distinct values from 1 to 8')
    ids={x['id'] for x in get_json(a.base+'/v1/models')['data']}
    if a.model not in ids: raise SystemExit(f'Refusing to benchmark a different model: serving {sorted(ids)}')
    a.out.mkdir(parents=True,exist_ok=True)
    run_started=time.time();run_origin=time.perf_counter()
    (a.out/'run.json').write_text(json.dumps({**vars(a),'out':str(a.out),'tensorfold_log':str(a.tensorfold_log) if a.tensorfold_log else None,'serving':sorted(ids),'started':time.time(),
        'notes':'Client effective prefill includes scheduling and first-token work; decode uses completion usage minus one over first-to-last streamed delta. Speculative decoding may return multiple tokens in a delta. Backend metrics are saved separately.'},indent=2))
    def metrics(name):
        try:
            with urllib.request.urlopen(a.base+'/metrics',timeout=10) as response: (a.out/name).write_bytes(response.read())
        except OSError: pass
    def request(c,r,i,barrier):
        rng=random.Random(f'gx10-1017-v1:{c}:{r}:{i}')
        prompt=f'Experiment {c:02d}-{r:02d}-{i:02d}. Read this ledger and then list its entries in reverse order, explaining each one. Continue until the response limit.\n'
        prompt+='\n'.join(f'Entry {j:04d}: alpha {rng.randrange(100000,999999)} beta {rng.randrange(100000,999999)} gamma {rng.randrange(100000,999999)}.' for j in range(a.input_lines))
        body={'model':a.model,'messages':[{'role':'user','content':prompt}], 'temperature':0,
              'max_tokens':min(64,a.output_tokens) if r<0 else a.output_tokens,'stream':True,'stream_options':{'include_usage':True},
              'chat_template_kwargs':{'thinking':a.thinking=='75','enable_thinking':a.thinking=='75','reasoning_effort':75}}
        if a.thinking=='75': body['reasoning_effort']='high'
        req=urllib.request.Request(a.base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        barrier.wait();start=time.perf_counter();first=last=None;usage=None;finish=None;chunks=0;extra={}
        with urllib.request.urlopen(req,timeout=1800) as response:
            for raw in response:
                if not raw.startswith(b'data:'): continue
                data=raw[5:].strip()
                if data==b'[DONE]': break
                event=json.loads(data)
                if 'error' in event: raise RuntimeError(event['error'])
                if event.get('usage'): usage=event['usage']
                for key in ('timings','stats'):
                    if key in event: extra[key]=event[key]
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',{})
                    if delta.get('content') or delta.get('reasoning_content'):
                        now=time.perf_counter(); first=now if first is None else first;last=now;chunks+=1
                    finish=choice.get('finish_reason') or finish
        end=time.perf_counter()
        if not usage or first is None or last is None: raise RuntimeError('Missing token usage or streamed text; no estimated counts accepted')
        n=usage['completion_tokens'];p=usage['prompt_tokens'];ttft=first-start;decode=last-first
        return {'label':a.label,'thinking':a.thinking,'input_lines':a.input_lines,'output_cap':a.output_tokens,'concurrency':c,'repeat':r,'stream':i,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'prompt_tokens':p,'completion_tokens':n,'ttft_s':ttft,'total_s':end-start,'decode_s':decode,
                'prefill_effective_tps':p/ttft,'decode_stream_tps':(n-1)/decode if decode>0 and n>1 else None,
                'first_relative_s':first-run_origin,'last_relative_s':last-run_origin,
                'finish_reason':finish,'chunks':chunks,'backend':extra}
    metrics('metrics-before.txt');summaries=[]
    with (a.out/'requests.jsonl').open('w') as output:
        for c in levels:
            if not a.skip_warmup:
                barrier=threading.Barrier(c)
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
                    warmups=list(ex.map(lambda i:request(c,-1,i,barrier),range(c)))
                for row in warmups: output.write(json.dumps({**row,'warmup':True})+'\n')
                output.flush()
            for r in range(a.repeats):
                metrics(f'c{c}-r{r}-metrics-before.txt')
                barrier=threading.Barrier(c);start=time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
                    rows=list(ex.map(lambda i:request(c,r,i,barrier),range(c)))
                elapsed=time.perf_counter()-start
                metrics(f'c{c}-r{r}-metrics-after.txt')
                for row in rows: output.write(json.dumps(row)+'\n')
                output.flush()
                summary={'label':a.label,'thinking':a.thinking,'input_lines':a.input_lines,'output_cap':a.output_tokens,'concurrency':c,'repeat':r,'prompt_tokens':sum(x['prompt_tokens'] for x in rows),
                    'completion_tokens':sum(x['completion_tokens'] for x in rows),'batch_s':elapsed,
                    'output_wall_tps':sum(x['completion_tokens'] for x in rows)/elapsed,
                    'ttft_median_s':statistics.median(x['ttft_s'] for x in rows),
                    'prefill_effective_aggregate_tps':sum(x['prompt_tokens'] for x in rows)/max(x['ttft_s'] for x in rows),
                    'decode_observed_aggregate_tps':(sum(x['completion_tokens'] for x in rows)-c)/(max(x['last_relative_s'] for x in rows)-min(x['first_relative_s'] for x in rows)) if a.output_tokens>1 and max(x['last_relative_s'] for x in rows)>min(x['first_relative_s'] for x in rows) else None,
                    'prefill_effective_median_tps':statistics.median(x['prefill_effective_tps'] for x in rows),
                    'decode_stream_median_tps':statistics.median(x['decode_stream_tps'] for x in rows if x['decode_stream_tps'] is not None) if any(x['decode_stream_tps'] is not None for x in rows) else None}
                summaries.append(summary);print(json.dumps(summary),flush=True)
                with (a.out/'summary.csv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=summary.keys());writer.writeheader();writer.writerows(summaries)
    metrics('metrics-after.txt')
    log=a.tensorfold_log
    if a.model.endswith('-tensorfold') and log and log.exists():
        with (a.out/'backend-requests.jsonl').open('w') as f:
            for line in log.read_text().splitlines():
                record=json.loads(line)
                if record.get('start',0)>=run_started and record.get('messages')==1 and record.get('max_tokens') in {a.output_tokens,min(64,a.output_tokens)}: f.write(line+'\n')

if __name__=='__main__': main()
