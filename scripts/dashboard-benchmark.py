"""Run the installed port-11001 dashboard benchmark functions as isolated jobs.

The supplied features.py remains an external dependency. Its prompt catalogs,
streaming protocol and BenchManager come from the local Anemll dashboard with
MiaAI sparkDash lineage; this script records their source hash and actual usage.
No model service, model selection or dashboard state is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
import urllib.request


def get(base, route):
    with urllib.request.urlopen(base.rstrip('/')+route, timeout=10) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--base', default='http://127.0.0.1:8888')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--repeats', type=int, default=2, choices=(1,2,3))
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_sha=hashlib.sha256(args.features.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('dashboard_features',args.features)
    features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)
    features._METRICS_URL=args.base.rstrip('/')+'/metrics'
    seen={m['id']for m in get(args.base,'/v1/models')['data']}
    if args.model not in seen:raise RuntimeError('requested DeepSeek model is not loaded')
    manifest=dict(schema=1,date=time.strftime('%Y-%m-%d',time.gmtime()),model=args.model,
                  features_sha256=source_sha,backend='TensorFold',thinking=False,
                  temperature=0,output_cap=400,concurrencies=[1,2,4,8],
                  prefill_targets=[1024,4096,16384,65536],repeats=args.repeats,
                  separate_job_order=['code','prose','structured','prefill'],
                  metric='unchanged dashboard BenchManager observed streaming rate',
                  source='local Anemll dashboard/features.py; MiaAI sparkDash prompt/protocol lineage',
                  status='running')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    class AuditedClient(features.LlmClient):
        def __init__(self):
            super().__init__(args.base)
            self.records=[]
            self.lock=threading.Lock()

        def stream_chat(self,model,prompt,max_tokens,temperature,thinking,timeout,cancel=None):
            record=dict(model=model,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        output_cap=max_tokens,thinking=thinking,started_s=time.monotonic(),usage={},error=None)
            pieces=[]
            try:
                for kind,value in super().stream_chat(model,prompt,max_tokens,temperature,thinking,timeout,cancel):
                    if kind=='usage':record['usage']=value
                    else:pieces.append(value)
                    yield kind,value
            except BaseException as error:
                record['error']=type(error).__name__+': '+str(error)
                raise
            finally:
                record['ended_s']=time.monotonic()
                record['output_text']=''.join(pieces)
                record['output_text_sha256']=hashlib.sha256(record['output_text'].encode()).hexdigest()
                with self.lock:self.records.append(record)

    manager=features.BenchManager(args.out/'latest.json')
    try:
        for kind in manifest['separate_job_order']:
            for repeat in range(args.repeats):
                before=get(args.base,'/health')
                if not before.get('ok')or before.get('busy')or before.get('requests_running'):
                    raise RuntimeError('service must be healthy and idle before each separate job')
                client=AuditedClient()
                job=(manager.start_prefill(client,args.model,manifest['prefill_targets'],0)
                     if kind=='prefill'else manager.start_decode(client,args.model,kind,[1,2,4,8],400,0))
                print(json.dumps(dict(workload=kind,repeat=repeat,status='running',job_id=job['id'])),flush=True)
                deadline=time.monotonic()+3600
                while job['status']=='running'and time.monotonic()<deadline:
                    time.sleep(1)
                    job=manager.get('prefill'if kind=='prefill'else'decode',job['id'])
                if job['status']=='running':
                    manager.cancel(job['kind'],job['id'])
                    raise TimeoutError('dashboard benchmark job exceeded one hour')
                after=get(args.base,'/health')
                records=sorted(client.records,key=lambda r:r['started_s'])
                complete=all(r['error']is None and type(r['usage'].get('completion_tokens'))is int
                             and type(r['usage'].get('prompt_tokens'))is int for r in records)
                expected=4 if kind=='prefill' else 16
                isolation=dict(usage_complete=complete,expected_requests=expected,recorded_requests=len(records),
                    request_counter_delta=after['requests_total']-before['requests_total'],
                    output_counter_delta=after['completion_tokens_total']-before['completion_tokens_total'],
                    recorded_output_tokens=sum(r['usage'].get('completion_tokens',0)for r in records),
                    idle_after=not after.get('busy')and not after.get('requests_running'))
                isolation['verified']=(complete and len(records)==expected and isolation['request_counter_delta']==expected
                    and isolation['output_counter_delta']==isolation['recorded_output_tokens']and isolation['idle_after'])
                result=dict(workload=kind,repeat=repeat,job=job,isolation=isolation,health_before=before,health_after=after)
                (args.out/f'{kind}-r{repeat}.json').write_text(json.dumps(result,indent=2)+'\n')
                with(args.out/f'{kind}-r{repeat}-requests.jsonl').open('w')as file:
                    for record in records:file.write(json.dumps(record)+'\n')
                if job['status']!='done'or not isolation['verified']:
                    raise RuntimeError('job failed or outside-client isolation/usage failed')
                values=job['results']if kind=='prefill'else job['levels']
                if any(value.get('error')or value.get('streamsFailed',0)for value in values):
                    raise RuntimeError('benchmark contains failed requests')
                print(json.dumps(dict(workload=kind,repeat=repeat,status='done',isolation=isolation,
                                      results=values)),flush=True)
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
