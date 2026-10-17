"""TensorFold on two Sparks. Reads config/cluster.env; commands affect only NAME."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def load_config():
 path=Path(os.environ.get('CLUSTER_CONFIG',str(ROOT/'config/cluster.env')))
 if not path.is_file():raise SystemExit('Copy config/cluster.env.example to config/cluster.env and edit it.')
 script='set -a; source "$1"; source "$2"; env -0'
 raw=subprocess.check_output(['bash','-c',script,'bash',str(path),str(ROOT/'config/tensorfold-knobs.env.example')])
 return dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)

C=load_config()
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',C['WORKER_SSH']]

def run(args,worker=False,**kw):
 return subprocess.run([*SSH,shlex.join(args)] if worker else args,check=True,**kw)

def docker_args(rank):
 model=C['MODEL_DIR'] if rank==0 else C['WORKER_MODEL']
 engram=C['HEAD_ENGRAM'] if rank==0 else C['WORKER_ENGRAM']
 state=C['HEAD_STATE'] if rank==0 else C['WORKER_STATE']
 args=['docker','run','-d','--name',C['NAME'],'--gpus','all','--network','host','--ipc','host',
       '--device','/dev/infiniband','--ulimit','memlock=-1','--cap-add','IPC_LOCK',
       '--log-opt','max-size=200m','--log-opt','max-file=3','-v',model+':/model:ro',
       '-v',engram+':/engram:ro','-v',state+':/state','-v',C['CACHE_VOL']+':/cache']
 env={k:v for k,v in C.items() if k.startswith(('TF_DSV41_','GLM53_TF_','MIMALLOC_','MALLOC_','NCCL_'))}
 env.update(TORCH_EXTENSIONS_DIR='/cache/torch_extensions',TRITON_CACHE_DIR='/cache/triton',
            CUDA_CACHE_PATH='/cache/nv/ComputeCache',GLOO_SOCKET_IFNAME=C['NCCL_SOCKET_IFNAME'],
            MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1',TF_DSV41_IMAGE_ID=C['IMAGE'])
 for key,value in sorted(env.items()):args+=['-e',key+'='+value]
 args+=['--entrypoint','python',C['IMAGE'],'-m','tensorfold','serve','/model','--tp','2','--rank',str(rank),
        '--master',C['HEAD_IP'],'--master-port',C['MASTER_PORT'],'--context',C['CONTEXT'],
        '--parallel',C['PARALLEL'],'--kv-dtype','fp8','--no-update-check']
 if rank==0:args+=['--host',C['HOST'],'--port',C['PORT'],'--name',C['SERVED_NAME'],'--max-tokens','32768']
 return args

def check():
 bundle=Path(C['MODEL_DIR']).parent
 assert not (bundle/'.incomplete').exists(),'Download or verification is incomplete'
 proof=json.loads((bundle/'verified.json').read_text())
 pin=json.loads((ROOT/'config/manifest.json').read_text())
 assert proof['pins']['model']['revision']==pin['model']['revision']
 assert proof['pins']['engram']['revision']==pin['engram']['revision']
 for name,meta in proof['files'].items():assert (bundle/name).stat().st_size==meta['bytes'],name
 index=json.loads((Path(C['MODEL_DIR'])/'model.safetensors.index.json').read_text())
 assert len(set(index['weight_map'].values()))==43
 assert C['TF_DSV41_BIAS_VL']=='/model', '1017 native vision requires its inline routing biases'
 assert C['PARALLEL']=='8' and C['TF_DSV41_PREFILL']=='full' and C['TF_DSV41_EXPERT_TOPP']=='1.0'
 keys=[]
 for worker in (False,True):
  data=json.loads(run(['docker','image','inspect',C['IMAGE']],worker,capture_output=True,text=True).stdout)[0]
  keys.append((data['Created'],data['RootFS']['Layers']))
  args=['docker','run','--rm','--network','none','--entrypoint','sh','-v',
        (C['MODEL_DIR'] if not worker else C['WORKER_MODEL'])+':/model:ro','-v',
        (C['HEAD_ENGRAM'] if not worker else C['WORKER_ENGRAM'])+':/engram:ro',C['IMAGE'],
        '-c',f'test -s /model/chat_template.jinja && test -s /model/config.json && test -s /model/tokenizer.json && test -s /engram/engram-l1-r{int(worker)}of2.bin && test -s /engram/engram-l14-r{int(worker)}of2.bin']
  run(args,worker)
 assert keys[0]==keys[1],'Ship the same built image to both nodes.'
 run(['docker','run','--rm','--network','none','--memory','2g','--cpus','2',
      '-v',C['MODEL_DIR']+':/model:ro','-v',str(ROOT/'scripts/check-app.py')+':/check.py:ro',
      '--entrypoint','python',C['IMAGE'],'/check.py'])
 print('Both images, rank-local Engram files and real serving metadata checked.')

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('command',choices=['args','check','start','stop','health'])
a=ap.parse_args()
if a.command=='args':
 for rank in (0,1):print(shlex.join(docker_args(rank)))
elif a.command=='check':check()
elif a.command=='health':
 with urllib.request.urlopen('http://127.0.0.1:'+C['PORT']+'/health',timeout=10) as r:print(r.read().decode())
elif a.command=='stop':
 for worker in (False,True):
  try:run(['docker','rm','-f',C['NAME']],worker)
  except subprocess.CalledProcessError as error:
   if error.returncode==255:raise
else:
 check()
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',int(C['PORT'])))==0:raise SystemExit('API port is occupied; stop its owner explicitly.')
 for worker in (False,True):
  pids=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],worker,capture_output=True,text=True).stdout.strip()
  if pids:raise SystemExit('A GPU workload is running; stop it explicitly before starting this profile.')
  run(['mkdir','-p',C['WORKER_STATE'] if worker else C['HEAD_STATE']],worker)
 try:
  run(docker_args(1),True);run(docker_args(0))
 except BaseException:
  for worker in (False,True):
   try:run(['docker','rm','-f',C['NAME']],worker)
   except subprocess.CalledProcessError:pass
  raise
 print('Both ranks started. Model loading and warmup can take several minutes; check /health and both container logs.')
