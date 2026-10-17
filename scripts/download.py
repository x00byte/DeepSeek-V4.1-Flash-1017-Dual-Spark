"""Fetch pinned public weights and official 1017 Engrams; verify Hub LFS SHA-256 hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

# Large Engram shards require native hf-xet. Keep its buffers bounded on unified-memory hosts.
os.environ.setdefault('HF_XET_FIXED_DOWNLOAD_CONCURRENCY','4')
os.environ.setdefault('HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS','1')
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES','0')
os.environ.setdefault('HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE','128mb')
os.environ.setdefault('HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE','32mb')
os.environ.setdefault('HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT','512mb')
os.environ.setdefault('HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER','128mb')
os.environ.setdefault('HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE','512mb')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb',buffering=0) as f:
  offset=0
  while block:=f.read(8*1024**2):
   h.update(block)
   if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),offset,len(block),os.POSIX_FADV_DONTNEED)
   offset+=len(block)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,required=True)
 a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 recipe=Path(__file__).resolve().parents[1]
 pin=json.loads((recipe/'config/manifest.json').read_text())
 from huggingface_hub import HfApi,snapshot_download
 api=HfApi();proof={}
 (a.root/'.incomplete').touch()
 for kind,sub in [('model','checkpoint'),('engram','engram-src-1017')]:
  item=pin[kind]
  selected=None if kind=='model' else ['config.json','model.safetensors.index.json',*item['files']]
  snapshot_download(item['repo'],revision=item['revision'],local_dir=a.root/sub,
                    allow_patterns=selected,max_workers=4 if kind=='model' else 1)
  metadata={x.rfilename:x for x in api.model_info(item['repo'],revision=item['revision'],files_metadata=True).siblings}
  for name,meta in metadata.items():
   if selected is not None and name not in selected:continue
   path=a.root/sub/name
   assert path.stat().st_size==meta.size,(name,'size')
   if meta.lfs:
    got=sha(path);assert got==meta.lfs.sha256,(name,'sha256')
    if kind=='engram':assert got==item['files'][name]
    proof[sub+'/'+name]={'bytes':meta.size,'sha256':got}
    print('Verified',sub+'/'+name,flush=True)
 # Add the reference-tested Jinja port as local serving metadata, retaining every downloaded shard byte.
 template=recipe/'build/mia-runtime/files/chat_template.jinja'
 if not template.exists():raise SystemExit('Run scripts/fetch-sources.sh before downloading assets.')
 shutil.copyfile(template,a.root/'checkpoint/chat_template.jinja')
 # The file-backed Engram loader only needs these four entries from the official index.
 idx=a.root/'engram-src-1017/model.safetensors.index.json'
 original=json.loads(idx.read_text());(a.root/'engram-official-index.json').write_text(json.dumps(original,indent=2)+'\n')
 original['weight_map']={k:v for k,v in original['weight_map'].items() if k.endswith(('.engram.embed.weight','.engram.embed.scale'))}
 assert len(original['weight_map'])==4
 original['metadata']['dsv41_engram_src']='embed-only'
 idx.write_text(json.dumps(original,indent=2)+'\n')
 (a.root/'verified.json').write_text(json.dumps({'pins':pin,'files':proof},indent=2)+'\n')
 (a.root/'.incomplete').unlink()
 print('Pinned checkpoint and matching 1017 Engrams verified. No GPU service was started.')

if __name__=='__main__':main()
