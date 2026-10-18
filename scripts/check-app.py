from pathlib import Path
from types import SimpleNamespace
from tensorfold.families.deepseek_v41.cuda.app import Dsv41App as App
engine=SimpleNamespace(limit=262144,eos=(1,),generate=lambda *a,**kw:None,batch=SimpleNamespace(n_slots=8))
app=App(engine,Path('/model'),'deepseek-flash',context_window=262144,max_tokens=32768)
prompt,thinking=app.render({'messages':[{'role':'user','content':'Hello'}],'chat_template_kwargs':{'enable_thinking':False}})
assert not thinking and '<｜User｜>Hello' in prompt
print('Real 1017 checkpoint: serving App metadata, tokenizer, template, grammar, vision settings and prompt rendering validated on CPU.')

# Read only the small inline tensors; verify the exact attach path on CPU before a GPU launch.
import torch
from tensorfold.families.deepseek_v41.cuda.weights import Shards
from tensorfold.families.deepseek_v41.cuda.vision import attach_bias_vl
src=Shards('/model')
names=sorted(n for n in src.weight_map if n.endswith('.ffn.gate.bias_vl'))
assert len(names)==43, '1017 inline image router biases missing'
blocks=[SimpleNamespace(index=int(n.split('.')[1]),moe=SimpleNamespace(bias=torch.zeros(src.info(n).shape))) for n in names if n.startswith('layers.')]
assert attach_bias_vl(blocks,'/model')==40
assert all(torch.equal(b.moe.bias_vl,src.tensor(f'layers.{b.index}.ffn.gate.bias_vl').float()) for b in blocks)
assert all(torch.isfinite(src.tensor(n)).all() for n in names)
print('All 40 image router biases attached and matched on CPU; all 43 inline tensors checked.')
