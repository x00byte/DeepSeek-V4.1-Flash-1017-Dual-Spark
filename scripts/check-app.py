from pathlib import Path
from types import SimpleNamespace
from tensorfold.families.deepseek_v41.cuda.app import Dsv41App as App
engine=SimpleNamespace(limit=262144,eos=(1,),generate=lambda *a,**kw:None,batch=SimpleNamespace(n_slots=8))
app=App(engine,Path('/model'),'deepseek-v4.1-flash-1017-tensorfold',context_window=262144,max_tokens=32768)
prompt,thinking=app.render({'messages':[{'role':'user','content':'Hello'}],'chat_template_kwargs':{'enable_thinking':False}})
assert not thinking and '<｜User｜>Hello' in prompt
print('Real 1017 checkpoint: serving App metadata, tokenizer, template, grammar, vision settings and prompt rendering validated on CPU.')
