"""Isolated performance-only benchmark; never supplies behavior labels."""
import json,time,sys,dataclasses
from pathlib import Path
import torch
from .real_run import cfg_for
from .adapters.qwen import load_real_qwen,QwenGenerationBackend,SlotRequest
from .bf16_batch import BatchedQwen
from .runtime import atomic_write_json
r=Path(sys.argv[1]);f=json.loads((r/'protocol/frozen.json').read_text());g=f['calibration'][0]
torch.set_num_threads(2);records=[]
for dtype in ['fp32_sdpa','bf16_sdpa']:
 cfg=cfg_for(r);cfg.server.thinking.numerical_backend=dtype;cfg.server.thinking.max_new_tokens=256
 t=time.monotonic();model,tok=load_real_qwen(cfg);torch.cuda.synchronize();cold=time.monotonic()-t
 back=QwenGenerationBackend(cfg.server.thinking,model,tok);batch=BatchedQwen(cfg.server.thinking,model,tok)
 def req(i):return SlotRequest('performance',f'perf:{dtype}:{i}',g['question'],'',7100+i)
 t=time.monotonic();batch.generate_batch([req(0)],max_tokens=8);torch.cuda.synchronize();warm=time.monotonic()-t
 for mode,n in ([('hf_serial',1)] if dtype=='fp32_sdpa' else [('hf_serial',1),('native_batch',12)]):
  torch.cuda.reset_peak_memory_stats();t=time.monotonic()
  outputs=[back.generate(req(0))] if mode=='hf_serial' else batch.generate_batch([req(i) for i in range(n)],max_tokens=256)
  torch.cuda.synchronize();seconds=time.monotonic()-t
  records.append(dict(dtype=dtype,mode=mode,batch=n,seconds=seconds,tokens=sum(x.generated_tokens for x in outputs),tokens_per_second=sum(x.generated_tokens for x in outputs)/seconds,peak_bytes=torch.cuda.max_memory_allocated(),cold_start_seconds=cold,warmup_seconds=warm,performance_only=True,device_map=str(getattr(model,'hf_device_map',None)),hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()),use_cache=True,offload=False))
  atomic_write_json(r/'performance/benchmark.json',records);print(json.dumps(records[-1]),flush=True)
 if dtype=='bf16_sdpa':
  a=batch.generate_batch([req(0)],max_tokens=8)[0];b=batch.generate_batch([req(0)],max_tokens=8)[0]
  assert a.generated_token_ids==b.generated_token_ids
  count=[0]
  def stop():count[0]+=1;return count[0]>=3
  batch.stop_check=stop
  c=batch.generate_batch([req(0)],max_tokens=8)[0];assert c.termination_reason=='external_stop' and c.generated_tokens==2
  atomic_write_json(r/'performance/validation.json',{'same_batch_seed_repeat':True,'mid_generation_stop':dataclasses.asdict(c),'bf16_supported':torch.cuda.is_bf16_supported()})
 del back,batch,model;torch.cuda.empty_cache()
