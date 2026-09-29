"""Performance-only long-cache diagnosis; artificial inputs, never behavior data."""
import torch,time,json,os,sys
from pathlib import Path
from aimo.real_run import cfg_for
from aimo.adapters.qwen import load_real_qwen
from aimo.bf16_budget import Guard
from aimo.runtime import atomic_write_json
root=Path(sys.argv[1]);guard=Guard(root,173)
assert not guard.should_stop()[0]
start=time.monotonic();torch.set_num_threads(2);cfg=cfg_for(root);cfg.server.thinking.numerical_backend='bf16_sdpa';model,tok=load_real_qwen(cfg)
atomic_write_json(root/'runtime/auxiliary_gpu.json',{'pid':os.getpid(),'gpu':2,'start':time.time(),'same_budget':str(root/'gpu_budget.json'),'bounded_timeout_seconds':120})
base=json.loads((root/'protocol/frozen.json').read_text())['calibration'][0]['tokens'];row=(base*((8192+len(base)-1)//len(base)))[:8192]
records=[]
with torch.inference_mode():
 for padded in [False,True]:
  ids=torch.tensor([row,row],device='cuda');mask=torch.ones_like(ids)
  if padded:mask[0,0]=0;ids[0,0]=tok.pad_token_id
  pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,1)
  torch.cuda.reset_peak_memory_stats();t=time.monotonic();out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=True,logits_to_keep=1);torch.cuda.synchronize();prefill=time.monotonic()-t
  cache=out.past_key_values;t=time.monotonic()
  for step in range(16):
   if guard.should_stop()[0]:raise RuntimeError('shared budget stop')
   ids=out.logits[:,-1].argmax(-1)[:,None];mask=torch.cat([mask,torch.ones((2,1),device='cuda',dtype=mask.dtype)],-1);pos=mask.cumsum(-1)-1
   out=model(input_ids=ids,attention_mask=mask,position_ids=pos[:,-1:],past_key_values=cache,use_cache=True,logits_to_keep=1)
  torch.cuda.synchronize();elapsed=time.monotonic()-t
  records.append({'padding_mask':padded,'batch':2,'cache_prefix_tokens':8192,'decode_steps':16,'decode_seconds':elapsed,'decode_tokens_s':32/elapsed,'prefill_seconds':prefill,'peak_live_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'performance_only':True,'greedy_only_for_cost_measurement_not_behavior_policy':True})
  del out,cache;torch.cuda.empty_cache();atomic_write_json(root/'performance/long_cache.json',records);print(json.dumps(records[-1]),flush=True)
atomic_write_json(root/'runtime/auxiliary_gpu_finished.json',{'seconds':time.monotonic()-start,'gpu_hours':(time.monotonic()-start)/3600,'gpu':2,'already_overlaps_primary_active_wall':True})
