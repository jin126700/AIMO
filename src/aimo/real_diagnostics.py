"""Small CUDA backend/seed diagnostics, separate from all behavioral slots."""
import dataclasses
import json
import sys
import time
from pathlib import Path
import torch
from .real_run import cfg_for
from .adapters.qwen import load_real_qwen,QwenGenerationBackend,SlotRequest
from .runtime import atomic_write_json

root=Path(sys.argv[1]);cfg=cfg_for(root)
start=time.monotonic();torch.set_num_threads(2)
model,tok=load_real_qwen(cfg)
profile=dataclasses.replace(cfg.server.thinking,max_new_tokens=8)
backend=QwenGenerationBackend(profile,model,tok)
req=SlotRequest('diagnostic','diagnostic','Compute 2 + 2.','4',12345)
a=backend.generate(req);b=backend.generate(req)
assert a.generated_token_ids==b.generated_token_ids
calls=[0]
def stop():
    calls[0]+=1
    return calls[0]>=2
backend.stop_check=stop
c=backend.generate(dataclasses.replace(req,slot_id='stop-diagnostic'))
assert c.termination_reason=='external_stop' and c.generated_tokens<8
out={'seed_repeat_identical':True,'stop_during_generation':True,'stop_tokens':c.generated_tokens,'results':[dataclasses.asdict(x) for x in [a,b,c]],'wall_seconds':time.monotonic()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'excluded_from_calibration_and_main':True}
atomic_write_json(root/'protocol/real_backend_diagnostic.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='results'}))
