"""CUDA plumbing audit on a real Page; not supervised training or an experimental result."""
import dataclasses,json,sys,time
from pathlib import Path
import torch
from .page import load_pages
from .data import OriginalGroup,PageDataset,compute_norm_stats,collate_panels
from .config import Config
from .train import build_model
from .runtime import atomic_write_json
root=Path(sys.argv[1]);torch.set_num_threads(2)
p=load_pages(next((root/'calibration/audit_pages').glob('*.npz')))[0]
v=dataclasses.replace(p,variant_id=p.variant_id+':runtime_identity_control')
g=OriginalGroup(p.original_id,p,[v]);data=PageDataset('train',[g])
stats=compute_norm_stats(data).to(torch.device('cuda'))
cfg=Config();cfg.model.name='joint';cfg.run.device='cuda'
m=build_model(cfg,data.hidden_size,data.n_blocks,data.n_landmarks).cuda()
b=collate_panels([g]).to(torch.device('cuda'))
out=m.forward_behavior(b.inputs,stats)
assert torch.isfinite(out.pair_drop).all()
out.pair_drop.sum().backward()
assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
atomic_write_json(root/'protocol/cuda_page_smoke.json',{'model_device':str(next(m.parameters()).device),'batch_device':str(b.inputs.orig_state.device),'finite_forward_backward':True,'normalization_on_cuda':True,'real_page_shape':list(p.state.shape),'identity_runtime_control_only':True,'not_a_trained_predictor':True,'peak_vram_bytes':torch.cuda.max_memory_allocated()})
print('actual Page CUDA model/batch/NormStats forward/backward passed')
