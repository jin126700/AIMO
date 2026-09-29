"""Calibration prompt-side Pages under the existing supervisor budget."""
import json,sys
from pathlib import Path
import torch
from .real_run import cfg_for,sha,REV
from .real_parallel import ReadBudget
from .adapters.qwen import load_real_qwen,QwenGenerationBackend,ExtractionRequest,select_landmarks,run_page_extraction
from .runtime import DedupLedger,atomic_write_json
root=Path(sys.argv[1]);guard=ReadBudget(root,175)
if guard.should_stop()[0]:raise RuntimeError('No new GPU work after cutoff')
torch.set_num_threads(2);cfg=cfg_for(root);model,tok=load_real_qwen(cfg);backend=QwenGenerationBackend(cfg.server.thinking,model,tok)
frozen=json.loads((root/'protocol/frozen.json').read_text());requests=[]
for g in frozen['calibration']:
 for p in [{'id':g['id'],'question':g['question'],'tokens':g['tokens']}]+g['variants']:
  rendered=backend.render(p['question']);assert rendered['input_ids'][0].tolist()==p['tokens']
  text=rendered['text'];start=text.index(p['question']);end=start+len(p['question'])
  spans=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  body=[i for i,(a,b) in enumerate(spans) if a>=start and b<=end and b>a]
  positions=[body[round(j*(len(body)-1)/15)] for j in range(16)]
  offsets,valid,rel=select_landmarks(positions,len(spans))
  requests.append(ExtractionRequest(g['id'],p['id'],rendered['input_ids'].to('cuda'),offsets,valid,rel))
prov=json.loads((root/'calibration/numerical_audit.json').read_text())['provenance']
assert prov['policy_hash']==cfg.server.thinking.protocol_hash() and prov['tokenizer_hash']==sha(tok.backend_tokenizer.to_str())
pages,report=run_page_extraction(model,requests,ledger=DedupLedger.open(root/'calibration','page_ledger.jsonl'),guard=guard,provenance=prov,output_dir=root/'calibration/audit_pages')
report['unique_prompt_pages']=len(pages);report['calibration_only']=True;report['max_residual_error']=max(p.residual_identity_error() for p in pages);report['peak_vram_bytes']=torch.cuda.max_memory_allocated()
atomic_write_json(root/'calibration/page_report.json',report);print(json.dumps(report))
