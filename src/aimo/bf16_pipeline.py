"""Follow-up pipeline; one cumulative budget, partial data retained at every gate."""
import sys,json
from pathlib import Path
from .bf16_budget import run
from .runtime import atomic_write_json

def main(root):
 for stage in ['calibration','collection']:
  if (root/stage/'exit.json').exists():
   # Explicit resume skips stage only if every planned slot has durable terminal evidence.
   groups=json.loads((root/stage/'groups.json').read_text())
   from .bf16_collect import slot_path
   complete=all(slot_path(root,stage,s['slot_id']).exists() and json.loads(slot_path(root,stage,s['slot_id']).read_text())['state']!='started' for g in groups for p in g['prompts'] for s in p['slots'])
   if complete:continue
  codes=run(root,[(i,[sys.executable,'-m','aimo.bf16_collect','worker',str(root),stage,str(i)]) for i in range(4)],stage)
  from .bf16_report import summarize
  summary=summarize(root,stage)
  if any(code!=0 for code in codes) or summary['counts'].get('infra_error',0):
   atomic_write_json(root/'runtime/blocked.json',{'stage':stage,'reason':'worker_or_infrastructure_failure','codes':codes});return
 from .bf16_post import post
 post(root)
if __name__=='__main__':main(Path(sys.argv[1]))
