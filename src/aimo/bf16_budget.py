"""One cumulative active-wall budget across every owned stage and GPU."""
import json,time,subprocess,os,sys,fcntl
from pathlib import Path
from .runtime import atomic_write_json
class Guard:
 def __init__(self,root,cutoff=180):self.root=Path(root);self.cutoff=cutoff;self.last=0;self.data={}
 def elapsed(self):
  if time.monotonic()-self.last>.5:self.data=json.loads((self.root/'gpu_budget.json').read_text());self.last=time.monotonic()
  return self.data['active_seconds']+(max(0,time.time()-self.data['updated_at']) if self.data.get('active_workers') else 0)
 def should_stop(self):return (self.root/'STOP').exists() or self.elapsed()>=self.cutoff*60,'shared_budget_or_external_stop'
def run(root,commands,stage):
 root=Path(root)
 with (root/'runtime/budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prior=json.loads((root/'gpu_budget.json').read_text());base=prior['active_seconds'];hours=prior['gpu_hours']
  if base>=175*60:return []
  children=[];logs=[];start=last=time.monotonic()
  try:
   for gpu,command in commands:
    log=(root/stage/f'gpu{gpu}.log').open('a');logs.append(log)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    children.append((gpu,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)))
   while True:
    now=time.monotonic();alive=[(g,p) for g,p in children if p.poll() is None];hours+=(now-last)*len({g for g,p in alive})/3600;last=now
    atomic_write_json(root/'gpu_budget.json',{**prior,'active_seconds':base+now-start,'gpu_hours':hours,'active_workers':len(alive),'updated_at':time.time(),'owned_pids':[p.pid for g,p in children],'stage':stage})
    if not alive:break
    if base+now-start>=180*60:
     for g,p in alive:p.terminate()
     for g,p in alive:
      try:p.wait(timeout=5)
      except subprocess.TimeoutExpired:p.kill();p.wait()
     break
    time.sleep(1)
  finally:
   for g,p in children:
    if p.poll() is None:p.terminate();p.wait()
   for log in logs:log.close()
   atomic_write_json(root/'gpu_budget.json',{**prior,'active_seconds':base+time.monotonic()-start,'gpu_hours':hours,'active_workers':0,'updated_at':time.time(),'owned_pids':[],'stage':stage})
  codes=[p.returncode for g,p in children];atomic_write_json(root/stage/'exit.json',{'codes':codes,'stage_seconds':time.monotonic()-start});return codes
if __name__=='__main__':
 r=Path(sys.argv[1]);run(r,[(0,[sys.executable,'-m','aimo.bf16_perf',str(r)])],'performance')
