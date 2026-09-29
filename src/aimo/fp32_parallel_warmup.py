"""Finite additional frozen groups while the original memory smoke finishes."""
import os,sys,time,json,fcntl,subprocess,shutil,traceback
from pathlib import Path
from .runtime import atomic_write_json as write
from .fp32_runtime import verify,Guard
from .bf16_collect import slot_path

def barrier(root):
    d=root/'runtime/oom_recovery_001/parallel_warmup'
    if not (d/'plan.json').exists():return
    start=time.monotonic()
    with (d/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=json.loads((d/'exit.json').read_text())
        assert result['codes']==[0,0,0] and not Guard(root).should_stop()[0],result
        for row in json.loads((d/'plan.json').read_text())['assignments']:
            groups=json.loads((root/'collection/groups.json').read_text())
            for p in groups[row['group']]['prompts']:
                for s in p['slots']:
                    raw=json.loads(slot_path(root,'collection',s['slot_id']).read_text())
                    assert raw['state']=='completed',s['slot_id']
        budget=json.loads((root/'gpu_budget.json').read_text())
        if not budget.get('parallel_warmup_accounted'):
            budget['gpu_hours']+=result['gpu_hours']
            budget['active_seconds']+=time.monotonic()-start
            budget['parallel_warmup_accounted']=True
            write(root/'gpu_budget.json',budget)

def main(root):
    d=root/'runtime/oom_recovery_001/parallel_warmup';d.mkdir(exist_ok=True)
    with (d/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (d/'plan.json').exists(),'already launched'
        verify(root)
        assert not Guard(root).should_stop()[0]
        groups=json.loads((root/'collection/groups.json').read_text())
        assignments=[{'gpu':g,'group':i} for g,i in [(0,5),(1,6),(3,7)]]
        for a in assignments:
            assert all(not slot_path(root,'collection',s['slot_id']).exists() for p in groups[a['group']]['prompts'] for s in p['slots'])
        write(d/'plan.json',{'assignments':assignments,'reason':'user requested expansion after five stable batch1 generations; unchanged original 12-slot smoke gate','time':time.time()})
        children=[];logs=[];hours=0.;last=time.monotonic()
        try:
            for a in assignments:
                log=(d/('gpu%d.log'%a['gpu'])).open('a');logs.append(log)
                code='from pathlib import Path; from aimo.fp32_collect import worker; worker(Path(%r),"collection",%d,group_indices=[%d])'%(str(root),a['gpu'],a['group'])
                p=subprocess.Popen([sys.executable,'-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a['gpu'])),stdout=log,stderr=subprocess.STDOUT)
                children.append(p)
            write(d/'started.json',{'pid':os.getpid(),'worker_pids':[p.pid for p in children],'time':time.time()})
            while True:
                alive=[p for p in children if p.poll() is None]
                now=time.monotonic();hours+=(now-last)*len(alive)/3600;last=now
                write(d/'cost.json',{'gpu_hours':hours,'active_workers':len(alive),'time':time.time()})
                if not alive:break
                if any(p.returncode not in (None,0) for p in children) or shutil.disk_usage(root).free<10*1024**3:
                    (root/'STOP').write_text('parallel warmup failure; preserve outputs')
                if Guard(root).should_stop()[0]:
                    raise RuntimeError('STOP during parallel warmup')
                time.sleep(5)
        except BaseException:
            (root/'STOP').write_text('parallel warmup failed; inspect preserved evidence')
            write(d/'failure.json',{'traceback':traceback.format_exc()})
            raise
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
            for log in logs:log.close()
            write(d/'exit.json',{'codes':[p.returncode for p in children],'gpu_hours':hours,'time':time.time()})
if __name__=='__main__':main(Path(sys.argv[1]))
