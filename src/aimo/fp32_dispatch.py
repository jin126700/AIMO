"""Work-conserving scheduling of the unchanged finite FP32 collection."""
import os,sys,time,json,fcntl,subprocess,shutil,traceback
from pathlib import Path
from .runtime import atomic_write_json as write
from .fp32_runtime import Guard,verify
from .bf16_collect import slot_path

def pending(root,g):
    return any(not slot_path(root,'collection',s['slot_id']).exists() for p in g['prompts'] for s in p['slots'])

def worker(root,gpu):
    from .fp32_collect import load,collect,memory_settings,sha
    groups=json.loads((root/'collection/groups.json').read_text())
    reserved={5,6,7} # Already owned by the original auxiliary workers.
    model,tok,backend,settings=load(root)
    write(root/'collection'/f'precision_dynamic_gpu{gpu}.json',settings)
    claims=root/'runtime/dynamic_dispatch/claims';claims.mkdir(exist_ok=True)
    for i,g in enumerate(groups):
        if i in reserved or not pending(root,g):continue
        with (claims/(sha(g['original_id'])+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if not pending(root,g):continue
            write(root/'runtime/dynamic_dispatch'/f'gpu{gpu}.json',{'group':i,'original_id':g['original_id'],'time':time.time(),'pid':os.getpid()})
            for p in g['prompts']:
                if Guard(root).should_stop()[0]:raise RuntimeError('STOP')
                collect(root,'collection',g,p,backend)

def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:return p.read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:return False

def main(root):
    d=root/'runtime/dynamic_dispatch';d.mkdir(exist_ok=True)
    a=root/'runtime/oom_recovery_001/parallel_warmup'
    with (root/'runtime/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify(root)
        from .fp32_recovery import preserve_check
        preserve_check(root)
        assert json.loads((root/'runtime/oom_recovery_001/smoke_passed.json').read_text())['passed']
        assert not Guard(root).should_stop()[0]
        old=json.loads((root/'gpu_budget.json').read_text())
        plan=json.loads((a/'plan.json').read_text())['assignments']
        pids=json.loads((a/'started.json').read_text())['worker_pids']
        dependencies={row['gpu']:pid for row,pid in zip(plan,pids)}
        children={};logs=[];launched=set();hours=0.;start=last=time.monotonic()
        # Account for wall time after the smoke, including the old barrier wait.
        base_seconds=old['active_seconds']+max(0,time.time()-old['updated_at'])
        try:
            with (root/'runtime/supervisor.lock').open('a') as stage_lock:
                fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                while True:
                    now=time.monotonic()
                    hours+=(now-last)*sum(p.poll() is None for p in children.values())/3600;last=now
                    if Guard(root).should_stop()[0] or any(p.returncode not in (None,0) for p in children.values()):
                        raise RuntimeError('worker failure or STOP')
                    if shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('disk below 10GiB')
                    for gpu in range(4):
                        if gpu in launched or (gpu in dependencies and alive(dependencies[gpu])):continue
                        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                        if free<33000:continue
                        log=(d/f'gpu{gpu}.log').open('a');logs.append(log)
                        children[gpu]=subprocess.Popen([sys.executable,'-m','aimo.fp32_dispatch','worker',str(root),str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                        launched.add(gpu)
                    aux=json.loads((a/'cost.json').read_text())
                    write(root/'gpu_budget.json',{'active_seconds':base_seconds+now-start,'gpu_hours':old['gpu_hours']+hours+aux['gpu_hours'],
                        'active_workers':sum(p.poll() is None for p in children.values())+aux['active_workers'],'stage':'collection_dynamic',
                        'owned_pids':[p.pid for p in children.values() if p.poll() is None],'updated_at':time.time(),'parallel_warmup_accounted':True})
                    write(d/'status.json',{'launched_gpus':sorted(launched),'pids':{g:p.pid for g,p in children.items()},'codes':{g:p.poll() for g,p in children.items()},'time':time.time()})
                    if len(launched)==4 and all(p.poll() is not None for p in children.values()) and (a/'exit.json').exists():break
                    time.sleep(5)
            assert json.loads((a/'exit.json').read_text())['codes']==[0,0,0]
            groups=json.loads((root/'collection/groups.json').read_text())
            assert all(not pending(root,g) for g in groups)
            assert all(json.loads(slot_path(root,'collection',s['slot_id']).read_text())['state']!='started' for g in groups for p in g['prompts'] for s in p['slots'])
            preserve_check(root)
            write(d/'done.json',{'time':time.time(),'codes':{g:p.returncode for g,p in children.items()}})
            from .fp32_pipeline import main as pipeline
            pipeline(root)
            (root/'runtime/pipeline.exit').write_text('0\n')
        except BaseException as e:
            (root/'STOP').write_text('dynamic dispatcher failure; preserve and inspect')
            write(root/'runtime/failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()})
            (root/'runtime/pipeline.exit').write_text('1\n')
            raise
        finally:
            for p in children.values():
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
            for log in logs:log.close()

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(Path(sys.argv[2]),int(sys.argv[3]))
    else:main(Path(sys.argv[1]))
