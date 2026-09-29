"""고정된 FP32 primary 전 단계를 재개 가능한 순서로 실행합니다."""
import json,sys,time,traceback,fcntl
from pathlib import Path
from .fp32_runtime import run,policy,Guard
from .runtime import atomic_write_json as write
from .real_run import record

def main(root):
    policy(root)
    recovery=root/'runtime/memory_settings.json'
    if recovery.exists() and not (root/'runtime/oom_recovery_001/smoke_passed.json').exists():
        raise RuntimeError('OOM memory smoke must pass; use fp32_recovery entry')
    from .fp32_parallel_warmup import barrier
    barrier(root)
    from .bf16_report import summarize
    if not (root/'calibration/numerical_audit.json').exists():raise RuntimeError('FP32 audit required')
    for stage in ['calibration','collection']:
        from .bf16_collect import slot_path
        groups=json.loads((root/stage/'groups.json').read_text())
        complete=all(slot_path(root,stage,s['slot_id']).exists() and json.loads(slot_path(root,stage,s['slot_id']).read_text())['state']!='started'
            for g in groups for p in g['prompts'] for s in p['slots'])
        if not complete:
            run(root,[(i,[sys.executable,'-m','aimo.fp32_collect','worker',str(root),stage,str(i)]) for i in range(4)],stage)
        summary=summarize(root,stage)
        if stage=='calibration' and not recovery.exists():
            assert summary['planned_slots']==36 and summary['started_slots']==36
            if any(k in summary['terminations'] for k in ['infra_error','external_stop','interrupted']):
                raise RuntimeError('calibration infrastructure issue')
            elapsed=summary['stage_wall_seconds']
            # 두 slot calibration과 네 slot 본실험의 부하 차이를 명시한 넓은 범위.
            nominal=elapsed*624/36
            eta={'collection_hours_low':nominal*.6/3600,'collection_hours_high':nominal*1.5/3600,
                 'calibration_wall_seconds':elapsed,'calibration_generated_tokens':summary['generated_tokens'],
                 'calibration_tokens_per_second':summary['stage_inclusive_tokens_per_second'],
                 'basis':'FP32 36 slots, four replicas; main batch4 vs calibration batch2 and length variation',
                 'necessity':'frozen 624 independent FP32 behavior slots; BF16 cannot supply labels',
                 'runtime_limit':None,'training_estimate_pending_actual_epoch_measurement':True}
            write(root/'runtime/eta.json',eta);print(json.dumps(eta),flush=True)
            record(root,'FP32_COLLECTION_READY',{'calibration':summary,'eta':eta})
    from .fp32_post import post
    post(root)

if __name__=='__main__':
    root=Path(sys.argv[1])
    try:
        with (root/'runtime/pipeline.lock').open('a') as pipeline_lock:
            fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # 이미 진행 중인 calibration supervisor가 종료할 때까지 OS lock으로 대기합니다.
            with (root/'runtime/supervisor.lock').open('a') as stage_lock:
                fcntl.flock(stage_lock,fcntl.LOCK_EX)
            main(root)
    except BaseException as exc:
        write(root/'runtime/failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()})
        raise
