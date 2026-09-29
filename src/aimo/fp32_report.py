"""FP32 primary 한국어 보고서: 관측, 비용, 비교와 한계를 구분합니다."""
import json,fcntl,time
from pathlib import Path
from .runtime import atomic_write_text,atomic_write_json
REPO=Path('/data1/HKM/AIMO')

def report(root,detail):
    c=detail['collection'];completion=detail['completion']
    ev=json.loads((root/'evaluation/completed.json').read_text()) if (root/'evaluation/completed.json').exists() else None
    text=f"## {root.name}\n\n"
    text+=f"- 상태: {completion['status']}\n- 본실험 계획: 52 originals / 104 pairs / 156 prompts / 624 slots.\n"
    text+=f"- 실제 counts: {c['counts']}; label-valid originals: {c['label_valid_originals']}; pairs: {c['label_valid_pairs']}.\n"
    text+=f"- unresolved 비율: {c['unresolved_fraction']:.3f}; drop 분포: {c['pair_drop_distribution']}.\n"
    text+="- Qwen 행동 / Page forward·저장 / predictor forward·backward: FP32. AMP/autocast/TF32 OFF.\n"
    text+="- 120/150/175/180분 제한 해제. 5시간은 상한이 아님. cap 16384와 sampling 조건은 유지.\n"
    text+="- BF16·calibration label 혼합 없음. 기존 BF16 결과를 보았으므로 untouched dataset 검증이 아님.\n"
    text+=f"- Page: {detail['pages']}.\n"
    if 'oom_recovery' in detail:
        recovery=detail['oom_recovery'];smoke=recovery['smoke']
        text+="\n### OOM과 같은 cohort 재개\n"
        text+="- GPU2 batch4가 generation14825 tokens에서 allocator90% 한도 내 추가934MiB 할당 실패. allocated33.31GiB, reserved-unused6.08GiB. DynamicCache 성장과 fragmentation/reservation 영향이며 추가 model replica 또는 과대 static cache 증거는 없음.\n"
        text+="- 변경: 4 replicas 유지, effective batch1, expandable_segments, allocator80%, sequence 종료 즉시 전체 cache 반환. FP32/policy/seed/cap/cohort 유지. batch shape에 따른 부동소수 연산의 bitwise 동일성은 주장하지 않음.\n"
        text+=f"- smoke peak allocated {smoke['peak_allocated_bytes']/2**30:.3f}GiB / reserved {smoke['peak_reserved_bytes']/2**30:.3f}GiB; single-replica {smoke['single_replica_tokens_per_second']:.3f}tokens/s. 이전 OOM batch는4×14825 tokens/3539sec이나 종료된 sequence를 포함해 유효 throughput과 직접 동등 비교할 수 없음.\n"
        text+=f"- 최종 상태: {recovery['final_operational_summary']}. 완료26개와 interrupted10개 원시 bytes 보존 검증 통과.\n"
        text+=f"- 재개 코드 SHA/dirty patch/source hashes: {recovery['code']}.\n"
    if ev:
        text+="\n| 모델 | Original-balanced MAE | 95% original bootstrap CI | n |\n|---|---:|---|---:|\n"
        for name,res in ev['outputs'].items():
            m=res['metrics']['pair_drop_mae']
            text+=f"| {name} | {m['mean']} | [{m['ci95_low']}, {m['ci95_high']}] | {m['n_originals']} |\n"
        text+="\nPaired improvement = baseline MAE − joint MAE; 양수일 때 joint 우세.\n"
        text+="\n| joint seed | 비교군 | 개선량 | paired 95% CI |\n|---:|---|---:|---|\n"
        for seed,comps in ev['paired'].items():
            for name,m in comps.items():text+=f"| {seed} | {name} | {m['improvement']} | {m['ci95']} |\n"
        swaps=[ev['outputs'][f'joint_seed{s}']['metrics']['pair_support_swap_mae_increase'] for s in [0,1,2]]
        flow=[ev['paired'][str(s)][f'behavior_seed{s}'] for s in [0,1,2]]
        pair_ok=all(m['mean'] is not None and m['mean']>0 for m in swaps) and swaps[0]['ci95_low'] is not None and swaps[0]['ci95_low']>0
        flow_ok=all(m['improvement'] is not None and m['improvement']>0 for m in flow) and flow[0]['ci95'][0]>0
        decision={'pair_specific_signal':'제한적 exploratory 근거' if pair_ok else '추가 가치 확인되지 않음',
                  'flow_auxiliary':'제한적 exploratory 개선' if flow_ok else '추가 가치 확인되지 않음'}
        from .fp32_post import compare
        def signal(name):
            m=ev['outputs'][name]['per_original_metrics']['pair_drop_mae']
            comparisons=[compare(ev['outputs'][b]['per_original_metrics']['pair_drop_mae'],m) for b in ['zero','constant_fit','behavior_m0_seed0']]
            swap=ev['outputs'][name]['metrics']['pair_support_swap_mae_increase']
            return all(c['ci95'] is not None and c['ci95'][0]>0 for c in comparisons) and swap['ci95_low'] is not None and swap['ci95_low']>0
        pair_signal=signal('joint_seed0') or signal('behavior_seed0')
        joint_all=all(
            ev['paired'][str(seed)][b]['improvement'] is not None and ev['paired'][str(seed)][b]['improvement']>0
            for seed in [0,1,2] for b in ['zero','constant_fit','behavior_m0_seed0',f'behavior_seed{seed}'])
        has_test=ev['outputs']['joint_seed0']['metrics']['pair_drop_mae']['n_originals']>0
        verdict=('DATA/PROTOCOL_LIMIT' if not has_test else
                 'PROMISING_JOINT_RESULT' if pair_signal and flow_ok and joint_all else
                 'FLOW_AUXILIARY_UNCONFIRMED' if pair_signal else 'NO_ESTABLISHED_PAIR_SIGNAL')
        decision.update(verdict=verdict,behavior_signal='PAIR_BEHAVIOR_SIGNAL' if pair_signal else None,
                        criterion='seed0 original-group paired CI against zero/constant/M0 plus support-swap; seed1/2 directional consistency for promising joint; exploratory only')
        atomic_write_json(root/'evaluation/interpretation.json',decision)
        text+=f"\n- 최종 판정: {verdict}.\n"
        counts=c['label_valid_originals']
        if any(counts.get(s,0)<n for s,n in [('train',24),('validation',8),('known_original_test',12)]):
            text+="- small exploratory result: 권장 최소 original 수에 미달하는 split이 있음. 실제 유효 감독으로 학습하고 한계를 표시했으며 split 재배정은 하지 않음.\n"
        text+=f"\n- Pair-specific signal: {decision['pair_specific_signal']}. 모든 유효 target의 sibling support-swap MAE 증가: {swaps}.\n"
        text+=f"- Flow auxiliary: {decision['flow_auxiliary']}. seed별 joint vs behavior-only 결과를 모두 보고함.\n"
        text+=f"- Training seed variance: {ev['training_seed_variance']}.\n"
        text+="- Flow next/within/2–4step rollout: evaluation/joint_flow.json. test Page는 normalization·pretraining에 미사용.\n"
    prior=detail['execution_amendment']['prior_cost'];current=detail['current_cost']
    text+=f"\n- 이번 FP32 active wall: {current['active_seconds']/3600:.3f} h; GPU-hours: {current['gpu_hours']:.3f}.\n"
    text+=f"- 이전 누적 비용: active wall {prior['active_seconds']/3600:.3f} h; GPU-hours {prior['gpu_hours']:.3f}; 이번 실행과 별도로 보존.\n"
    text+=f"- 전체 GPU-hours(기존+이번): {prior['gpu_hours']+current['gpu_hours']:.3f}. 별도 technical audit 시간은 protocol/technical_cost.json 참조.\n"
    text+=f"- 실행 코드: {detail['code']}. 상세 data/policy/source hashes: protocol/manifest.json 및 executed_source_hashes.json.\n"
    text+="- 네 번의 sampling target은 population correctness의 정확한 값이 아님. original bootstrap은 관측 label에 조건부이고 training seed variance와 별개임.\n"
    text+="- Binary robust head null/untrained. 공식 robustness 또는 수학적 causal mechanism 검증으로 해석하지 않음.\n"
    text+=f"- 재개: sh {root}/runtime/resume.sh\n- 결과: {root}\n"
    atomic_write_text(root/'report.md',text)
    begin=f'<!-- run:{root.name}:start -->';end=f'<!-- run:{root.name}:end -->'
    with (REPO/'.Loop_result.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=REPO/'Loop_result.md';old=path.read_text()
        section=begin+'\n'+text+end
        if begin in old:
            a=old.index(begin);b=old.index(end,a)+len(end);old=old[:a]+section+old[b:]
        else:old+='\n'+section+'\n'
        atomic_write_text(path,old)
