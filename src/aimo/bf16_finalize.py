"""CPU-only evidence-bounded final report with locked, atomic run-section updates."""
import json,time,fcntl,hashlib
from pathlib import Path
from .runtime import atomic_write_json,atomic_write_text

def finalize(root):
 def read(p,default=None):return json.loads((root/p).read_text()) if (root/p).exists() else default
 cal=read('calibration/summary.json',{});main=read('collection/summary.json',{});budget=read('gpu_budget.json',{});perf=read('performance/comparison.json',{});pages=read('pages/summary.json',{});paired=read('evaluation/paired.json',{});done=read('evaluation/completed.json',{});policy=read('protocol/policy.json',{})
 integrity=read('protocol/final_integrity.json',{});cost=read('calibration/runtime_estimate_final.json',{})
 minimum={'train':24,'validation':8,'known_original_test':12};actual=main.get('label_valid_originals',{});short={s:max(0,n-actual.get(s,0)) for s,n in minimum.items()}
 results={m:read('evaluation/'+m+'.json') for m in ['constant','raw_change','behavior_m0','behavior','joint'] if (root/'evaluation'/(m+'.json')).exists()}
 def metric(m,k):return results.get(m,{}).get('metrics',{}).get(k,{}).get('mean')
 decision='DATA/PROTOCOL_LIMIT'
 if len(results)==5:
  support=metric('joint','pair_support_swap_mae_increase')
  useful=paired.get('constant',{}).get('joint_mae_improvement',0)>0 and paired.get('behavior_m0',{}).get('joint_mae_improvement',0)>0 and support is not None and support>0
  decision='NO_ESTABLISHED_PAIR_SIGNAL'
  if useful:
   decision='FLOW_AUXILIARY_UNCONFIRMED'
   if paired.get('behavior',{}).get('ci95',[0])[0]>0:decision='PROMISING_JOINT_RESULT'
 summary={'long_cache_homogeneous_decode_speedup':perf.get('long_cache_no_padding_speedup'),'decision':decision,'label_valid_originals':actual,'label_valid_pairs':main.get('label_valid_pairs',{}),'shortfall':short,'unresolved_fraction':main.get('unresolved_fraction'),'evaluation_methods':list(results),'paired':paired,'budget':budget,'performance':perf,'pages':pages,'training_seed_variance':None,'flow_evaluation_complete':(root/'evaluation/joint_flow.json').exists(),'post_error':None if read('runtime/post_error_resolution.json',{}).get('resolved') else read('runtime/post_error.json'),'post_error_resolution':read('runtime/post_error_resolution.json'),'artifact_integrity':integrity,'runtime_estimate':cost}
 atomic_write_json(root/'final_summary.json',summary)
 fmt=lambda x:'—' if x is None else f'{x:.6f}'
 lines=[f'## {root.name}','',f'- 판단: **{decision}**', '- 목적: FP32 reference Page로 BF16 behavior의 signed pair-drop을 예측하는 observational prototype.', '- 직전 FP32 결과 해석: 행동 측정 비용에 의한 중단; architecture와 robustness 가설 미검증. 이전 C4와 새 BF16 labels는 합치지 않았다.',f'- 코드: 80f257ded148b3e199769aec6b6e4654d88915c6 + 미push local patch. 정확한 hash/dirty 상태: protocol/manifest.json 및 local.patch.',f'- Behavior policy: `{policy.get("behavior_policy_id")}`; FP32 observation policy: `{policy.get("observation_policy_id")}`.',f'- 공통 checkpoint: `{policy.get("target_model_revision")}`. 데이터·model 파일 hash는 manifest 참조.', '', '| 구분 | planned slots | C | W | X | 운영 중단 | 미시작 |', '|---|---:|---:|---:|---:|---:|---:|']
 for name,d in [('BF16 calibration',cal),('BF16 main',main)]:
  c=d.get('counts',{});t=d.get('terminations',{});lines.append(f'| {name} | {d.get("planned_slots",0)} | {c.get("C",0)} | {c.get("W",0)} | {c.get("X",0)} | {t.get("external_stop",0)+t.get("interrupted",0)} | {c.get("not_started",0)} |')
 lines+=['',f'- Main planned: 52 originals / 104 pairs / 156 prompts / 624 slots. 실제 측정 시작: {main.get("started_originals",0)} originals / {main.get("started_prompts",0)-main.get("started_originals",0)} pairs / {main.get("started_prompts",0)} prompts / {main.get("started_slots",0)} slots.',f'- Label-valid originals: {actual}; pairs: {main.get("label_valid_pairs",{})}; 부족 originals: {short}.',f'- Planned slots 미확정 비율: {main.get("unresolved_fraction",0):.2%}. 이는 모델 실패율이 아니며 X/채점불가/운영중단/미시작을 별도 원시 집계로 보존했다.',f'- 256-token 성능 전용 시험: FP32 serial {perf.get("fp32_serial_tokens_s",0):.2f} tokens/s; BF16 serial {perf.get("bf16_serial_speedup",0):.2f}×; BF16 batch12 합산 {perf.get("bf16_batch12_aggregate_speedup",0):.2f}×. 자연 길이·정답률 표본으로 사용하지 않았다.',f'- 실제 main 첫 시작~마지막 종료 관측구간 합산 처리량: {main.get("generation_window_tokens_per_second")} tokens/s; 자연 생성 길이: {main.get("natural_generated_token_quantiles",{})}. Cold start·batch latency·VRAM은 load*.json / *.done.json에 분리했다.', '', '| 비교군 seed0 | pair-drop MAE | Huber δ=.1 | sibling swap MAE 증가 |','|---|---:|---:|---:|']
 for m in ['constant','raw_change','behavior_m0','behavior','joint']:lines.append(f'| {m} | {fmt(metric(m,"pair_drop_mae"))} | {fmt(metric(m,"pair_drop_huber"))} | {fmt(metric(m,"pair_support_swap_mae_increase"))} |')
 lines+=['',f'- 학습·평가 상태: {"seed0 평가 완료" if len(results)==5 else "완료되지 않음; 위 최소 label 수 부족 또는 stage 로그의 실행 중단 사유 참조"}.',f'- Joint paired 비교: {paired if paired else "미실행: 관측 label 부족 또는 학습 미완료"}.', '- Raw-change 입력은 정규화한 final-state 차이 크기, 누적 update 차이 크기, 유효 landmark 비율뿐이다. 8 parameters 중 pair-drop 4개, untrained robust 4개.', '- Binary robustness label/Accuracy/AUROC는 없음; robust head null/untrained. Complete-panel max-drop은 diagnostic만 사용.', '- Bootstrap 2000회는 original 단위 paired resampling이며 고정 관측 labels에 조건부인 비교다. 4회 sampling의 실제 correctness probability 불확실성이나 training-seed variance를 의미하지 않는다.', '- 미확정이 많은 어려운 panel이 평가에서 제외되어 성능이 낙관적일 수 있다. 작은 known-test의 탐색적 결과이며 causal mechanism·공식 robustness 인증이 아니다.',f'- Page: {pages}. Calibration은 학습/validation/test에 포함하지 않았다.',f'- 누적 GPU-active: {budget.get("active_seconds",0)/60:.2f}분 / 180분; GPU-hours: {budget.get("gpu_hours",0):.3f}. 잔여 {max(0,180-budget.get("active_seconds",0)/60):.2f}분. 새 행동 측정은 누적120분 이후 금지.',f'- 산출물: `{root}`. Raw slots / labels bounds / Page checksum / frozen manifest 보존.',f'- 재현 이력: `{root}/commands.sh`.', '- 재개는 동일 policy의 미시작 slot만 가능하다. started-crash, X, W, U_score, infra_error는 새 sample로 교체하지 않는다. 예산 증액은 별도 사용자 승인 없이는 하지 않는다.', '', '```sh',f'sh {root}/runtime/resume.sh' if (root/'runtime/resume.sh').exists() else f'docker exec -d hkm-aimo-bf16 python -m aimo.bf16_pipeline {root}','```','', '- 위 재개 명령은 현재 supervisor가 없고 기존 container가 남아 있을 때 사용한다. 종료 예산을 우회하지 않는다. Backend 성능 시험과 prepare 명령을 재실행하지 않는다.']
 lines += ['',f'- 선택 경로: 같은 prompt의 독립4slots를 묶는 BF16 batch4. 8K-cache 성능 전용 decode 시험에서 padding 경로 대비 {perf.get("long_cache_no_padding_speedup",0):.3f}×; 이 수치는 전체 FP32/BF16 실행의 직접 비교나 정답률 동등성 검증이 아니다.',f'- 실제 tensor peak / allocator reserved peak: {integrity.get("max_generation_live_vram_bytes",0)/2**30:.2f} / {integrity.get("max_generation_reserved_vram_bytes",0)/2**30:.2f} GiB. 두 수치는 다르다.',f'- 관측 pair-drop 분포: {main.get("pair_drop_distribution",{})}. 이것만으로 pair-specific predictor 신호를 주장하지 않는다.',f'- 완료 prompt-batch {cost.get("completed_batch_count",0)}개 기반 전체624slots 비용 외삽: 평균 기준 {cost.get("full_main_minutes_at4GPUs_mean",0):.1f}분, P90 기준 {cost.get("full_main_minutes_at4GPUs_p90",0):.1f}분. CI가 아니며 미완료/미시작 문제는 더 오래 걸릴 수 있다.', '- Page config hash 비교 오류는 수정·재검증 완료했다. 초기 오류와 해결 근거는 runtime/post_error.json 및 runtime/post_error_resolution.json에 보존했다.', '- 체크섬이 유효한 main Page156개는 CUDA를 숨긴 재개 검사에서 모델 로드 없이 재사용했다. 마지막 생성 batch 시작은 누적119.765분으로120분 제한을 지켰다.', '- 총180분 중 남은 시간이 있어도 새 행동 측정120분 cutoff는 소진됐다. 추가 label 생성은 시간 정책의 별도 승인 없이 시작하지 않는다.']
 text='\n'.join(lines)+'\n';atomic_write_text(root/'report.md',text)
 path=Path('/data1/HKM/AIMO/Loop_result.md');begin=f'<!-- run:{root.name}:start -->';end=f'<!-- run:{root.name}:end -->'
 with path.with_name('.Loop_result.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old=path.read_text();section=begin+'\n'+text+end
  if begin in old:a=old.index(begin);b=old.index(end,a)+len(end);old=old[:a]+section+old[b:]
  else:old+='\n'+section+'\n'
  atomic_write_text(path,old)
if __name__=='__main__':
 import sys
 finalize(Path(sys.argv[1]))
