# SERVER_HANDOFF

로컬 CPU에서 끝난 범위와, 서버 GPU에서 이어서 할 일을 정리합니다. v2 primary는
**behavior supervision + flow auxiliary joint** 학습입니다.

## 1. 역할 분담

| 환경 | 하는 일 |
| --- | --- |
| 로컬 CPU | 구현, 단위/계약 test, synthetic toy end-to-end(E0/E0b), leakage/label routing/reload 검증, preflight, label store 구성 |
| 서버 GPU | DeepMath 후보 freeze, 검증된 pair 확보, calibration, 반복 행동 측정, Qwen3-4B Page extraction, E2 학습, E3/E4 평가 |

로컬에서는 CUDA/MPS 실행, 실제 Qwen weights 다운로드, 대규모 DeepMath 다운로드, 실제 답변
generation, GPU 실험을 하지 않습니다. GPU 실행은 CLI에서 `--execute-gpu`를 명시해야
시작되며, thinking profile의 `needs_calibration`이 비어 있지 않으면 full run이 차단됩니다.
기본 dry-run/preflight는 model weights를 로드하지 않습니다.

## 2. 환경

| 항목 | 로컬에서 실제로 검증한 버전 |
| --- | --- |
| Python | 3.12.2 |
| torch | 2.2.1 (CPU) |
| numpy | 1.26.4 |
| PyYAML | 6.0.1 |
| pytest | 9.0.3 |
| ruff | 0.16.9 |
| transformers | 4.39.1 (optional. Qwen3에는 부족 -> Qwen2 tiny config로 hook 논리만 검증) |

서버에서 추가로 필요한 것:

- `transformers >= 4.51` (Qwen3 config/모델 class)
- Qwen3-4B weights (서버에서만 내려받습니다)
- `datasets`(또는 pinned local parquet/JSONL) — DeepMath-103K snapshot
- 검증된 original-variant pair (semantic validation evidence 포함)
- (legacy 대조를 쓸 때만) MathGAP 설치와 **확인된** dotted path

`pyproject.toml`의 extras를 씁니다.

```bash
pip install -e ".[dev]"       # 로컬 CPU 개발/test
pip install -e ".[server]"    # 서버 전용 optional 의존성
```

## 3. 경로와 config

예시 경로는 `configs/server.example.yaml`에 있습니다.

```
repo:   /data1/HKM/AIMO
input:  /data1/HKM/data
output: /data1/HKM/result/AIMO/HKM/aimo_v2/<run_id>
```

경로는 전부 config로 관리합니다. 코드에는 개인 경로를 하드코딩하지 않습니다. 실제 값은
Git에서 제외되는 `configs/server.yaml`로 복사해 채웁니다 (`.gitignore` 참고).
`.env`, credentials, model weights, dataset cache, results, checkpoints, 대용량 로그는
Git에 넣지 않습니다.

## 4. 실행 순서

```bash
# 0) 환경과 계약 확인 (weights 로드 없음)
aimo check     --config configs/server.yaml
aimo preflight --config configs/server.yaml     # needs_calibration 목록을 확인합니다

# 1) DeepMath 후보와 split freeze
aimo prepare-data --config configs/server.yaml --input <deepmath snapshot>

# 2) 검증된 original-variant pair 확보 후 freeze (행동 실행 전에)
aimo import-pairs --config configs/server.yaml --input <verified_pairs.jsonl>

# 3) 작은 calibration으로 policy/scorer/budget 확정 (E1c)
#    runtime, VRAM, 완료율, 원본 정답률, label uncertainty를 확인하고
#    configs/server.yaml의 server.thinking에 확정값을 기록합니다.
aimo collect-outcomes --config configs/server.yaml --execute-gpu   # calibration subset

# 4) 반복 행동 측정 결과를 적재하고 label을 만듭니다
aimo collect-outcomes --config configs/server.yaml --from-file <outcomes.jsonl> \
  --merge-mode new_only
aimo build-labels --config configs/server.yaml

# 5) Page extraction
aimo extract --config configs/server.yaml --execute-gpu

# 6) behavior-only -> joint 학습과 평가
aimo train    --config configs/server.yaml --set train.task=behavior \
  --set model.name=behavior --run-id behavior_only
aimo train    --config configs/server.yaml --set train.task=joint --run-id joint_v2
aimo evaluate --config configs/server.yaml --run-id joint_v2 \
  --splits validation known_test unseen_perturbation_test harder_test
aimo predict-behavior --config configs/server.yaml --run-id joint_v2 --split known_test
```

중간에 끊기면 같은 config로 `aimo resume`(또는 `train --resume`)을 씁니다. config가 바뀌면
`ConfigMismatchError`로 막히므로 새 `run_id`를 씁니다.

**300개 원문을 곧바로 대규모 generation으로 자동 시작하지 않습니다.** 먼저 작은
calibration으로 runtime·VRAM·완료율·원본 정답률·label uncertainty를 확인합니다.

## 5. 필요한 산출물

`<output>/<run_id>/` 아래에 다음이 남습니다.

| 파일 | 내용 |
| --- | --- |
| `config.json` | 이 run에 고정된 config와 config hash |
| `run.lock` | 실행 중 lock (정상 종료 시 삭제) |
| `extract_ledger.jsonl` | extraction dedup/resume용 완료 key |
| `deepmath_candidates.json` | 후보 metadata와 split 배정 |
| `frozen_pairs.json` | freeze된 original-variant 후보 (hash 포함) |
| `outcomes.json` | prompt별 outcome counts |
| `labels.json` | pair drop / panel label store와 coverage |
| `gpu_budget.json` | 누적 GPU-active 시간 |
| `last.pt` / `best.pt` | checkpoint (model, optimizer, norm stats, hashes, RNG state) |
| `train_summary.json` | epoch history와 parameter 수 |
| `eval_summary.json` | split별 metrics와 bootstrap CI |

행동 측정 산출물에는 slot별 `C/W/X/U_score/infra_error/not_started`, `planned_trials`,
`completed_trials`, `termination_reason`, `policy_hash`를 그대로 남깁니다. 실패·미확정 pair를
성공할 때까지 다시 생성하지 않습니다. label store에는 signed pair drop과 bounds, 제외 사유,
coverage를 함께 남깁니다.

`train_summary.json`에는 `task`, `select_metric`, `trained_heads`,
`behavior_supervision_seen`, `label_coverage`가 들어갑니다. behavior supervision이 전혀 없는
실행에는 warning이 붙고 joint 성공으로 보고하지 않습니다.

## 6. GPU 예산 watchdog

누적 GPU-active 시간이 175분 이상이면 새 작업 시작을 막고, 진행 중이어도 180분에서
협조적으로 종료합니다. resume해도 같은 `gpu_budget.json`을 읽으므로 예산이 유지됩니다.
step 사이에서만 중단을 판단하며, **다른 사용자의 process나 container를 종료하는 코드는
없습니다**. 이 동작은 로컬 CPU에서 mock worker로 검증했습니다
(`tests/test_runtime.py::test_gpu_budget_stops_at_the_hard_limit_and_blocks_new_work`).

서버 단계는 두 부분을 씁니다. `GpuBudget.can_start()`는 CLI의 GPU 경로
(`screen`, `--execute-gpu` extract)에서 새 작업 시작을 막는 gate이고,
`run_budgeted(step_fn, budget, max_steps)`는 step 단위 작업을 예산 안에서 돌리는
wrapper입니다. 실제 GPU worker를 `step_fn`으로 넘기면 됩니다.

## 7. SERVER_PENDING 목록과 미확정 protocol

| 항목 | 왜 아직 아닌지 |
| --- | --- |
| DeepMath pinned snapshot revision | 아직 고정하지 않았습니다. 추측해서 채우지 않습니다. |
| 실제 DeepMath 전체 자료와 후보 300개 확정 | 로컬에서 대규모 다운로드를 하지 않습니다. |
| 검증된 original-variant pair | semantic validation evidence가 있는 pair가 아직 없습니다. 자동 변형은 formatting과 scope 검증된 alpha-rename만 지원합니다. |
| thinking profile의 `max_new_tokens`, `samples_per_prompt`, `max_total_context`, `numerical_backend`, `scorer_id`, `scorer_version` | E1c calibration에서 확정합니다. 확정 전에는 GPU full run이 차단됩니다. |
| `final_cap_failure_policy` | 최종 cap에서 실패 점수를 어떻게 정의할지 protocol에 명시해야 합니다. |
| 실제 Qwen3-4B 행동 측정과 Page extraction | 실제 weights와 GPU가 필요합니다. 로컬에서는 Qwen2 tiny random-init로 hook 논리만 검증했습니다. |
| 4B numerical audit | 실제 weights에서만 의미가 있습니다. |
| binary robust label 정의 | 출처와 frozen definition이 없으면 robust label은 null로 유지합니다. 현재 실자료 경로에서는 pair-drop regression이 실제 supervision입니다. |
| 독립적인 panel-only max-drop target | 없으면 max-drop loss는 끄고 diagnostic으로만 보고합니다. |
| E2/E3/E4 실자료 결과 | 위 단계가 끝난 뒤입니다. |
| 완전한 benchmark decontamination | 표면 n-gram/exact 겹침만 검사했습니다. 검증하지 못했습니다. |

## 8. 재개 방법 요약

1. `git pull` 후 `aimo check`로 계약, schema version, label coverage를 확인합니다.
2. `aimo preflight`로 경로, GPU 예산, thinking profile calibration, SERVER_PENDING 목록을
   확인합니다.
3. DeepMath snapshot revision과 검증된 pair 파일을 `configs/server.yaml`에 연결합니다.
4. 위 4절 순서대로 prepare-data -> import-pairs -> calibration -> collect-outcomes ->
   build-labels -> extract -> train을 돕니다.
5. 끊기면 같은 config로 `aimo resume`을 씁니다. GPU 예산은 `gpu_budget.json`에 누적되어
   유지됩니다.
6. 실행 환경·policy·Page가 바뀌면 hash가 달라집니다. 공식 모델 parameter가 다르다는 이유로
   global environment를 일괄 upgrade하지 않습니다. 이번 작업에서 서버 파일을 삭제하지
   않았습니다.
