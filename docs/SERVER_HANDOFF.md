# SERVER_HANDOFF

로컬 CPU에서 끝난 범위와, 서버 GPU에서 이어서 할 일을 정리합니다.

## 1. 역할 분담

| 환경 | 하는 일 |
| --- | --- |
| 로컬 CPU | 구현, 단위/계약 test, E0 synthetic end-to-end, leakage/rollout/reload 검증, preflight |
| 서버 GPU | MathGAP screening, Qwen3-4B Page extraction, E2 학습, E3/E4 평가 |

로컬에서는 CUDA/MPS 실행, 실제 Qwen weights 다운로드, 실제 screening, GPU 실험을 하지
않습니다. GPU 실행은 CLI에서 `--execute-gpu`를 명시해야 시작됩니다. 기본
dry-run/preflight는 model weights를 로드하지 않습니다.

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
- MathGAP 설치와 **확인된** generator/renderer/oracle dotted path

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
output: /data1/HKM/result/AIMO/HKM/aimo_v1/<run_id>
```

경로는 전부 config로 관리합니다. 코드에는 개인 경로를 하드코딩하지 않습니다. 실제 값은
Git에서 제외되는 `configs/server.yaml`로 복사해 채웁니다 (`.gitignore` 참고).
`.env`, credentials, model weights, dataset cache, results, checkpoints, 대용량 로그는
Git에 넣지 않습니다.

## 4. 실행 순서

```bash
# 0) 환경과 계약 확인 (weights 로드 없음)
aimo check     --config configs/server.yaml
aimo preflight --config configs/server.yaml

# 1) MathGAP 자료 준비 (dotted path가 확인된 뒤에만 통과)
aimo prepare-data --config configs/server.yaml

# 2) screening: 실제 GPU 실행
aimo screen --config configs/server.yaml --execute-gpu

# 3) Page extraction
aimo extract --config configs/server.yaml --execute-gpu

# 4) Stage 1 학습과 평가
aimo train    --config configs/server.yaml
aimo evaluate --config configs/server.yaml \
  --splits validation known_test unseen_perturbation_test harder_test

# 또는 한 번에 (미지원 단계에서 즉시 멈춤)
aimo run-stage1 --config configs/server.yaml --execute-gpu
```

중간에 끊기면 같은 config로 `--resume`을 붙여 다시 시작합니다. config가 바뀌면
`ConfigMismatchError`로 막히므로 새 `run_id`를 씁니다.

## 5. 필요한 산출물

`<output>/<run_id>/` 아래에 다음이 남습니다.

| 파일 | 내용 |
| --- | --- |
| `config.json` | 이 run에 고정된 config와 config hash |
| `run.lock` | 실행 중 lock (정상 종료 시 삭제) |
| `extract_ledger.jsonl` | extraction dedup/resume용 완료 key |
| `gpu_budget.json` | 누적 GPU-active 시간 |
| `last.pt` / `best.pt` | checkpoint (model, optimizer, norm stats, hashes, RNG state) |
| `train_summary.json` | epoch history와 parameter 수 |
| `eval_summary.json` | split별 metrics와 bootstrap CI |

screening 산출물에는 slot별 `C/W/X/U_score/infra_error/not_started`와 pair eligibility
사유를 그대로 남깁니다. 실패·미확정 pair를 성공할 때까지 다시 생성하지 않습니다.

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

## 7. SERVER_PENDING 목록

| 항목 | 왜 아직 아닌지 |
| --- | --- |
| MathGAP generator/renderer/oracle 경로 | 공식 API와 revision을 아직 확인하지 못했습니다. 추측해서 채우지 않습니다. |
| 실제 screening (Qwen3-4B, 독립 4 slots) | 실제 weights와 GPU가 필요합니다. |
| Qwen3-4B Page extraction | `transformers >= 4.51`과 실제 weights가 필요합니다. 로컬에서는 Qwen2 tiny random-init로 hook 논리만 검증했습니다. |
| 4B numerical audit (residual identity 수치 확인) | 실제 weights에서만 의미가 있습니다. |
| E2/E3/E4의 실자료 결과 | 위 단계가 끝난 뒤입니다. |
| Stage 2 classifier / threshold | 이번 revision의 범위가 아닙니다. |

## 8. 재개 방법 요약

1. `git pull` 후 `aimo check`로 계약과 버전을 확인합니다.
2. `aimo preflight`로 경로, GPU 예산, SERVER_PENDING 목록을 확인합니다.
3. MathGAP dotted path를 `configs/server.yaml`에 채웁니다.
4. `aimo run-stage1 --execute-gpu`로 screening -> extraction -> 학습을 순서대로 돕니다.
5. 끊기면 같은 config로 `--resume`을 붙입니다.
