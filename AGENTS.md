# AGENTS.md

AIMO 저장소에서 작업할 때의 공통 지침입니다. 상세 내용은
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
[docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md),
[docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md),
[docs/FIXES.md](docs/FIXES.md)에 있습니다.

## 0. 현재 확정된 결정 (v2)

primary는 **Behavior-supervised Looped Transformer + Flow auxiliary**입니다.

- robust/non-robust 또는 correctness drop의 **행동 감독**으로 내부 변화 representation을
  학습합니다. Flow prediction은 실제 내부 전개를 설명하게 하는 auxiliary입니다.
- 이 결정은 이전의 "stable-only primary"와 "Behavior head를 구현하지 않는다"는 지침을
  **대체합니다**. Behavior head(pair signed drop, panel robustness pooling)는 승인된
  구현 대상입니다.
- stable-only anomaly 접근과 v1 flow-only 설정은 **legacy baseline**으로 보존합니다
  (`configs/toy.yaml`, `configs/stage1.yaml`, `train.task=flow`).
- Flow는 성공·실패 양쪽의 유효한 train pair에서 학습합니다. stable-stable로 제한하지
  않습니다.
- prediction residual 자체를 non-robust probability로 쓰지 않습니다.
- concept / recipe / 중요 layer / Gram matrix를 target으로 주지 않습니다.
- 완전한 비지도학습이 아니라 **outcome-supervised representation learning**입니다.
- correctness, robustness, uncertainty, causal mechanism을 구분합니다. 특정 label을 잘
  예측했다고 수학적 구조나 인과성을 발견했다고 쓰지 않습니다.
- v2는 protocol / code schema 버전이며 새 프로젝트 이름이 아닙니다. 프로젝트 이름은 계속
  AIMO입니다.

## 1. 확정 architecture

이미 확정된 것을 구현합니다. 과거 RFT/ACD/SAE/transcoder/classifier를 다시 섞거나 새
architecture 탐색으로 범위를 바꾸지 않습니다.

Behavior view와 Flow view는 **같은 `LoopedCore` 객체**를 공유하고 readout head만 다릅니다.
독립적인 Transformer를 두 개 만들지 않습니다. 두 view는 서로 다른 forward이며 Behavior의
hidden state나 KV cache를 Flow에 재사용하지 않습니다.

```
native vectors
  -> layer 간 공유하는 작은 learned input embedding
  -> shared pre-LN Transformer block 반복 (동일 parameter 객체 4회 호출)
  -> layer 간 공유하는 native-space readout
```

- 기본값: `d_model 128`, `heads 4`, `FFN 256`, `dropout 0.1`, loop 4회.
- 독립 block copy 4개를 만들지 않습니다 (`untied4` 비교군만 예외).
- 매 loop에서 고정 observation embedding을 다시 주입합니다:
  `x_{k+1} = Block(x_k + inject(x_0))`.
- 단일 `z64` bottleneck을 쓰지 않습니다. adaptive halting, loop sweep, per-loop 전용
  module을 넣지 않습니다.
- Page 계약: `state [L+1, P, H]`, `updates [L, P, 2, H]`, stream 0 = Mixer, 1 = FFN,
  `P = 17`. native `H`를 보존하고 projection으로 줄이지 않습니다.
- cut `d`에서의 예측은 original 전체 + variant prefix를 조건으로 하는 conditional
  prediction입니다. 순수 forecasting이나 LLM simulator라고 쓰지 않습니다.
- LLM의 layer `d`와 predictor의 loop `k`는 다른 개념입니다.
- Behavior view는 original/variant의 **전체** Page(state[0:L+1], updates[0:L])를 읽습니다.
  Flow의 cut 경로를 재사용해 마지막 variant block을 누락하지 않습니다.
- Behavior pair readout은 `Z_ij in R^128`이고, panel robustness는 variant 순서 embedding이
  없는 작은 learned attention pooling으로 집계합니다. panel max-drop은
  `max(0, max_j d_hat_ij)`이며 별도 대형 head를 만들지 않습니다.

### 절대 깨면 안 되는 것

- `variant state[d+1:]`와 `variant updates[d:]`는 Flow model input 객체에 넣지 않습니다.
- reference cell은 variant 관측을 읽지 않습니다 (간접 leakage 차단). Behavior forward를
  거친 뒤에도 Flow 예측이 바뀌지 않아야 합니다.
- query / pair readout cell에는 activation을 넣지 않습니다. 위치와 역할 정보만 넣습니다.
- `BehaviorInput`에는 정답·topic·difficulty·recipe·ID·sampling counts·label validity를
  넣지 않습니다.
- rollout은 `normalize -> predict -> inverse normalize -> raw recurrence` 순서입니다.
  정규화 좌표에 raw update를 더하지 않습니다. 최초 cut 이후 실제 variant future를 다시
  쓰지 않습니다.
- normalization scale은 유효한 train originals만으로 계산해 freeze합니다 (성공 사례만
  쓰지 않고 실패 사례도 포함). test split의 unlabeled Page를 flow pretraining이나
  normalization에 쓰지 않습니다. subset을 쓰면 그 subset 안에서만 다시 계산합니다.
- label mask는 loss routing에만 씁니다. **missing label을 0이나 False로 바꾸지 않습니다.
  label 0은 실제 label입니다.**
- original-panel label을 모든 variant의 pair label로 복사하지 않습니다.
- cap-hit(X) / U_score / infra_error / not_started를 W로 합치지 않습니다. 채점 불가와 명확한
  오답을 구분하고, 지원 범위를 벗어난 표현은 `U_score`입니다 (임의 eval 금지).
- outcome counts는 음이 아닌 정수이고 합이 `planned_trials`와 같아야 합니다. 미기록 slot은
  명시적 import 규칙으로만 `not_started`로 채웁니다. `N=0`의 확률·bounds는 undefined입니다.
- 완료된 slot을 새 독립 generation으로 대체하지 않습니다. 미시작 slot만 채우거나
  (`fill_not_started`) 별도 cohort로 보존합니다 (`separate_cohort`). exact continuation은
  slot 단위 trajectory 증거가 있어야 합니다.
- loss 항은 합 + count로 모으고 effective batch의 global denominator로 한 번만 나눕니다.
  서로 다른 항의 valid count를 합치지 않습니다. training과 evaluation은 같은 Huber helper를
  씁니다.
- seed는 data / split / train / sampler / eval로 분리합니다. train seed를 바꿔도
  data/split/label hash는 같아야 합니다.
- data hash는 ID 목록이 아니라 Page 내용 fingerprint와 label 값을 포함합니다.
- `run.device`가 실제 model/batch/NormStats에 적용됩니다. 전체 dataset을 올리지 않고
  microbatch만 옮기며, CUDA가 없으면 명시적 오류입니다.
- 예산은 예외·중단 시에도 elapsed를 저장하고, 소유한 subprocess만 종료합니다.
- 서로 다른 model / thinking mode / sampling policy의 label을 섞지 않습니다.
- joint에서 두 view의 gradient는 같은 core에 누적됩니다. shared parameter를 optimizer에
  중복 등록하지 않고, flow gradient를 detach하지 않습니다.
- checkpoint에는 schema / model / task version을 명시합니다. 구버전 checkpoint를 조용히 새
  모델로 해석하지 않습니다.

## 2. 언어와 코드 스타일

- 설명 문장은 한국어, 전문용어는 English로 씁니다 (README, docs, 주석, docstring, 보고).
- 변수·함수·파일명·config key·CLI flag는 명확한 English를 씁니다.
- 불필요한 한영 병기와 긴 번역 설명은 넣지 않습니다.
- type hints와 필요한 입력 검증을 씁니다. 함수는 작고 명확하게 둡니다.
- 주석은 shape/indexing/mask/normalization 중심으로 짧게 답니다.
- 과도한 class hierarchy, registry, 범용 framework, 빈 추상 class를 만들지 않습니다.
  반대로 모든 것을 거대한 파일 하나에 몰아넣지도 않습니다.
- 실제로 쓰이는 기능만 구현합니다.
- 진입점은 `aimo` 하나입니다. `python -m aimo`도 같은 CLI를 호출합니다.

## 3. 로컬 CPU와 서버 GPU

- 로컬: 구현, test, synthetic toy 검증, preflight. CPU만 씁니다.
- 로컬에서 하지 않는 일: CUDA/MPS 실행, 실제 Qwen weights 다운로드, 대규모 DeepMath
  다운로드, 실제 답변 generation, GPU 실험, 4B numerical audit.
- 실제 GPU full run은 thinking profile의 needs_calibration이 비어 있어야 시작됩니다.
- 실제 GPU 실행에는 `--execute-gpu`가 필요합니다. 기본 dry-run/preflight는 model
  weights를 로드하지 않습니다.
- 확인하지 못한 외부 API는 SERVER_PENDING으로 표시하고 fail-fast합니다. 빈 성공 파일이나
  허위 완료 결과를 만들지 않습니다.
- 다른 사용자의 process나 container를 종료하는 코드를 만들지 않습니다.

## 4. Tests

- 처음에 필요한 test를 돌리고, 수정 후에는 영향받는 범위부터 다시 돕니다.
- 의미 없는 수백 개의 parametrized test를 만들지 않습니다.
- core CPU test는 네트워크나 대형 weights를 요구하지 않습니다. optional 의존성이 필요한
  test는 `pytest.importorskip`으로 건너뜁니다.
- 반드시 유지할 targeted test 주제:
  Behavior/Flow의 실제 shared parameter identity, full-page Behavior와 masked Flow의 정보
  경로 분리, Behavior forward 후에도 Flow future-leakage 없음,
  학습/eval dropout mask와 독립적인 original/variant validity,
  pair/panel label routing과 missing label / valid label 0,
  negative signed drop과 panel max-drop 처리,
  panel permutation / padding / partial coverage,
  original non-robust를 모든 pair에 복제하지 않음,
  cap-hit / U_score가 wrong으로 바뀌지 않음,
  train-only normalization과 split/model/policy mismatch 차단,
  shared optimizer의 실제 gradient update, checkpoint/schema/reload/resume 검증,
  그리고 기존 flow 항목(shared parameter 수, untied 독립성, variant future 불변성,
  reference 경유 leakage, rollout 미래 정보 미사용, normalization inverse/mask/zero-scale).

```bash
.venv/bin/python -m pytest tests/
.venv/bin/ruff check src tests
```

## 5. 파일 관리와 Git

- 전역 Python 환경을 수정하지 않고 프로젝트 로컬 `.venv`를 씁니다.
- `.env`, credentials, 개인 server 설정, model weights, dataset cache, results,
  checkpoints, 대용량 로그는 Git에서 제외합니다 (`.gitignore`).
- commit 전에 `git diff --check`, 변경 파일과 staged diff 검토, secrets/대용량 파일 확인을
  합니다. source/config/docs/tests만 stage합니다.
- force push, repository 삭제, 보호 설정 우회, 자동 merge는 하지 않습니다.
- remote에 다른 변경이 있으면 force push 대신 branch를 분리하고 PR을 만듭니다.
- token이나 비밀번호를 코드·로그·명령 인자에 남기지 않습니다.

## 6. 보고 원칙

- 명세에 있는 것, 구현한 것, 실제로 검증한 것을 구분해 적습니다.
- 실행하지 않은 성능이나 명령을 "검증 완료"라고 쓰지 않습니다.
- 4/4 성공은 population robustness 인증이 아닙니다. 큰 prediction error가 곧
  non-robust를 뜻하지 않습니다.
- head를 구현했다는 사실과 실제 label로 학습했다는 사실을 구분합니다. behavior
  supervision이 전혀 없는 실행을 joint 성공으로 보고하지 않습니다.
- 집계 단위는 original group/panel입니다. layer나 variant를 독립 원문처럼 세지 않습니다.
- bootstrap CI와 training seed variance를 구분해 보고합니다.
- 모두 non-robust로 예측해 accuracy가 높아진 것을 성공이라고 하지 않습니다.
- panel 재정렬 불변성은 set pooling 검사이며 pair 정보 사용 여부의 강한 ablation이
  아닙니다. pair support-swap과 구분합니다.
- 학습된 pair score나 pooling weight를 개별 변형의 causal importance라고 쓰지 않습니다.
- train만 좋아지거나 `m0`와 차이가 없으면 근거가 부족한 것으로 보고합니다.
- 관측 성능을 causal mechanism이나 전체 AIMO 제출 성능으로 확대하지 않습니다.
- Cross-model transfer는 후속 목표이며 이번 결과로 주장하지 않습니다.
