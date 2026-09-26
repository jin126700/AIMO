# AGENTS.md

AIMO 저장소에서 작업할 때의 공통 지침입니다. 상세 내용은
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
[docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md)에 있습니다.

## 1. 확정 architecture

이미 확정된 것을 구현합니다. 과거 RFT/ACD/SAE/transcoder/classifier를 다시 섞거나 새
architecture 탐색으로 범위를 바꾸지 않습니다.

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

### 절대 깨면 안 되는 것

- `variant state[d+1:]`와 `variant updates[d:]`는 model input 객체에 넣지 않습니다.
- reference cell은 variant prefix를 읽지 않습니다 (간접 leakage 차단).
- query cell에는 activation을 넣지 않습니다. 위치와 역할 정보만 넣습니다.
- rollout은 `normalize -> predict -> inverse normalize -> raw recurrence` 순서입니다.
  정규화 좌표에 raw update를 더하지 않습니다. 최초 cut 이후 실제 variant future를 다시
  쓰지 않습니다.
- normalization scale은 train originals만으로 계산해 freeze합니다. subset을 쓰면 그
  subset 안에서만 다시 계산합니다.

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

- 로컬: 구현, test, E0 synthetic 검증, preflight. CPU만 씁니다.
- 로컬에서 하지 않는 일: CUDA/MPS 실행, 실제 Qwen weights 다운로드, 실제 screening,
  GPU 실험, 4B numerical audit.
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
  shared parameter identity와 loop1/loop4 parameter 수, untied parameter 독립성,
  variant future 변경에 대한 prediction 불변성, reference 경유 간접 leakage 차단,
  raw recurrence/rollout의 미래 정보 미사용, normalization inverse/valid mask/zero-scale,
  sibling/M0/support-swap group semantics, 실제 gradient와 parameter update,
  checkpoint reload/resume 재현성.

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
- 집계 단위는 original group입니다. layer나 variant를 독립 원문처럼 세지 않습니다.
- bootstrap CI와 training seed variance를 구분해 보고합니다.
- train만 좋아지거나 `m0`와 차이가 없으면 근거가 부족한 것으로 보고합니다.
- Cross-model transfer는 후속 목표이며 이번 결과로 주장하지 않습니다.
