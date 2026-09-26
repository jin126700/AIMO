# AIMO

original–variant pair의 **행동 변화**(correctness drop / robustness)를 감독 신호로 삼아
내부 변화 representation을 학습하는 연구 프로젝트입니다. 작은 Looped Transformer 하나가
behavior 예측과 flow 예측을 함께 담당합니다.

## 1. 연구 목적

**Primary (v2)**: robust/non-robust 또는 correctness drop의 행동 감독으로 유용한 내부 변화
representation을 학습합니다. Flow prediction(다음 update 차이 예측)은 실제 내부 전개를
설명하게 하는 **auxiliary**입니다.

```
L = L_behavior + 0.1 * L_flow
```

- Flow는 성공·실패 양쪽의 유효한 train pair에서 학습합니다.
- prediction residual 자체를 non-robust probability로 쓰지 않습니다.
- concept·recipe·중요 layer·Gram matrix를 target으로 주지 않습니다.
- 완전한 비지도학습이 아니라 outcome-supervised representation learning입니다.
- correctness / robustness / uncertainty / causal mechanism을 구분합니다.

**Legacy baseline**: screened-stable pair만으로 flow를 학습하는 v1 설정(stable-only anomaly
접근)은 `train.task=flow`와 `configs/toy.yaml` / `configs/stage1.yaml`로 그대로 실행할 수
있습니다.

초기 대상 model은 Qwen3-4B 하나이고 primary 데이터는
[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)입니다. Cross-model
transfer는 후속 목표입니다. v2는 protocol/code schema 버전이며 프로젝트 이름은 계속
AIMO입니다.

## 2. Architecture

```
Page 관측    state [L+1, P, H]   updates [L, P, 2, H]   (stream 0 = Mixer, 1 = FFN)
             H[d+1] = H[d] + U[d,Mixer] + U[d,FFN]

             ┌─ Behavior view ─ original+variant 전체 Page ─ pair readout ─ Z_ij (R^128)
LoopedCore ──┤                                               ├─ tanh -> signed pair drop [-1,1]
 (공유 1개)  │                                               └─ set pooling -> robust probability
             └─ Flow view ───── original 전체 + variant prefix[0:d] ─ V_hat[d] (auxiliary)

core         공유 input embedding -> shared pre-LN block 4회 -> view별 readout
             x_{k+1} = Block(x_k + inject(x_0))      (동일 parameter 객체 4회 호출)

rollout      normalize -> predict -> inverse normalize -> raw recurrence
             U_hat_var[d] = U_orig[d] + V_hat[d]
             H_hat_var[d+1] = H_var_or_pred[d] + sum_c U_hat_var[d,c]
```

`d_model 128`, `heads 4`, `FFN 256`, `dropout 0.1`. Behavior와 Flow는 **같은 LoopedCore
객체**를 쓰고 readout head만 다릅니다 (Transformer를 두 개 만들지 않습니다). 두 view는 서로
다른 forward이며, Flow model input에 variant future가 들어가지 않습니다. LLM의 layer `d`와
predictor의 loop `k`는 서로 다른 개념입니다. 자세한 내용은
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 있습니다.

## 3. 현재 상태

**로컬 CPU에서 실제 검증 완료 (E0b, v2 primary)**

- synthetic Page + behavior label 생성 -> behavior-only 학습 -> joint 학습 ->
  pair/panel/flow 평가 -> checkpoint reload (train 24 originals / 72 pairs)
- 같은 `LoopedCore`에 두 view의 gradient가 누적되고 shared parameter는 optimizer에 한 번만
  등록됨. behavior forward 뒤에도 flow 예측이 bit 단위로 동일 (future leakage 없음)
- label routing: missing label은 mask(0으로 바꾸지 않음), label 0은 실제 label,
  cap-hit/U_score는 wrong으로 바뀌지 않음, panel label을 pair로 복사하지 않음
- panel permutation 불변성(최대 `2.8e-09`), padding/partial panel, untrained head 표시
- 별개 process에서 `predict-behavior`를 두 번 실행해 panel 출력이 완전히 동일
- run lock, atomic save, config/schema mismatch 차단, dedup/resume, GPU-active 175분 신규
  시작 차단과 180분 종료 (mock worker)
- DeepMath / perturbation / Qwen thinking adapter를 작은 fixture로 CPU 검증

known-test에서 `joint`(186k params)은 pair drop MAE 0.164로 behavior-only 0.237,
`behavior_m0` 0.335, `constant` 0.274보다 낮고, pair support-swap을 하면 MAE가 늘어납니다
(+0.052). `behavior_m0`는 panel 안 예측 spread가 정확히 0이어서 variant를 구분하지 못함이
수치로 확인됩니다.

한계도 같이 기록합니다. seed 3개의 support-swap 증가량은 0.033 ± 0.034로 **seed에 걸쳐
안정적이지 않습니다.** robust classification의 labeled panel은 split당 5~7개뿐입니다.
flow 지표만 보면 legacy flow-only(0.257)가 joint(0.423)보다 좋습니다 — joint에서 flow는
가중치 0.1의 auxiliary입니다. toy label은 synthetic 생성 규칙이며 실제 LLM robustness가
아닙니다. 전체 숫자와 해석은 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)의 E0b 결과 절에
있습니다.

**legacy 검증 (E0, flow-only)**: v1 설정의 결과는 같은 문서의 E0 절에 보존했습니다.

**서버 실험 전 (SERVER_PENDING)**

- DeepMath pinned snapshot revision과 후보 300개 확정
- semantic validation evidence가 있는 original-variant pair
- thinking profile calibration (확정 전에는 GPU full run이 차단됩니다)
- 실제 Qwen3-4B 행동 측정과 Page extraction, 4B numerical audit
- binary robust label의 frozen definition (없으면 null, pair-drop regression이 실제 supervision)
- E2/E3/E4의 실자료 결과

전체 목록은 [docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md)에 있습니다.

## 4. Quick start

아래 명령은 로컬 CPU에서 실제로 통과한 것입니다.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/
.venv/bin/aimo check    --config configs/toy_behavior.yaml
.venv/bin/aimo make-toy --config configs/toy_behavior.yaml --out runs/e0b_pages
.venv/bin/aimo train    --config configs/toy_behavior.yaml --run-id e0b_joint \
  --set model.name=joint --set train.task=joint
.venv/bin/aimo evaluate --config configs/toy_behavior.yaml --run-id e0b_joint \
  --set model.name=joint --set train.task=joint \
  --splits validation known_test unseen_perturbation_test harder_test
.venv/bin/aimo predict-behavior --config configs/toy_behavior.yaml --run-id e0b_joint \
  --set model.name=joint --set train.task=joint --split known_test --n-originals 4
.venv/bin/aimo preflight --config configs/toy_behavior.yaml
```

`python -m aimo`도 같은 CLI를 호출합니다. 실제 GPU 실행에는 `--execute-gpu`가 필요하고,
기본 dry-run/preflight는 model weights를 로드하지 않습니다. legacy flow-only는
`--set train.task=flow --set train.select_metric=total`로 돌립니다.

## 5. 폴더 구조

```
configs/    toy_behavior.yaml (E0b), behavior.yaml (v2 서버), deepmath.example.yaml
            toy.yaml / stage1.yaml / server.example.yaml (legacy flow-only)
docs/       ARCHITECTURE.md, EXPERIMENTS.md, DATA_PROTOCOL.md, SERVER_HANDOFF.md
src/aimo/   page, data, labels, model, behavior, rollout, losses, train, evaluate,
            runtime, cli
            adapters/ (deepmath, perturbation, qwen, mathgap: optional server 의존성)
tests/      Page 계약, behavior/leakage, label routing, joint 학습/checkpoint,
            rollout/normalization, loss group semantics, runtime, adapters, CLI
```

- [AGENTS.md](AGENTS.md) - 작업 지침 (확정 결정, architecture 불변식, 스타일, test, Git)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Behavior/Flow view, mask, loop, loss, schema
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) - dataset, split, baselines, metrics, 실측 결과
- [docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md) - label, counts/bounds, provenance, protocol
- [docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) - 서버 환경, 실행 순서, SERVER_PENDING

## 6. 주의

- prediction residual은 robustness 지표가 아닙니다. 큰 prediction error가 곧
  non-robust를 뜻하지 않습니다.
- 4/4 성공은 population robustness 인증이 아닙니다. thinking profile은 연구용이며 공식 AIMO
  평가 정책과 동일하지 않습니다.
- head를 구현한 것과 실제 label로 학습한 것은 다릅니다. 학습되지 않은 head의 출력은 검증된
  robust probability가 아닙니다.
- 학습된 pair score나 pooling weight는 개별 변형의 causal importance가 아닙니다.
- Flow는 original 전체를 조건으로 하는 conditional prediction입니다. 순수 forecasting도,
  완전한 LLM simulator도 아닙니다.
- 집계 단위는 original panel입니다. layer나 variant를 독립 문제로 세지 않습니다.
  bootstrap CI는 seed variance가 아닙니다.
- toy synthetic 자료와 label은 검증용이며 실제 robust dataset이 아닙니다.
- 특정 label을 잘 맞혔다고 수학적 구조나 인과성을 발견한 것은 아닙니다.
