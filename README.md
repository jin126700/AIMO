# AIMO

정답을 유지하는 original–variant pair의 내부 update 흐름을
작은 Looped Transformer로 학습하는 연구 프로젝트입니다.

## 1. 연구 목적

**Stage 1 (이번 범위)**: 고정 LLM에서 반복 정답을 관측한 original–variant pair를
선별하고, 개념이나 변형 종류 라벨 없이 내부 update 흐름을 자기지도학습합니다. 목표는
모든 state 좌표를 복원하는 것이 아니라, 서로 다른 variants의 변화 **방향·규모·깊이별
전개**를 구분해 예측하는 것입니다. 데이터 선별에는 correctness supervision을 쓰지만
predictor에는 정답·concept·recipe·robustness label을 주지 않습니다.

**Stage 2 (후속)**: predictor·normalization·관측 protocol을 freeze한 뒤, 실패 pair의
residual 양상이 robustness와 연결되는지 검증합니다. 이번 revision에는 Stage 2
classifier와 threshold를 구현하지 않습니다.

초기 대상 model은 Qwen3-4B 하나입니다. Cross-model transfer는 후속 목표입니다.

## 2. Architecture

```
Page 관측       state [L+1, P, H]              updates [L, P, 2, H]   (stream 0 = Mixer, 1 = FFN)
                H[d+1] = H[d] + U[d,Mixer] + U[d,FFN]

cut d 입력      original 전체 Page + variant state[0:d+1] + variant updates[0:d]
                (variant future는 model input에 들어가지 않습니다)

sequence        [original reference] [observed variant prefix] [next-depth query]

predictor       native vectors
                  -> 공유 learned input embedding      x_0
                  -> shared pre-LN block 4회           x_{k+1} = Block(x_k + inject(x_0))
                  -> 공유 native-space readout
                V_hat[d], shape [P, 2, H]

rollout         normalize -> predict -> inverse normalize -> raw recurrence
                U_hat_var[d]   = U_orig[d] + V_hat[d]
                H_hat_var[d+1] = H_var_or_pred[d] + sum_c U_hat_var[d,c]

loss            L = L_next + L_within + 0.25 * L_roll
```

`d_model 128`, `heads 4`, `FFN 256`, `dropout 0.1`. LLM의 layer `d`와 predictor의
loop `k`는 서로 다른 개념입니다. 자세한 내용은
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 있습니다.

## 3. 현재 상태

**로컬 CPU에서 실제 검증 완료 (E0)**

- synthetic Page 생성 (`H=32, L=6, P=4`), train originals만으로 normalization 계산
- forward/backward, 실제 학습, horizon 2/4 rollout 평가, checkpoint 저장과 reload 일치
- leakage 차단: variant future 불변성, reference 경유 간접 leakage 차단, mask 구조
- normalization inverse / valid mask / zero-scale inactive 처리
- sibling / M0 / support-swap group semantics, checkpoint resume 재현성
- run lock, atomic save, config mismatch 차단, dedup/resume, GPU-active 175분 신규 시작
  차단과 180분 종료 (mock worker로 CPU 검증)
- Qwen adapter: random-init tiny config에서 Page 추출과 residual identity 확인
- 비교군 6개 + seed 3개 + 25/50/100% subset을 같은 조건으로 학습/평가
  (train 24 originals / 72 variants, 100 epochs)

known-test에서 loop4는 pair-specific 변화를 학습했습니다: next-update normalized MSE
0.261 (persistence 0.459, M0 0.480), direction cosine 0.691, sibling-difference gain
0.636, support-swap degradation 0.462. M0는 persistence와 사실상 같고 sibling gain과
swap degradation이 정확히 0이므로, 차이는 variant prefix에서 온 것입니다.

한계도 같이 기록합니다. unseen-perturbation split에서는 loop model이 persistence보다
나쁘고(0.478 vs 0.420) cosine이 0.21로 떨어집니다. 작은 linear baseline은 next-update MSE
만 보면 known-test에서 경쟁력이 있습니다(0.229). 전체 숫자와 해석은
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)의 E0 결과 절에 있습니다.

**서버 실험 전 (SERVER_PENDING)**

- MathGAP generator/renderer/oracle의 확인된 API 경로
- 실제 Qwen3-4B screening과 Page extraction, 4B numerical audit
- E2/E3/E4의 실자료 결과

## 4. Quick start

아래 명령은 로컬 CPU에서 실제로 통과한 것입니다.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/
.venv/bin/aimo check --config configs/toy.yaml
.venv/bin/aimo make-toy --config configs/toy.yaml --out runs/e0_toy/pages
.venv/bin/aimo train --config configs/toy.yaml --run-id e0_loop4 --set model.name=loop4
.venv/bin/aimo evaluate --config configs/toy.yaml --run-id e0_loop4 --set model.name=loop4 \
  --splits validation known_test unseen_perturbation_test harder_test
.venv/bin/aimo predict --config configs/toy.yaml --run-id e0_loop4 --set model.name=loop4 --cut 2
.venv/bin/aimo preflight --config configs/toy.yaml
```

`python -m aimo`도 같은 CLI를 호출합니다. 실제 GPU 실행에는 `--execute-gpu`가 필요하고,
기본 dry-run/preflight는 model weights를 로드하지 않습니다.

## 5. 폴더 구조

```
configs/    toy.yaml (E0), stage1.yaml (E2), server.example.yaml
docs/       ARCHITECTURE.md, EXPERIMENTS.md, SERVER_HANDOFF.md
src/aimo/   page, data, model, rollout, losses, train, evaluate, runtime, cli
            adapters/ (mathgap, qwen: optional server 의존성)
tests/      Page 계약, leakage, rollout/normalization, loss group semantics,
            checkpoint, runtime, adapters, CLI
```

- [AGENTS.md](AGENTS.md) - 작업 지침 (architecture 불변식, 스타일, test, Git)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - shape, indexing, mask, loop, rollout, loss
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) - dataset, split, screening, baselines, metrics
- [docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) - 서버 환경, 실행 순서, SERVER_PENDING

## 6. 주의

- prediction residual은 robustness 지표가 아닙니다. 큰 prediction error가 곧
  non-robust를 뜻하지 않습니다.
- 4/4 성공은 population robustness 인증이 아닙니다. screening은 연구용이며 공식 AIMO
  평가 정책이 아닙니다.
- 이것은 original 전체를 조건으로 하는 conditional prediction입니다. 순수 forecasting도,
  완전한 LLM simulator도 아닙니다.
- 집계 단위는 original group입니다. layer나 variant를 독립 원문처럼 세지 않습니다.
  bootstrap CI는 training seed variance가 아닙니다.
- E0의 synthetic 자료는 검증용이며 실제 robust dataset이 아닙니다.
