# EXPERIMENTS

Stage 1의 dataset, split, screening, 비교군, metrics, 실험 순서를 정리합니다.
명세(계획), 구현(코드에 있는 것), 검증(실제로 돌린 것)을 구분해 적습니다.

## 1. 연구 목적과 경계

Stage 1은 고정 LLM에서 반복 정답을 관측한 original-variant pair를 선별하고, 개념이나
변형 종류 라벨 없이 내부 update 흐름을 자기지도학습합니다. 중점은 모든 state 좌표를
완벽히 복원하는 것이 아니라, 서로 다른 variants의 **변화 방향·규모·깊이별 전개**를
구분해 예측하는 것입니다.

Stage 2는 후속 단계입니다. predictor·normalization·관측 protocol을 freeze한 뒤, 실패
pair의 residual 양상이 robustness와 연결되는지 검증합니다. **이번 revision에는 Stage 2
classifier와 threshold를 구현하지 않습니다.**

구분해서 기억할 것:

- 4/4 성공은 population robustness 인증이 아닙니다.
- 데이터 선별에는 correctness supervision이 쓰입니다.
- predictor에는 정답·concept·recipe·robustness label을 주지 않습니다.
- 큰 prediction error가 곧 non-robust를 뜻하지 않습니다.
- 새로운 정상 perturbation과 실제 실패를 구분해야 합니다.
- 초기 대상은 Qwen3-4B 하나입니다.
- Cross-model transfer는 후속 목표이며, 이번 검증 결과로 주장하지 않습니다.

## 2. 실험 순서

| 실험 | 내용 | 상태 |
| --- | --- | --- |
| E0 | CPU synthetic end-to-end, leakage, rollout, reload 검증 | 로컬에서 실제 실행 완료 |
| E1 | 서버에서 MathGAP screening과 Qwen3-4B Page extraction 검증 | SERVER_PENDING |
| E2 | screened-stable 자료로 실제 update 흐름 학습 | SERVER_PENDING |
| E3 | 새 original·새 perturbation·높은 난이도 평가 | SERVER_PENDING |
| E4 | 독립 original 25/50/100% 학습곡선으로 data efficiency 비교 | 기계는 구현/toy에서 실행, 실자료는 SERVER_PENDING |

Stage 2는 이 순서 뒤에 오며 이번에 실행하지 않습니다.

## 3. Dataset과 split

split은 다섯 개입니다.

```
train / validation / known-test / unseen-perturbation-test / harder-test
```

- 같은 original의 variants와 seeds는 **항상 같은 split**에 둡니다. 분할 단위는 original
  group입니다.
- train 안의 rename이나 유효 reorder는 train 쪽 다양성이고, held-out 동치 표현과 변형
  합성은 별도 split입니다. 두 가지를 섞지 않습니다.
- sibling objective는 original당 stable variants가 최소 2개 필요합니다. 후보 variant
  수와 실제 stable로 남은 수를 보고에서 구별합니다.
- 25/50/100% subset은 original-group 단위의 **고정 nested subset**입니다 (정렬된 group
  목록의 prefix). normalization도 각 training subset 안에서만 계산합니다.

E0의 synthetic 자료는 `H=32, L=6, P=4`이고 exact residual identity와 nonzero sibling
dynamics를 가집니다. split 사이에서 한 번에 한 요인만 바뀝니다.

- `known_test`: train과 같은 delta basis family, 같은 delta scale, 새 original group
- `unseen_perturbation_test`: 같은 delta scale, **train에서 쓰지 않은 family**
- `harder_test`: 같은 family(unseen), **더 큰 delta scale**

따라서 `known -> unseen` 차이는 새 perturbation 효과, `unseen -> harder` 차이는 난이도
효과로 읽습니다. 이것은 검증용 자료이며 실제 robust dataset이 아닙니다.

## 4. Screening (E1)

MathGAP generator/renderer/oracle는 최소 adapter로 연결합니다. 공식 API와 revision의
필요한 부분만 config의 dotted path로 지정하며, 확인하지 못한 API는 추측해서 지원한다고
쓰지 않습니다. 경로가 비어 있으면 adapter가 SERVER_PENDING으로 fail-fast합니다.

초기 screening 설정:

```
non-thinking, temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0
max_new_tokens = 256, unique prompt당 독립 4 slots
```

고정 final-answer parser(마지막 `\boxed{}` -> `answer:` 라벨 -> 마지막 숫자)와 exact
oracle을 씁니다. 이는 **연구용 screening**이며 공식 AIMO 평가 정책이 아닙니다.

slot outcome은 분리해서 셉니다.

| 코드 | 뜻 |
| --- | --- |
| `C` | 정답 |
| `W` | 오답 |
| `X` | cap-hit (max_new_tokens에서 잘림). `W`로 합치지 않습니다. |
| `U_score` | 채점 불가 (답을 못 뽑음) |
| `infra_error` | 실행 오류 |
| `not_started` | 시작되지 않음 |

Eligibility는 `semantic-valid AND original C4 AND variant C4`입니다. 실패하거나 미확정인
pair를 성공할 때까지 다시 생성하지 않습니다.

Qwen adapter는 random-init tiny config로 CPU에서 검증합니다. 실제 4B weights는 로컬에서
내려받지 않으며, real screening과 4B numerical audit는 SERVER_PENDING입니다.

## 5. 비교군

| 이름 | 설명 |
| --- | --- |
| `persistence` | `V_hat = 0` |
| `linear` | 작은 linear conditional baseline |
| `m0` | original-only. pair-specific 정보 없음 |
| `loop1` | shared block 1회 |
| `loop4` | shared block 4회. **primary** |
| `untied4` | 독립 block 4개 |

같은 input/output/data/loss 조건으로 비교합니다. baseline 특성상 다른 부분은 명시합니다:
`persistence`는 parameter가 없고, `m0`는 observed variant cell을 sequence에서 뺍니다.

Support-swap은 **같은 original의 sibling variants 사이에서만** 수행합니다. original
참조와 target은 그대로 두고 관측 prefix만 sibling의 것으로 바꿉니다.

## 6. Metrics

- next-update raw MSE / normalized MSE
- direction cosine (landmark x stream 단위, H 방향)
- relative magnitude (`||V_hat|| / ||V||`)
- 2-step / 4-step rollout error (normalized)
- sibling-difference gain (`1 - MSE(V_hat_a - V_hat_b, V_a - V_b) / MSE(0, V_a - V_b)`)
- correct-support vs swapped-support degradation
- identity example의 normalized MSE (zero-V 대조)

집계 규칙:

- zero/noise norm의 cosine과 relative magnitude는 값을 만들지 않고 **undefined로 셉니다**.
- 집계 단위는 항상 original group입니다. layer나 variant를 독립 원문처럼 세지 않습니다.
- 불확실성은 original-group bootstrap으로 보고하고, training seed 사이의 분산은 별도로
  기록합니다. bootstrap CI는 seed variance가 아닙니다.

보고 기준:

- train만 좋아지거나 `m0`와 차이가 없으면, pair-specific 변화를 배웠다는 근거가 부족한
  것으로 보고합니다.
- unseen perturbation 일반화와 harder 일반화는 각각 따로 평가합니다.

## 7. 로컬 재현 명령 (E0)

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
```

비교군은 `--set model.name=<name> --run-id e0_<name>`으로 같은 방식으로 돌립니다.
seed variance는 `--set run.seed=1`처럼 seed만 바꿔 여러 run을 만들고,
data efficiency는 `--set data.subset_fraction=0.25`로 만듭니다. 결과는
`aimo evaluate --compare runs/e0_loop4 runs/e0_m0 ...`로 한 표에 모읍니다.

## 8. E0 실행 결과 (로컬 CPU, 실측)

설정: toy synthetic `H=32, L=6, P=4`, train 24 originals / 72 variants (identity 11%),
validation·각 test split 8 originals / 24 variants, 100 epochs, patience 15,
`AdamW lr 3e-4`, effective batch 8 originals. 모든 run은 같은 자료·loss·metric을 씁니다.

### known-test (train과 같은 perturbation family, 새 original group)

| model | params | next MSE | cosine | ratio | h2 | h4 | sibling gain | swap deg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 0 | 0.4586 | n/a | n/a | 0.0932 | 0.2538 | 0.0000 | 0.0000 |
| linear | 6,592 | 0.2289 | 0.6104 | 0.7226 | 0.0446 | 0.1193 | 0.5372 | 0.3306 |
| M0 (original-only) | 168,608 | 0.4795 | -0.0926 | 0.1942 | 0.0984 | 0.2686 | 0.0000 | 0.0000 |
| loop1 | 168,608 | 0.2737 | 0.6556 | 1.0506 | 0.0623 | 0.1863 | 0.6222 | 0.4484 |
| **loop4 (primary)** | 168,608 | **0.2606** | **0.6906** | 1.0843 | 0.0566 | 0.1679 | **0.6364** | **0.4618** |
| untied4 | 566,048 | 0.2446 | 0.7101 | 1.0495 | 0.0547 | 0.1592 | 0.6537 | 0.4737 |

`next MSE`/`h2`/`h4`는 낮을수록, `cosine`/`sibling gain`/`swap deg.`는 높을수록 좋습니다.
`ratio`는 1에 가까울수록 크기가 맞습니다. persistence의 cosine과 ratio는 `V_hat = 0`이라
undefined로 기록됩니다.

읽는 방법:

- loop model은 M0와 명확히 다릅니다 (0.261 vs 0.480). M0의 sibling gain과 swap
  degradation은 정확히 0입니다. 따라서 known family에서는 **variant prefix에서 온
  pair-specific 신호를 학습했다**는 근거가 있습니다.
- `loop4`는 `loop1`보다 낫고, `untied4`(parameter 3.4배)와는 거의 같습니다. shared block
  4회가 parameter를 4배 쓰지 않고 같은 수준에 도달합니다.
- 작은 `linear` baseline은 next-update MSE만 보면 loop4보다 낫습니다(0.229). 다만
  cosine(0.610), sibling gain(0.537), swap degradation(0.331)은 모두 loop4보다 낮습니다.
  즉 이 toy에서 MSE 하나만으로 비교하면 결론이 바뀝니다.
- identity example의 normalized MSE는 persistence 0.0000, loop4 0.1330입니다. loop model은
  zero-V 대조에서 약간 과예측합니다.

### 일반화: unseen perturbation과 harder

| model | unseen next | unseen cosine | unseen sib gain | harder next | harder cosine |
| --- | --- | --- | --- | --- | --- |
| persistence | 0.4200 | n/a | 0.0000 | 1.0148 | n/a |
| linear | 0.3668 | 0.3423 | 0.1803 | 0.8631 | 0.3727 |
| M0 | 0.4273 | 0.0025 | 0.0000 | 1.0160 | 0.0468 |
| loop4 | 0.4783 | 0.2132 | 0.1734 | 0.9623 | 0.3338 |
| untied4 | 0.4389 | 0.2500 | 0.2168 | 0.9133 | 0.3746 |

- **unseen perturbation family에서는 loop model이 persistence보다 나쁩니다** (0.478 vs
  0.420). cosine은 0.69에서 0.21로, sibling gain은 0.64에서 0.17로 떨어집니다. 새 family로의
  일반화는 이 toy 설정에서 확인되지 않았습니다.
- harder split(같은 unseen family, 더 큰 delta)에서는 loop4가 persistence보다 약간 낫지만
  차이가 작습니다(0.962 vs 1.015).
- 두 결과를 하나로 묶어 "일반화했다"고 쓰지 않습니다. known family 안에서의 학습과
  새 family로의 일반화는 서로 다른 주장입니다.

### 불확실성: bootstrap CI와 seed variance는 다른 양

`loop4`, known-test, original-group bootstrap 200회 기준 95% CI입니다.

| metric | mean | 95% CI | n_originals | n_undefined |
| --- | --- | --- | --- | --- |
| next_mse_norm | 0.2606 | [0.2228, 0.3107] | 8 | 0 |
| direction_cosine | 0.6906 | [0.6432, 0.7324] | 8 | 121 |
| relative_magnitude | 1.0843 | [0.9291, 1.2972] | 8 | 121 |
| rollout_h2 | 0.0566 | [0.0479, 0.0666] | 8 | 24 |
| rollout_h4 | 0.1679 | [0.1401, 0.1997] | 8 | 72 |
| sibling_difference_gain | 0.6364 | [0.6074, 0.6594] | 8 | 0 |
| support_swap_degradation | 0.4618 | [0.3394, 0.5754] | 8 | 0 |

seed 0/1/2의 known-test next MSE는 0.2606 / 0.2825 / 0.2197이고 sd는 0.0319입니다. 이 값은
위 bootstrap CI와 다른 양이며, 두 폭이 비슷하므로 seed 하나만 보고 model을 비교하면
안 됩니다. `n_undefined`는 zero/noise norm 때문에 cosine·ratio를 만들지 않은 cell 수와
layer 범위를 넘은 rollout step 수입니다.

### 데이터 효율 (E4 형식, toy)

original-group 단위 고정 nested subset이고 normalization도 각 subset 안에서 다시
계산했습니다.

| train originals | known-test next MSE | cosine | sibling gain | swap deg. |
| --- | --- | --- | --- | --- |
| 6 (25%) | 0.4243 | 0.3398 | 0.2412 | 0.1669 |
| 12 (50%) | 0.3450 | 0.5185 | 0.5334 | 0.3639 |
| 24 (100%) | 0.2606 | 0.6906 | 0.6364 | 0.4618 |

원문 수가 늘면 모든 지표가 단조로 좋아집니다. 24 originals에서도 아직 포화하지 않았으므로,
실자료에서 원문 수가 지배적인 요인일 가능성이 있습니다.

이 결과는 **synthetic 검증 자료**에서 나온 것입니다. 실제 LLM Page와 실제 perturbation에
대한 주장은 E2/E3에서만 할 수 있습니다.
