# EXPERIMENTS

Stage 1의 dataset, split, screening, 비교군, metrics, 실험 순서를 정리합니다.
명세(계획), 구현(코드에 있는 것), 검증(실제로 돌린 것)을 구분해 적습니다.

## 1. 연구 목적과 경계 (v2)

현재 primary는 **행동 감독으로 유용한 내부 변화 representation을 학습하는 것**입니다.
robust/non-robust 또는 correctness drop을 감독 신호로 쓰고, Flow prediction은 실제 내부
전개를 설명하게 하는 auxiliary입니다.

```
L = L_behavior + 0.1 * L_flow
```

이전 목표(screened-stable pair만으로 flow를 학습하고 예측 오차가 failure와 연결되는지
후속 검사)는 **legacy baseline**으로 보존합니다. 용어 정리:

| 용어 | 지금의 뜻 |
| --- | --- |
| primary | supervised behavior + flow joint training (`train.task=joint`) |
| behavior-only | 같은 core로 behavior head만 학습 (`train.task=behavior`) |
| legacy flow-only | stable-only anomaly 접근, v1 설정 (`train.task=flow`) |

경계:

- Flow는 성공·실패 양쪽의 유효한 train pair에서 학습합니다. sibling label 조합을
  제한하지 않습니다 (stable-stable / stable-failed / failed-failed 모두 허용).
- **prediction residual 자체를 non-robust probability로 쓰지 않습니다.**
- concept / recipe / 중요 layer / Gram matrix를 target으로 주지 않습니다.
- 완전한 비지도학습이 아니라 outcome-supervised representation learning입니다.
- correctness, robustness, uncertainty, causal mechanism을 구분합니다.
- 특정 label을 잘 예측했다고 수학적 구조나 인과성을 발견했다고 쓰지 않습니다.
- 4/4 성공은 population robustness 인증이 아닙니다.
- 데이터 선별에는 correctness supervision이 쓰이지만 predictor input에는 정답·concept·
  recipe·robustness label이 들어가지 않습니다.
- 초기 대상은 Qwen3-4B 하나입니다. Cross-model transfer는 후속 목표이며 이번 결과로
  주장하지 않습니다.
- head를 구현했다는 사실과 실제 label로 학습했다는 사실을 구분합니다. behavior
  supervision이 전혀 없는 실행은 joint 성공으로 보고하지 않습니다.

## 2. 실험 순서

| 실험 | 내용 | 상태 |
| --- | --- | --- |
| E0 | (legacy) flow-only CPU synthetic end-to-end, leakage/rollout/reload 검증 | 실행 완료 |
| E0b | CPU synthetic **behavior+flow joint** end-to-end, label routing/gradient/reload 검증 | 실행 완료 |
| E1 | 서버에서 DeepMath 후보·split freeze, 검증된 pair 확보, Page extraction 검증 | SERVER_PENDING |
| E1c | 작은 calibration으로 model policy / scorer / budget 확정 (runtime·VRAM·완료율·원본 정답률·label uncertainty 확인) | SERVER_PENDING |
| E2 | 반복 행동 측정 -> label 연결 -> behavior-only / joint 학습 | SERVER_PENDING |
| E3 | 새 original·새 perturbation·higher difficulty 평가 | SERVER_PENDING |
| E4 | 독립 original 25/50/100% 학습곡선으로 data efficiency 비교 | 기계는 구현, toy에서 실행. 실자료는 SERVER_PENDING |

300개 원문을 곧바로 대규모 generation으로 시작하지 않습니다. E1c calibration을 먼저 돕니다.

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

## 4. 행동 측정 (E1 / E2)

새 DeepMath 경로는 research thinking profile을 씁니다. 자세한 규칙과 provenance 요구사항은
[DATA_PROTOCOL.md](DATA_PROTOCOL.md)에 있습니다. 여기에는 실험 설계에 필요한 부분만 적습니다.

```
model_id = Qwen/Qwen3-4B, enable_thinking = true, do_sample = true
temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0
max_new_tokens / samples_per_prompt / max_total_context /
numerical_backend / scorer_id / scorer_version  ->  needs_calibration (E1c)
```

이는 Qwen thinking 기반 **연구 profile**이며 공식 AIMO 평가와 동일하지 않습니다. legacy
non-thinking / 256-token screening 설정(MathGAP 경로)은 저난도 대조용으로 보존하지만 새
경로에서 조용히 재사용하지 않습니다.

slot outcome은 분리해서 셉니다.

| 코드 | 뜻 |
| --- | --- |
| `C` | 정답 |
| `W` | 오답 |
| `X` | cap-hit / 미완료. `W`로 합치지 않습니다. |
| `U_score` | 채점 불가 (제출 형식 없음) |
| `infra_error` | 실행 오류 |
| `not_started` | 시작되지 않음 |

**중요한 변경**: primary loader에서 "C4 pair만 남김" 필터를 제거했습니다. 성공·실패·성능
유지·개선 사례를 모두 보존하고, 미확정 pair만 behavior loss에서 제외합니다 (flow에는 쓸 수
있습니다). 실패하거나 미확정인 pair를 성공할 때까지 다시 생성하지 않습니다.

Qwen adapter는 random-init tiny config로 CPU에서 검증합니다. 실제 4B weights는 로컬에서
내려받지 않으며, real behavior measurement와 4B numerical audit는 SERVER_PENDING입니다.

## 5. 비교군

주 비교는 behavior 예측입니다.

| 이름 | task | 설명 |
| --- | --- | --- |
| `constant` | behavior | constant/prior baseline (상수 drop + 상수 robust logit, 2 params) |
| `raw_change` | behavior | 작은 raw-change baseline (scalar 3개만 봅니다) |
| `behavior_m0` | behavior | original-only. variant 관측·길이·validity·ID·count가 들어가지 않습니다 |
| `behavior` | behavior | 같은 Looped core, behavior-only |
| `joint` | joint | behavior + flow. **primary** |
| `joint_loop1` / `joint_untied4` | joint | shared block 1회 / 독립 block 4개 |
| `loop4` | flow | legacy flow-only (auxiliary 비교) |

`persistence` / `linear` / `m0` / `loop1` / `untied4`(flow)와 original 25/50/100% 학습곡선도
그대로 유지합니다. M0에는 variant count / 길이 / label validity가 새지 않아야 하며, panel
크기·source·난이도의 영향은 별도로 보고합니다.

## 6. Metrics

### Pair drop

- MAE / Huber(delta=0.1) — training과 evaluation이 같은 helper를 씁니다
- 같은 original 내 variant별 예측 차이(spread)와 순위 상관(Spearman, 동점은 average rank,
  한쪽이 상수면 undefined)
- sampling uncertainty: 저장된 drop bound 폭
- supervised coverage: 실제 drop label이 있는 pair 비율

### Robust classification (label이 있을 때만)

- accuracy, balanced accuracy
- 양 class recall / precision
- AUROC, Brier, log-loss
- confusion counts (tp / fp / tn / fn)
- 한 class로만 예측했는지(`predicted_all_one_class`)와 majority-class accuracy

### Flow (auxiliary)

next-update raw/normalized MSE, direction cosine, relative magnitude, 2/4-step rollout
error, sibling-difference gain, support-swap degradation.

### 집계와 해석 규칙

- 집계 단위는 original panel입니다. 같은 원문의 variants나 layers를 독립 문제로 세지
  않습니다.
- 불확실성은 original-group bootstrap으로 보고하고 training seed variance와 구분합니다.
- zero/noise norm의 cosine과 relative magnitude는 undefined로 셉니다.
- **같은 panel의 variants 순서를 바꾸는 것은 set pooling의 permutation invariance 검사이며
  pair 정보 사용 여부의 강한 ablation이 아닙니다.**
- **Pair support-swap**은 target variant를 고정한 채 같은 original의 다른 variant Page를
  대입해 pair-drop 변화를 봅니다. **유효한 모든 target variant**를 평가하고 원문 안에서 평균한
  뒤 원문 단위 동일 가중치로 집계합니다. panel 단순 재정렬의 불변성과 구분합니다.
- 학습되지 않은 head의 지표는 canonical 결과로 보고하지 않습니다 (`robust_head_status`가
  `untrained`이면 `robust_classification`은 null이고 원시 score는 debug 출력에만 있습니다).
- pair-derived max-drop은 별도 학습 head가 아니라 학습된 pair 예측에서 계산한 diagnostic입니다
  (`max_drop_source`).
- 모두 non-robust로 예측해 accuracy가 높아진 것을 성공이라고 하지 않습니다.
- 새 원문 / 새 perturbation / higher difficulty를 각각 따로 보고합니다.
- 관측 성능을 causal mechanism이나 전체 AIMO 제출 성능으로 확대하지 않습니다.
- train만 좋아지거나 `behavior_m0`와 차이가 없으면 pair-specific 행동 신호를 배웠다는
  근거가 부족한 것으로 보고합니다.

## 7. 로컬 재현 명령

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/
.venv/bin/ruff check src tests

# E0b (v2 primary): behavior + flow joint
.venv/bin/aimo check      --config configs/toy_behavior.yaml
.venv/bin/aimo make-toy   --config configs/toy_behavior.yaml --out runs/e0b_pages
.venv/bin/aimo preflight  --config configs/toy_behavior.yaml --run-id e0b_pre
.venv/bin/aimo train      --config configs/toy_behavior.yaml --run-id e0b_behavior \
  --set model.name=behavior --set train.task=behavior
.venv/bin/aimo train      --config configs/toy_behavior.yaml --run-id e0b_joint \
  --set model.name=joint --set train.task=joint
.venv/bin/aimo evaluate   --config configs/toy_behavior.yaml --run-id e0b_joint \
  --set model.name=joint --set train.task=joint \
  --splits validation known_test unseen_perturbation_test harder_test
.venv/bin/aimo predict-behavior --config configs/toy_behavior.yaml --run-id e0b_joint \
  --set model.name=joint --set train.task=joint --split known_test --n-originals 4

# legacy flow-only (select_metric도 함께 바꿉니다)
.venv/bin/aimo train --config configs/toy_behavior.yaml --run-id e0b_flow_legacy \
  --set model.name=loop4 --set train.task=flow --set train.select_metric=total
```

비교군은 `--set model.name=<name> --set train.task=<task> --run-id e0b_<name>`으로 같은
방식으로 돌리고, data efficiency는 `--set data.subset_fraction=0.25`로 만듭니다. 결과는
`aimo evaluate --compare runs/e0b_joint runs/e0b_m0 ...`로 한 표에 모읍니다.

seed는 역할별로 분리되어 있으므로 자료를 고정한 채 학습 초기화만 바꿀 수 있습니다.

```bash
.venv/bin/aimo train --config configs/toy_behavior.yaml --run-id e0b_joint_t1 \
  --set model.name=joint --set train.task=joint \
  --set run.seeds.train=1 --set run.seeds.sampler=1
```

이 경우 `split_hashes`와 `train_subset_hash`가 seed 0과 동일해야 합니다
(`tests/test_joint_train.py::test_train_seed_does_not_change_data_or_labels`).

label 경로(실자료 기준)는 다음 순서입니다.

```bash
# 준비된 registry가 있으면 frozen split을 그대로 씁니다.
.venv/bin/aimo prepare-data     --config configs/deepmath.example.yaml --prepared <registry_dir>
# 없으면 pinned snapshot에서 후보를 고릅니다 (parquet은 batch 단위로 읽습니다).
.venv/bin/aimo prepare-data     --config configs/deepmath.example.yaml --input <snapshot>
.venv/bin/aimo import-pairs     --config configs/behavior.yaml --input <verified_pairs.jsonl>
# 서버 결과 적재 (미시작 slot만 채울 때는 --merge-mode fill_not_started)
.venv/bin/aimo collect-outcomes --config configs/behavior.yaml --from-file <outcomes.jsonl>
.venv/bin/aimo build-labels     --config configs/behavior.yaml
```

검수 수정 내역과 CPU 검증 범위는 [FIXES.md](FIXES.md)에 있습니다.

## 8. E0b 실행 결과 (로컬 CPU, 실측)

설정: toy synthetic `H=32, L=6, P=4`, train 24 originals / 72 pairs, validation·각 test split
8 originals / 24 pairs, 100 epochs 상한, patience 15, `AdamW lr 3e-4`, effective batch 8
originals, `L = L_behavior + 0.1 * L_flow`, `use_max_drop: false`,
`select_metric: behavior_total`. **toy label은 명시적인 synthetic 생성 규칙으로 만든 값이며
실제 LLM robustness가 아닙니다.** 이 실행의 목적은 성능 우열이 아니라 end-to-end 학습·label
routing·gradient·reload 검증입니다.

train split의 label coverage (실측):

```
pairs 72, drop label 있는 pair 54 (제외율 25.0%)
제외 사유: identity_no_behavior_label 8, unresolved_trials 10
panels 24, robust label 있는 panel 15 (나머지는 partial coverage -> null)
panel-only max-drop target 0개  ->  max-drop loss는 꺼짐 (diagnostic만)
```

### known-test: pair drop과 robust classification

`dropMAE`는 낮을수록, `rankCorr`(panel 안 순위 상관)와 `acc`/`auroc`는 높을수록 좋습니다.
`spread`는 같은 panel 안 예측의 표준편차, `swapMAE+`는 pair support-swap 후 MAE 증가량입니다.

| run | task | params | dropMAE | rankCorr | spread | swapMAE+ | acc | balAcc | auroc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `constant` | behavior | 2 | 0.2735 | n/a | 0.0000 | 0.0000 | 0.400 | 0.500 | 0.500 | 0.2500 |
| `raw_change` | behavior | 8 | 0.8108 | -0.125 | 0.0182 | 0.0352 | 0.400 | 0.500 | 0.000 | 0.3775 |
| `behavior_m0` | behavior | 186,147 | 0.3352 | n/a | 0.0000 | 0.0000 | 0.600 | 0.583 | 0.667 | 0.2494 |
| `behavior` | behavior | 186,147 | 0.2370 | 1.000 | 0.2692 | -0.0362 | 0.800 | 0.833 | 1.000 | 0.1166 |
| **`joint`** | joint | 186,147 | **0.1638** | 0.875 | 0.2179 | **0.0521** | 0.800 | 0.833 | 1.000 | 0.1531 |
| `joint_loop1` | joint | 186,147 | 0.1908 | 0.875 | 0.1438 | 0.0207 | 0.800 | 0.833 | 1.000 | 0.1423 |
| `joint_untied4` | joint | 583,587 | 0.2284 | 1.000 | 0.1824 | -0.0524 | 0.800 | 0.833 | 0.833 | 0.1416 |

읽는 방법과 한계:

- `behavior_m0`는 panel 안 예측 spread가 정확히 0이고 rank 상관이 undefined입니다. variant
  관측을 받지 않으므로 **variant를 구분할 수 없다**는 것이 수치로 확인됩니다.
- `joint`는 `behavior`(0.237)와 `behavior_m0`(0.335)보다 pair drop MAE가 낮고, pair
  support-swap을 하면 MAE가 늘어납니다(+0.052). 즉 target variant의 Page를 실제로 쓰고
  있습니다.
- **`constant`(2 params)가 `behavior_m0`(186k params)보다 MAE가 낮습니다.** variant 정보가
  없는 큰 모델은 상수 예측보다 나쁠 수 있습니다. MAE 하나만 보고 비교하면 안 됩니다.
- `joint_untied4`는 parameter가 3.1배인데 `joint`보다 나쁩니다. shared block 4회가 더
  많은 parameter를 쓰지 않고 같거나 더 좋은 수준에 도달합니다.
- `behavior`(behavior-only)의 `swapMAE+`는 음수(-0.036)입니다. 이 설정에서는 support-swap이
  오차를 줄였다는 뜻이므로, behavior-only만으로는 pair-specific 사용의 증거가 약합니다.
- robust classification의 labeled panel은 split당 5~7개뿐입니다. accuracy와 AUROC는 이
  표본 크기에서 해석해야 하며, synthetic 규칙이 학습 가능하도록 만들어졌기 때문에 높은
  값이 나옵니다. **실제 LLM robustness 성능이 아닙니다.**

### joint의 split별 지표와 bootstrap CI

original-group bootstrap 200회, 95% CI입니다.

| split | dropMAE [CI] | rankCorr [CI] | swapMAE+ [CI] | robust acc (n_panels, pos/neg) | AUROC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 0.1030 [0.064, 0.150] | 0.875 [0.625, 1.000] | 0.0657 [0.014, 0.121] | 1.000 (7, 4/3) | 1.000 | 0.0162 |
| known_test | 0.1638 [0.118, 0.216] | 0.875 [0.747, 1.000] | 0.0521 [-0.042, 0.145] | 0.800 (5, 2/3) | 1.000 | 0.1531 |
| unseen_perturbation | 0.1001 [0.068, 0.129] | 0.750 [0.500, 1.000] | 0.0274 [-0.017, 0.078] | 1.000 (7, 4/3) | 1.000 | 0.0034 |
| harder | 0.1224 [0.074, 0.201] | 0.625 [0.500, 0.875] | 0.1066 [0.031, 0.203] | 1.000 (6, 3/3) | 1.000 | 0.0000 |

`swapMAE+`의 CI는 known_test와 unseen에서 0을 포함합니다. pair 정보 사용의 증거는
validation과 harder에서 더 뚜렷하고, known_test에서는 이 표본 크기로 단정할 수 없습니다.
`drop_bound_width`(sampling uncertainty) 평균은 split별로 0.016~0.037입니다.

### Flow auxiliary (같은 자료, known_test)

| run | next MSE | cosine | rollout h2 | sibling gain |
| --- | --- | --- | --- | --- |
| `joint` | 0.4231 | 0.156 | 0.0796 | 0.054 |
| `joint_loop1` | 0.4707 | 0.069 | 0.0917 | 0.004 |
| `joint_untied4` | 0.4771 | 0.041 | 0.0964 | 0.007 |
| `loop4` (legacy flow-only) | 0.2571 | 0.629 | 0.0532 | 0.580 |

flow-only로 100 epochs 학습한 legacy가 flow 지표에서는 joint보다 훨씬 좋습니다. joint에서는
flow가 가중치 0.1의 auxiliary이고 behavior 지표로 checkpoint를 골랐으며 43 epoch에서
멈췄습니다. **joint가 flow 예측에서도 더 낫다고 주장하지 않습니다.**

### 데이터 효율 (E4 형식, toy)

original-group 단위 고정 nested subset이고 normalization도 각 subset 안에서 다시 계산합니다.

| train originals | dropMAE (known_test) | rankCorr | spread | robust acc |
| --- | --- | --- | --- | --- |
| 6 (25%) | 0.4045 | 0.750 | 0.0201 | 0.000 |
| 12 (50%) | 0.2766 | 0.750 | 0.0162 | 0.400 (한 class로만 예측) |
| 24 (100%) | 0.1638 | 0.875 | 0.2179 | 0.800 |

원문 수가 줄면 panel 안 예측 spread가 거의 0으로 붕괴하고 robust head가 한 class로만
예측합니다. 실자료에서도 원문 수가 지배적인 요인일 가능성이 있습니다.

### Seed variance (bootstrap CI와 다른 양)

`joint`을 seed 0/1/2로 돌린 known_test 결과입니다.

| seed | epochs (best) | dropMAE | rankCorr | swapMAE+ | robust acc | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 43 (27) | 0.1638 | 0.875 | 0.0521 | 0.800 | 1.000 |
| 1 | 33 (17) | 0.0935 | 1.000 | 0.0543 | 0.571 | 0.917 |
| 2 | 27 (11) | 0.1782 | 0.375 | -0.0063 | 0.833 | 0.889 |
| mean ± sd | | 0.1452 ± 0.0453 | 0.750 ± 0.331 | 0.0334 ± 0.0344 | 0.735 ± 0.143 | 0.935 ± 0.058 |

- 이 값은 seed 0의 original-group bootstrap CI(`dropMAE` [0.118, 0.216])와 **다른 양**입니다.
  두 폭이 비슷하므로 seed 하나만 보고 model을 비교하면 안 됩니다.
- **주의**: `run.seed`는 synthetic 자료 생성과 학습 초기화를 함께 바꿉니다. 따라서 위 값은
  순수 training seed variance가 아니라 **data + training seed variance**입니다. 순수 training
  seed variance를 보려면 자료를 고정한 채 초기화만 바꿔야 합니다.
- `swapMAE+`는 seed 2에서 음수(-0.006)입니다. pair support-swap으로 본 pair 정보 사용의
  증거는 이 toy 표본 크기에서 **seed에 걸쳐 안정적이지 않습니다.** 실자료에서 다시
  확인해야 합니다.
- robust accuracy는 labeled panel이 5개뿐이어서 한 panel 차이로 0.2씩 움직입니다.

### 검증된 항목 (실행 기준)

- behavior-only 학습, joint 학습, pair/panel/flow 평가, serialization/reload를 실제로 실행
- 같은 `LoopedCore`에 두 view의 gradient가 누적되고 shared parameter가 optimizer에 한 번만
  등록됨
- behavior forward 뒤에도 flow 예측이 bit 단위로 동일 (future leakage 없음)
- panel 재정렬에 대한 robust probability 변화 최대 `2.8e-09` (permutation invariance)
- 별개 process에서 `predict-behavior`를 두 번 실행해 panel 출력이 완전히 동일
- 학습되지 않은 head(`max_drop`)가 untrained로 표시됨
- label 제외율 25%와 제외 사유가 그대로 보고됨

## 9. E0 실행 결과 (legacy flow-only, 로컬 CPU, 실측)

아래는 v1 flow-only 설정(`configs/toy.yaml`, stable-only 가정)에서 측정한 legacy
baseline입니다. v2 primary 결과와 같은 자료가 아니므로 직접 비교하지 않습니다.
(v2에서 synthetic 생성 규칙이 behavior label을 포함하도록 바뀌었으므로 숫자를 그대로 재현하려면
v1 revision을 쓰세요.)

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
| **loop4 (v1 primary)** | 168,608 | **0.2606** | **0.6906** | 1.0843 | 0.0566 | 0.1679 | **0.6364** | **0.4618** |
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
