# DATA_PROTOCOL

label, provenance, 실행 protocol을 한 곳에 모읍니다. model input과 label/evidence
metadata는 항상 분리합니다: predictor는 Page만 보고, label은 loss와 evaluator만 봅니다.

## 1. 데이터 경로

| 단계 | 산출물 | 담당 |
| --- | --- | --- |
| `prepare-data` | DeepMath 후보 + split 배정 | 로컬/서버 |
| `import-pairs` | 검증된 original-variant pair (freeze) | 로컬/서버 |
| `collect-outcomes` | prompt별 outcome counts | 서버 GPU (로컬은 적재만) |
| `build-labels` | pair drop / panel label store | 로컬/서버 |
| `extract` | Page store (`state`, `updates`) | 서버 GPU (로컬은 tiny 검증) |
| `train` / `evaluate` | checkpoint, 지표 | 로컬 CPU(toy) / 서버 GPU |

Primary source는 `zwhe99/DeepMath-103K`
(https://huggingface.co/datasets/zwhe99/DeepMath-103K)입니다. MathGAP/GSM 경로는 구현·
저난도 대조용 **legacy**입니다. AIMO 자료와 고난도 외부 benchmark는 학습 기본 경로에서
제외합니다. 이미 여러 번 확인한 137개 AIMO 자료를 untouched test라고 부르지 않습니다.

## 2. DeepMath field 사용 규칙

이 프로젝트가 읽는 field는 `question`, `final_answer`(필수)와 `difficulty`, `topic`,
`r1_solution_1/2/3`(선택)뿐입니다.

- `r1_solution_*`은 대상 LLM의 행동 기록도, perturbation도 아닙니다. **predictor 입력,
  prompt, robust label 생성에 넣지 않습니다.** adapter는 존재 개수만 기록합니다.
- `topic`과 `difficulty`는 curator / split / evaluator metadata입니다. predictor input에
  넣지 않습니다.
- DeepMath native difficulty를 MATH Level 1~5와 같은 척도로 취급하지 않습니다.
- 대학원 수준이라는 이유만으로 AIMO와 가깝다고 간주하지 않습니다.

후보 선별은 자족적이고 그림 없이 이해 가능하며 final answer 채점이 신뢰성 있는 범위를
우선합니다 (algebra / number theory / combinatorics·discrete / geometry). 그림·외부 자료
참조가 의심되면 제외하고 이유를 남깁니다. 서버의 첫 후보 규모는 약 300 originals로
설정 가능하며, 이는 **후보 수**이지 확보된 labeled pair 수가 아닙니다.

benchmark 겹침은 표면 n-gram/exact 검사만 구현했습니다. **완전한 decontamination은 검증하지
못했습니다** (`benchmark_overlap`의 `decontamination_verified`는 항상 False).

## 3. Split

```
calibration / train / validation / held_out_original / harder
```

배정 단위는 original입니다. 같은 original과 그 variants/seeds는 한 split에만 들어갑니다.
`harder`는 native difficulty 상위 구간에서 고르고, difficulty가 없는 row를 harder로 추측해
넣지 않습니다. 알 수 없는 metadata는 추측해 채우지 않습니다.

## 4. Perturbation

기본 경로는 **검증된 pair import**입니다. 필요한 field는
`original_id`, `variant_id`, `original_text`, `variant_text`, `original_answer`,
`variant_answer`, `semantic_validation_evidence`, `source_revision`입니다.

- evidence가 없으면 `semantic_valid`를 verified로 올리지 않습니다.
- **정답이 같다는 이유만으로 semantic-valid로 판정하지 않습니다.**
- 정답이 다르면 `answer_mapping`에 명시된 대응이 있어야 합니다.

자동 생성은 두 가지만 적용합니다.

1. `formatting`: allowlist된 표기 치환 (`dfrac_to_frac`, `tfrac_to_frac`,
   `drop_left_right`, `collapse_spaces`). 목록 밖의 규칙은 오류입니다.
2. `alpha_rename`: 단일 문자 변수 이름만 바꾸며, 예약 기호·단위 토큰·순서/지시어 토큰·
   이름 충돌·질문 문장 변경을 검사합니다. 하나라도 걸리면 `unknown`(검증 대기)로 남깁니다.

substring replace나 sentence shuffle을 무조건 적용한 뒤 의미보존이라고 표시하지 않습니다.
**일반적인 수학적 동치 검증은 구현하지 않았습니다.**

수치·조건·연산을 본질적으로 바꾸는 변형은 `stress` namespace에 두고 meaning-preserving
primary와 섞지 않습니다. 대회 제출 사용 가능 여부는 `usability` 필드로 따로 관리하며
기본값은 `research_only`입니다.

후보와 split은 **행동 실행 전에 freeze**합니다 (`FrozenPairStore`, hash로 변경 감지).
모델의 실패 여부를 보고 perturbation을 다시 만들지 않습니다.

## 5. Outcome counts

prompt 하나당 다음을 저장합니다.

```
C, W, X, U_score, infra_error, not_started,
planned_trials, completed_trials, termination_reason, policy_hash
```

- 판정된 outcome은 C와 W뿐입니다. `X`(cap-hit), `U_score`(채점 불가), `infra_error`,
  `not_started`는 모두 unresolved이며 **W로 합치지 않습니다**.
- 같은 original의 실행 결과는 여러 variants에서 참조하지만 `prompt_id`가 같으면 같은
  관측이므로 중복 집계하지 않습니다.
- `counts`는 음이 아닌 정수이고 **합이 정확히 `planned_trials`** 와 같아야 합니다. 기록되지
  않은 planned slot은 불완전 입력으로 거부하고, 명시적 import 규칙
  (`fill_missing_as_not_started`)으로만 `not_started`로 채웁니다.
- `completed_trials`(generation 종료 수)와 `n_resolved`(점수 확정 수 = C + W)는 다릅니다.
  `not_started`는 completed로 세지 않습니다.
- merge mode를 구분합니다.
  - `new_only`: 새 prompt만 추가
  - `exact_continuation`: 같은 trajectory를 이어받아 미확정만 채움. slot 단위 trajectory
    증거(`request_id`, `seed`, `prompt_hash`, `policy_hash`, `token_prefix_hash`)가 양쪽에
    있어야 하고, 증거가 없는 기존 aggregate record에는 이력을 만들어내지 않습니다
  - `fill_not_started`: **미시작 slot만** 채우는 delta 기록. 완료된 slot(C/W/X/U_score/
    infra_error)은 건드리지 않습니다
  - `separate_cohort`: 독립 재실행. primary 기록을 바꾸지 않고 별도 cohort로 보존합니다
  `X`만 새 독립 generation으로 바꿔 기존 완료 기록과 합치지 않습니다.
- `scorer_version`이 다른 기록을 같은 store에 섞지 않습니다. 이전 결과를 새 scorer 결과로
  덮어쓰지 않습니다.

## 6. Behavior label

계획된 trial이 모두 판정 가능할 때만 point estimate를 씁니다.

```
p_original_hat = C_original / N_original
p_variant_hat  = C_variant  / N_variant
d_hat_observed = p_original_hat - p_variant_hat        # signed, [-1, 1]
```

이는 **sampling estimate**이며 정확한 population probability가 아닙니다. counts와
uncertainty를 함께 보존합니다.

미확정이 남으면 bounds만 저장합니다.

```
lower = C / N
upper = (C + unresolved) / N          # 미확정은 정답일 수도 있으므로 upper에 포함
pair drop bounds = (o_lower - v_upper, o_upper - v_lower)
```

따라서 `planned=4, C=1, 나머지 3 미확정`이면 bounds는 `[0.25, 1.0]`입니다.
`N = 0`이면 확률과 bounds가 모두 **undefined(None)** 입니다.

이 구간을 95% CI라고 부르지 않습니다. **midpoint나 0으로 바꿔 supervised target을 만들지
않습니다.** 미확정 pair는 behavior loss에서 제외하지만, Page와 semantic validity가 유효하면
flow에는 쓸 수 있습니다. 제외율과 class/difficulty별 coverage를 보고합니다.

### Panel coverage와 max-drop

`expected_members`는 **frozen candidate manifest 전체**에서 만듭니다. 측정 기록이 없는
variant도 기대 구성원으로 남으므로 partial panel이 complete로 표시되지 않습니다. coverage는
세 단계로 구분해 기록합니다.

```
page_members    : Page가 존재하는 variant
outcome_members : 행동 측정 기록이 존재하는 variant
actual_members  : 점수가 확정되어 signed drop label이 생긴 variant
```

정의가 같은 signed pair drops, 같은 model/policy, 명시된 같은 panel이 모두 대응할 때만
max-drop을 계산합니다. partial panel의 관측 최대값을 full-panel target으로 쓰지 않습니다. pair drop과
같은 counts에서 파생되면 이중 감독이 되므로 **기본적으로 pair regression만 켜고**
max-drop은 diagnostic으로 보고합니다 (`train.use_max_drop: false`). 독립적인 panel-only
target이 있을 때만 해당 loss를 켭니다.

### Binary robust label

- 출처와 정의가 있는 제공 label은 해당 단위로 사용합니다.
- DeepMath 정답·난이도·R1 풀이로 robust label을 만들지 않습니다.
- 원본부터 못 푼 경우를 자동 non-robust로 만들지 않습니다.
- 단일 C→W나 4/4 성공을 robustness 인증으로 만들지 않습니다.
- 연구용 binary labeling은 별도 frozen definition(`definition_id` + `source`)이 있을 때만
  활성화합니다. 정의가 없으면 robust label은 **null**로 유지합니다.
- **label 0은 실제 label이며 missing이 아닙니다.** missing을 0이나 False로 바꾸지 않습니다.
- original-panel non-robust label을 모든 variant의 negative pair label로 복사하지 않습니다.

서로 다른 LLM·thinking mode·sampling policy의 label을 섞지 않습니다 (`policy_hash`로
검사합니다). Qwen3-4B Page에 다른 모델의 label을 붙이지 않습니다.

## 7. 필수 provenance metadata

| 항목 | 위치 |
| --- | --- |
| original_id, variant_id, split, panel_id | `OriginalGroup`, `PairLabel` |
| semantic_valid (verified / rejected / unknown) | `PairLabel`, `PairCandidate` |
| source dataset / revision / row id | `DeepMathRow`, `PairCandidate` |
| model / tokenizer revision | Page `provenance` |
| thinking / template / sampling / scorer / budget hash | `policy_hash`, `protocol_hash` |
| Page backend / dtype / extractor hash | Page `provenance` |
| expected panel members와 실제 coverage | `PanelCoverage` |
| native outcomes와 label source/version | `PromptOutcome`, `PairLabel` |

## 8. 실행 protocol

새 DeepMath 경로는 research thinking profile을 씁니다.

```
model_id = Qwen/Qwen3-4B
enable_thinking = true
do_sample = true
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0
```

이는 Qwen thinking 기반 **연구 profile**이며 공식 AIMO와 동일하지 않습니다. legacy
non-thinking / 256-token 설정은 보존하지만 새 경로에서 조용히 재사용하지 않습니다.

다음 값은 서버 calibration manifest에서 확정·freeze합니다 (확정 전에는 실제 GPU full run이
차단됩니다).

```
max_new_tokens, samples_per_prompt, max_total_context,
numerical_backend, scorer_id, scorer_version
```

- 총 context는 `prompt tokens + generated tokens`로 검사합니다. 모델 한도를 조용히 늘리거나
  truncate하지 않습니다. RoPE scaling을 바꾸면 protocol hash가 달라집니다.
- reasoning을 허용하되 최종 답을 명확히 제출하게 합니다. **생각 중간에 정답 숫자가 나타났다는
  이유로 C 처리하지 않습니다** (thinking 블록 밖의 제출 영역만 채점합니다).
- scorer는 version pin된 exact scorer입니다 (현재 `2`). 지원 범위는 integer / 유한 decimal /
  `a/b`·`\frac{a}{b}`이며 `fractions.Fraction`으로 정확히 비교합니다. 범위를 벗어난 표현은
  **오답이 아니라 채점 불가**입니다. 임의 eval이나 untrusted expression 실행, LLM judge를
  쓰지 않습니다.
- 최종 답은 `\boxed{...}`(중첩 brace 지원) -> `final answer:` 라벨 순서로만 찾습니다.
  임의의 마지막 숫자를 답으로 쓰지 않습니다. 같은 우선순위 후보가 여러 개면 전부 동치일 때만
  채택하고, 하나라도 다르면 `U_score`입니다.
- thinking 블록은 tag 상태를 순서대로 훑어 판정합니다. `</think>`가 `<think>`보다 먼저 나오면
  prompt/template이 열어 둔 것으로 봅니다. generated text에 `<think>`가 있다고 가정하지
  않으며, tag가 없다는 이유로 thinking 중간 내용을 최종 답으로 채점하지 않습니다.
- cap-hit / 운영 중단 / 채점 모호를 W로 합치지 않습니다. 중간 checkpoint의 unresolved와
  명시된 최종 scoring deadline을 구분하며, 최종 cap에서의 실패 점수 정의는 protocol의
  `final_cap_failure_policy`로 따로 명시합니다.
- 미완료 generation을 completed-wrong으로 기록하지 않습니다.

## 9. Toy synthetic label (E0b)

로컬 검증용 toy label은 다음 규칙으로 만든 **synthetic 값**이며 실제 LLM robustness가
아닙니다.

1. 고정 fragility 방향 `f`를 seed로 만듭니다 (label 생성에만 쓰고 model input에는 넣지
   않습니다).
2. variant delta에서 `f` 성분을 제거한 뒤, panel 패턴이 정한 만큼만 다시 넣습니다
   (`robust` / `mixed` / `fragile` / `improve`).
3. `target_drop = tanh(proj_gain * <delta, f> / delta_scale)`을 원본 정답률에서 빼
   variant 정답률을 만들고, `planned_trials`에 맞춰 정수 count로 양자화합니다.
4. 실제 label은 그 counts에서 `pair_drop`으로 계산합니다 (실제 pipeline과 같은 경로).
5. 일부 variant는 `X` + `U_score`를 넣어 미확정으로 만들고, identity variant는 behavior
   label 없이 flow 대조로만 씁니다.
6. panel robust label은 `max_j signed_drop_j < robust_threshold`이며, definition_id
   `synthetic_toy_panel_max_drop_below_threshold`로 frozen 처리합니다.

label·ID·namespace를 predictive feature로 직접 넣지 않습니다.
