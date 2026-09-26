# FIXES

검수 기준 commit `0448258`에서 발견된 문제와 그 수정을 정리합니다. **무엇을 고쳤는지**,
**어떤 CPU 검증을 실제로 했는지**, **어떤 코드가 구현됐지만 서버 검증 전인지**를 구분합니다.

architecture는 바꾸지 않았습니다: shared LoopedCore ×4, Behavior/Flow의 동일 parameter 공유,
native Page와 17 landmarks, Behavior full-page / Flow masked-prefix 분리,
`L = L_behavior + 0.1 * L_flow`, 기존 모델 크기와 input injection을 모두 유지했습니다.

## 1. Scorer와 thinking parser (`adapters/qwen.py`, 신규 `scoring.py`)

| 문제 | 수정 |
| --- | --- |
| `split_thinking`이 generated text에 `<think>`가 있다고 가정. prompt가 열고 suffix에 `</think>`만 있으면 unclosed로 오판 | tag 상태를 순서대로 훑는 상태 기계로 교체. `</think>`가 `<think>`보다 먼저 나오면 prompt가 열어 둔 것으로 **추론**하고, `thinking_already_open` flag도 받습니다 |
| 미완료·non-thinking·special token만 생성된 경우 구분 없음 | `no_thinking` / `closed` / `unclosed` / `empty` 상태와 `generated_token_ids` + `special_token_ids` 검사 추가 |
| boxed parser가 nested brace 미지원 (`\boxed{\frac{1}{2}}` 추출 실패) | brace matching scanner `extract_boxed`로 교체. 닫히지 않은 `\boxed{`는 버립니다 |
| float 비교가 큰 정수를 같다고 판정 (`9007199254740993 == 9007199254740992`) | `fractions.Fraction` 기반 exact 비교로 교체. 지원 범위는 integer / 유한 decimal / `a/b`·`\frac{a}{b}`이며, 그 밖은 **오답이 아니라 채점 불가(unsupported)** 입니다. 임의 eval을 하지 않습니다 |
| 채점 불가와 오답이 섞임 | `compare_answers`가 correct / wrong / unsupported 세 값을 돌려주고, slot 분류가 unsupported를 `U_score`로 보냅니다 |
| 복수의 상충하는 최종 답 규칙 없음 | 같은 우선순위 후보가 여러 개면 **전부 동치일 때만** 채택하고, 하나라도 다르면 `conflicting` -> `U_score` |
| scorer version 고정 안 됨 | `SCORER_VERSION`을 `2`로 올리고, `PromptOutcome`/`OutcomeStore`/`LabelStore`가 scorer version을 들고 다니며 **다른 version의 기록을 같은 store에 섞지 않습니다** |

CPU 검증: `tests/test_scoring.py` 15개 (1/2 == 0.5, 큰 정수 구분, thinking 중간 정답 배제,
nested boxed, unsupported -> U_score, 상충 답 규칙, prompt-opened suffix, 여러 thinking 블록,
special token만 생성 등).

## 2. Outcome / panel / label 연결 (`labels.py`, `data.py`)

| 문제 | 수정 |
| --- | --- |
| counts 검증 없음 (음수·비정수 허용, 합과 planned 불일치 허용) | 음이 아닌 정수 검증과 `sum(counts) == planned_trials` 강제. 기록되지 않은 slot은 **불완전 입력으로 거부**하고, `from_dict(fill_missing_as_not_started=True)`라는 명시적 import 규칙으로만 `not_started`로 채웁니다 |
| `planned=4, C=1`만으로 bounds가 `[0.25, 0.25]` | 미확정 slot이 upper에 모두 포함되어 `[0.25, 1.0]`이 됩니다 |
| `N=0`의 확률·bounds가 `0.0` | `p_hat()`과 `bounds()`가 `None`(undefined)을 돌려주고, `pair_drop`이 `no_planned_trials`로 제외합니다 |
| generation 완료 수와 score 확정 수가 섞임 | `completed_trials`(generation 종료)와 `n_resolved`(C+W)를 분리하고 정의를 docstring에 적었습니다. `not_started`는 completed로 세지 않습니다 |
| `build_label_store`가 outcome이 있는 variant만 expected_members로 사용 | **frozen candidate manifest 전체**에서 expected_members를 만들고, outcome이 없는 variant도 panel에 남깁니다 |
| Page/outcome/score coverage 구분 없음 | `PanelCoverage`가 `page_members` / `outcome_members` / `actual_members`를 따로 들고 `page_ratio`·`outcome_ratio`·`missing_members`를 보고합니다 |
| id 대응·중복 검사 없음 | variant_id 중복, `variant_id == original_id`, 한 original이 두 panel_id에 매핑되는 경우를 거부합니다 |
| partial panel의 관측 최대값이 full-panel max-drop이 될 수 있음 | coverage가 complete가 아니면 `None`, complete여도 expected 집합과 drop 수가 정확히 일치해야 계산합니다 |
| `request_resume`가 기존 기록을 독립 generation으로 **교체** (X가 새 성공으로 대체됨) | 모드를 다시 설계했습니다: `fill_not_started`는 **미시작 slot만** 더하고 완료 slot(X 포함)을 보존하며, 독립 재실행은 `separate_cohort`로 primary와 분리해 보존합니다 |
| `exact_continuation`이 counts 단조성만으로 인정 | slot 단위 trajectory 증거(`request_id`, `seed`, `prompt_hash`, `policy_hash`, `token_prefix_hash`)를 요구합니다. 증거가 없는 기존 aggregate record는 **이력을 만들어내지 않고 거부**합니다 |
| store 저장이 비원자적이고 lock 없음 | `store_lock` + `atomic_write_json`으로 저장. 중복 slot·충돌 기록·policy/scorer 혼합을 조용히 통과시키지 않습니다 |
| `attach_labels`가 label 내부 policy만 확인 | 실제 Page provenance(`policy_hash`/`model_hash`/`tokenizer_hash`/`config_hash`)와의 대응을 검사합니다. **빈 hash를 동일성의 근거로 쓰지 않고**, `source == "synthetic"` fixture만 명시적으로 허용합니다 |

CPU 검증: `tests/test_labels.py` 30개, `tests/test_runners.py`의 store/lock 항목.

## 3. Microbatch loss와 metric (`losses.py`, `train.py`, `evaluate.py`)

| 문제 | 수정 |
| --- | --- |
| missing label이 있으면 microbatch 크기에 따라 effective objective가 달라짐 (측정값 0.70 ~ 1.17) | `TermValue`가 mean 대신 **합 + count**를 들고, effective batch 전체의 항별 global denominator로 한 번만 나눕니다. denominator는 model forward 없이 `behavior_term_counts` / `flow_term_counts`로 먼저 셉니다 |
| validation이 다른 방식으로 집계 | validation도 같은 sum/count 방식이며, 각 항은 **자기 항의 count로만** 나눕니다 (서로 다른 항의 count를 합치지 않습니다) |
| evaluation의 Huber가 training과 스케일이 다름 (|d|=0.5에서 0.45 vs 0.045, 10배) | 공용 `huber_elementwise(delta=0.1)`을 training과 evaluation이 함께 씁니다 |
| Spearman이 동점을 average rank로 처리하지 않음 | `_average_rank`로 동점 평균 처리, 한쪽이 상수면 undefined. AUROC도 같은 helper를 씁니다 |
| support-swap이 slot 0만 평가 | 유효한 **모든 target variant**를 평가하고 원문 안에서 평균한 뒤 원문 단위 동일 가중치로 집계합니다 |
| 학습되지 않은 robust head의 원시 score가 canonical 출력에 나감 | canonical 출력은 `null` + `robust_head_status: "untrained"`이고 원시 score는 `debug_*`로만 분리합니다. 가중치가 0인 항은 학습된 head로 표시하지 않습니다 |
| pair-derived max-drop이 학습 head처럼 보임 | `max_drop_source: "derived_from_pair_predictions"`로 표시하고 loss routing에서 제외된 상태를 유지합니다 |

CPU 검증: `tests/test_aggregation.py` 11개. dropout OFF, 같은 effective batch에서
microbatch=1/2/3/6의 loss 차이 `<1e-6`, microbatch=1/2/4의 gradient 상대 차이 `<1e-5`.
label이 전혀 없는 항은 NaN gradient 없이 제외됩니다.

## 4. Seed / checkpoint / validation (`config.py`, `train.py`, `page.py`, `data.py`)

| 문제 | 수정 |
| --- | --- |
| seed 하나가 자료·split·초기화·sampler·평가를 모두 결정 | `run.seeds`에 `data` / `split` / `train` / `sampler` / `eval`을 분리하고, 빈 항목은 `run.seed`(master)로 채웁니다 |
| data hash가 ID 목록만 사용 (같은 ID에 다른 값이 와도 구분 못 함) | `Page.content_fingerprint()`(shape/dtype/bytes, 캐시)와 label 값을 포함한 `data_hash()`로 교체. ID만 보는 `split_structure_hash()`는 진단용으로 남겼습니다 |
| 선택 지표에 유효 label이 없어도 조용히 학습 시작 | 학습 전에 `check_supervision`으로 fail-fast합니다. 활성화한 behavior objective에 supervision이 전혀 없을 때도 멈춥니다 |
| flow-only 실행이 joint 성공처럼 보고될 수 있음 | `is_joint_result` 필드와 명시적 note를 남깁니다. flow-only는 `train.task='flow'`를 고른 경우에만 돌아갑니다 |
| checkpoint가 torch CPU RNG만 저장 | torch CPU + Python `random` + NumPy를 저장하고 복원합니다. CUDA 상태는 가능할 때 저장하되 `cuda_verified: false`로 표시합니다 |
| schema 호환성 미명시 | checkpoint schema를 `aimo-checkpoint-v3`로 올리고, version이 없거나 v2인 checkpoint는 **명시적으로 거부**합니다 (조용히 재해석하지 않습니다) |

CPU 검증: `tests/test_joint_train.py` 21개. train seed만 바꿔도 data/split/label hash 동일,
data seed를 바꾸면 달라짐, 내용만 바뀌어도 hash가 달라짐, RNG 보존, resume 재현성.

## 5. 서버용 실행 경로 (코드만 수정, 실제 실행 없음)

### A. Device (`runtime.py`, `data.py`, `train.py`)

`run.device`가 실제로 적용됩니다. `resolve_device`가 CUDA/MPS 부재 시 **명시적 오류**를 내고
CPU로 조용히 fallback하지 않습니다. model과 NormStats만 미리 옮기고, batch는
`PairBatch.to` / `PanelBatch.to` / `FlowInput.to` / `BehaviorInput.to`로 **microbatch만**
옮깁니다. hashing은 항상 CPU에서 합니다. 로컬은 CPU 경로만 검증했고
**실제 CUDA 검증은 SERVER_PENDING**입니다.

### B. Page extractor (`adapters/qwen.py`)

hook 안에서 필요한 landmark만 골라 CPU float32로 복사합니다(이전에는 모든 layer의 전체
`[T, H]`를 보관). LM head를 지나지 않도록 backbone(`model.model`)을 직접 호출해
`[T, vocab]` logits을 만들지 않습니다. tiny random-init 모델에서 기존 추출 값과 일치,
residual identity `7.5e-09`, `H[0]` = 첫 block 입력, `state[L]` = pre-final-norm,
landmark/validity 처리를 확인했습니다.

### C. 실행 진입점 (`collect.py`, `adapters/qwen.py`, `cli.py`)

`collect-outcomes`와 `extract`가 무조건 SERVER_PENDING으로 끝나는 placeholder가 아니라 실제
runner를 씁니다.

- `collect_outcomes(backend, plans, profile, ledger, guard)`: config(calibration) 검증,
  slot ledger dedup/resume, 총 context 검사, slot 분류, `PromptOutcome` 집계,
  오류·중단 전달
- `run_page_extraction(model, requests, ledger, guard)`: dedup/resume, 예산·중단, residual
  identity 검사
- backend는 주입식입니다. 로컬은 `MockGenerationBackend`로 전체 경로를 검증하고,
  `QwenGenerationBackend` / `load_real_qwen`은 calibration과 실제 weights를 요구하며
  로컬에서는 SERVER_PENDING으로 fail-fast합니다. **실제 weights를 로드하거나 generate하지
  않았습니다.**

### D. 예산 / preflight (`runtime.py`, `cli.py`)

`BudgetGuard`와 `StopRequest`를 training / collection / extraction runner에 연결했습니다.
누적 175분에서 신규 시작을 막고 180분에서 종료하며, **예외·중단 시에도 elapsed를 저장**합니다
(`__exit__`의 finally). 단계나 resume가 누적 예산을 리셋하지 않습니다. 소유한 subprocess만
`terminate_owned`로 종료하고 다른 process는 건드리지 않습니다. step 사이 검사만으로 hard
stop을 보장한다고 쓰지 않았고, 작업 쪽에서 `should_stop()`을 polling합니다.

`preflight`는 선택한 task와 `data.source`에 필요한 항목만 blocker로 봅니다. DeepMath 경로에
MathGAP 설정을 요구하지 않고, `gpu_full_run_allowed`는 calibration만이 아니라 snapshot·
transformers 조건까지 함께 봅니다.

CPU 검증: `tests/test_runners.py` 17개 (mock backend collection, dedup/resume, 중단 전달,
context 초과, tiny 추출 값 일치, 예산 175/180과 예외 시 적립, 소유 subprocess만 종료,
store lock).

## 6. Dataset 연결 (`adapters/deepmath.py`, `adapters/perturbation.py`, configs)

| 문제 | 수정 |
| --- | --- |
| parquet 전체를 `to_pylist`로 펼침 | `ParquetFile.iter_batches(batch_size, columns=...)`로 **필요한 column만 batch 단위**로 읽습니다. r1 풀이 본문은 즉시 버리고 개수만 남깁니다 |
| 준비된 registry를 직접 읽는 경로 없음 | `load_prepared_registry`가 `inputs/answers/metadata/pairs/splits`를 읽고, `splits.json`이 있으면 **그 frozen split을 그대로** 씁니다 (`assign_splits(frozen=...)`로 덮어쓰지 않습니다) |
| 외부 semantic status 변환 규칙 없음 | `verified_by_construction`(생성 규칙이 근거) / `verified_by_existing_evidence`(evidence 필수) / `pending` / `rejected`를 내부 schema로 옮기며 근거를 보존합니다. **pending·rejected는 verified로 승격하지 않습니다** |
| dataset/result 경로 | `configs/deepmath.example.yaml`에 dataset `/data1/Data/AIMO/Datasets`, result `/data1/HKM/result/AIMO/HKM/aimo_v2/<run_id>/`를 반영했습니다. 로컬에서 경로 존재를 확인하거나 만들지 않았습니다 |

CPU 검증: 작은 JSONL/parquet fixture만 사용했습니다. 새 후보 선정·split 생성·screening 계획
실행은 하지 않았습니다.

## 7. 실제로 완료한 CPU 검증

```
pytest tests/          224 passed
ruff check src tests   All checks passed
git diff --check       clean
```

짧은 synthetic end-to-end (`configs/toy_behavior.yaml`, 6 epochs):

| 단계 | 결과 |
| --- | --- |
| `check` / `make-toy` | ok |
| behavior-only 학습 | 6 epochs, best 5, val 0.6448, heads `{pair_drop: True, robust: True}` |
| joint 학습 | 6 epochs, best 5, val 0.6443, `is_joint_result: True` |
| 평가 | known_test pair drop MAE 0.260, panel permutation shift `5.96e-08`, support-swap 8 원문 |
| `predict-behavior` 2회 (별개 process) | panel 출력 완전 동일 |
| `resume` | 같은 config로 이어서 실행 ok |

숫자 비교 실험(3 seed 학습곡선 등)은 이번 범위가 아니므로 하지 않았습니다. README의 E0b
수치는 **이 수정 이전**에 측정한 값이며 그대로 보존했습니다.

## 8. 서버에서만 검증할 남은 항목 (SERVER_PENDING)

- 실제 CUDA 실행과 CUDA RNG 재현성
- 실제 Qwen3-4B weights 로드·generation·Page extraction과 4B numerical audit
  (`transformers >= 4.51` 필요)
- DeepMath pinned snapshot revision과 실제 자료 기반 후보/split 확정
- thinking profile calibration 값 확정 (`max_new_tokens`, `samples_per_prompt`,
  `max_total_context`, `numerical_backend`, `scorer_id`, `scorer_version`)과
  `final_cap_failure_policy`
- 실제 검증된 original-variant pair 확보
- binary robust label의 frozen definition (없으면 robust label은 null이고 pair-drop
  regression이 실제 supervision)
- 독립적인 panel-only max-drop target
- 실제 자료에서의 예산 소진 동작 (로컬은 fake clock과 mock worker로만 검증)
