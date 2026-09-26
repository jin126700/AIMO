# ARCHITECTURE

확정된 v2 architecture(**Behavior-supervised Looped Transformer + Flow auxiliary**)의
shape, indexing, input/output, attention mask, loop, rollout, loss를 정리합니다. 여기 적힌
내용은 `src/aimo/` 구현과 1:1로 대응합니다.

## 0. 두 view, 하나의 core

```
                  ┌─ Behavior view ── full-page pair readout ─ Z_ij ─┬─ pair signed drop
native vectors ─ LoopedCore                                          └─ panel set pooling ─ robust prob
                  └─ Flow view ───── masked prefix (cut d) ───────── V_hat[d]   (auxiliary)
```

`LoopedCore`(input embedding + injection + shared block)는 **하나의 객체**이고 두 view가
그것을 공유합니다. 독립적인 Transformer를 두 개 만들지 않습니다. 두 view는 서로 다른
forward이며 Behavior의 hidden state나 KV cache를 Flow에 재사용하지 않습니다.

primary objective는 `L = L_behavior + 0.1 * L_flow`입니다. Flow는 내부 전개를 설명하게 하는
auxiliary이고, prediction residual 자체를 non-robust probability로 쓰지 않습니다.
stable-only anomaly 접근(v1 flow-only)은 legacy baseline으로 보존합니다.

## 1. Page 데이터 계약

Batch dimension을 제외하면 관측은 두 tensor입니다.

```
state:   [L + 1, P, H]      # block 경계의 residual stream
updates: [L, P, 2, H]       # block별 stream 기여
```

- stream index는 고정입니다: `0 = Mixer (self-attention)`, `1 = FFN`.
- `P = 17`: 문제 본문의 실제 token landmark 16개 + canonical final prompt token 1개.
- `H`는 원래 model hidden size를 그대로 씁니다. 고정 random32 projection이나 learned
  target encoder를 쓰지 않습니다. Input embedding은 학습하지만 prediction target은
  언제나 native H-space입니다.
- 중복 landmark와 padding은 `valid: [P] bool` mask로 처리합니다. 한 prompt 안에서
  token 위치는 모든 layer에서 고정이므로 mask는 depth와 무관합니다.
- 원본과 변형의 같은 landmark 순번이 같은 수학 개념이라고 가정하지 않습니다.
- 실제 token offset, relative position, valid mask, model/tokenizer/config hash를
  Page provenance에 함께 보존합니다.

block index `d`는 0-based이고 residual identity는 다음과 같습니다.

```
H[d+1] = H[d] + U[d, Mixer] + U[d, FFN]

D[d]   = H_variant[d] - H_original[d]
V[d,c] = U_variant[d,c] - U_original[d,c]

D[d+1] = D[d] + sum_c V[d,c]
```

`H[0]`은 첫 block의 실제 input입니다. Final LayerNorm 이후의 hidden state는 마지막
block output이 아니므로 `H[L]`과 혼동하지 않습니다 (`adapters/qwen.py` 참고).

## 2. Flow view 입력과 미래 leakage 차단

한 forward는 prefix cut `d`에서 다음 update 차이 하나를 예측합니다.

```
Input:  original 전체 Page
        variant state[0:d+1]
        variant updates[0:d]
        cut d, 관측 위치(token offset / relative position), validity
Output: V_hat[d], shape [P, 2, H]
```

`variant state[d+1:]`와 `variant updates[d:]`는 model input 객체(`PredictorInput`)에
아예 들어가지 않습니다. Target은 loss와 evaluator만 가진 별도 객체(`PairBatch`)입니다.

이것은 **original 전체를 참조하는 conditional prediction**입니다. 양쪽의 미래를 모두
모르는 순수 forecasting이 아니고, 완전한 LLM simulator도 아닙니다.

## 2b. Behavior view 입력

Behavior view는 original과 variant의 **전체 Page**를 봅니다.

```
Input:  original 전체 Page (state[0:L+1], updates[0:L])
        variant  전체 Page (state[0:L+1], updates[0:L])
        각각의 실제 token 위치와 validity (original/variant 독립)
Output: Z_ij in R^128  ->  pair signed drop d_hat_ij in [-1, 1]
                       ->  panel set pooling  ->  robust probability
```

정답·topic·difficulty·recipe·ID·sampling counts·label validity는 model input에 들어가지
않습니다 (`BehaviorInput`에 해당 field가 없습니다). Flow의 cut 경로를 재사용하지 않으므로
마지막 variant block이 누락되지 않습니다.

Behavior sequence와 mask:

```
[original reference depth 0..L] [variant depth 0..L] [pair readout query 1]
```

| row | 읽을 수 있는 column |
| --- | --- |
| original reference | original reference만 |
| variant depth `r` | original 전체 + variant depth `<= r` |
| pair readout query | original/variant 전체 관측 |

reference/variant row는 readout query를 읽지 않고, 모든 loop에서 같은 mask를 씁니다.
구현상 Behavior layout의 `cut`을 `L`로 두어 Flow와 **같은 mask 함수**를 재사용합니다.

### Behavior heads

```
Z_ij      = pair_norm(x_K[pair readout cell])            # R^128
d_hat_ij  = tanh(linear_drop(Z_ij))                      # signed drop, [-1, 1]
a_j       = softmax_j(pool_score(Z_ij))                  # masked attention pooling
pooled_i  = sum_j a_j * pool_value(Z_ij)                 # variant 순서 embedding 없음
p_robust  = sigmoid(robust_head(pool_norm(pooled_i)))
D_hat_i   = max(0, max_j d_hat_ij)                       # 별도 대형 head 없음
```

padding slot은 pooling에서 제외되고, 구성원이 없는 panel은 `panel_valid=False`와 NaN으로
표시됩니다. 순서를 바꿔도 panel 출력이 같습니다 (permutation invariance).

학습된 pair score나 pooling weight를 **개별 변형의 causal importance라고 해석하지 않습니다**.
실제 label로 학습되지 않은 head는 checkpoint의 `head_trained` flag로 표시되고, 추론 결과에
untrained 상태가 함께 나옵니다.

## 3. Flow cell 구성과 indexing

sequence는 세 종류의 cell group으로 구성됩니다.

```
[reference]        original depth r = 0..L,   P landmarks
[observed variant] variant  depth r = 0..d,   P landmarks
[next-depth query] depth d,               2P cells (Mixer / FFN)
```

cell feature는 `[state, update_mixer, update_ffn]`를 이어 붙인 `3H` vector입니다.
indexing 규칙은 하나입니다: **depth `r` cell에는 `state[r]`과 이미 완료된 `update[r-1]`만
넣습니다** (`r = 0`이면 update 자리는 0). 따라서 예측 target인 `U[d]`가 `H[d]` cell에
들어가는 일이 없습니다. observed cell은 `updates[0:d]`만 소비합니다.

query cell에는 activation을 전혀 넣지 않습니다. 위치(landmark index, relative
position, depth)와 역할(query, stream) 정보만 들어갑니다.

## 4. Flow attention mask

mask는 depth 단위로 정의되고 같은 depth의 landmarks는 하나의 group으로 처리됩니다.

| row (query 주체) | 읽을 수 있는 column |
| --- | --- |
| reference | reference만 |
| observed depth `r` | reference 전체 + observed depth `<= r` |
| next-depth query | reference 전체 + observed depth `<= d` |

- reference와 observed cell은 query cell을 읽지 않습니다.
- reference는 variant prefix를 전혀 읽지 않으므로, reference를 경유한 간접 leakage도
  생기지 않습니다 (`tests/test_model_leakage.py`가 mask 구조와 활성값 둘 다 확인).
- padding landmark는 key에서 제외합니다. 전부 차단된 row는 self를 열어
  all-masked-row NaN을 막습니다.
- 같은 leakage 방지 규칙을 모든 loop에서 동일하게 적용합니다 (mask는 loop와 무관).

## 5. 확정 Looped architecture

```
native vectors (3H per cell)
  -> layer 간 공유하는 작은 learned input embedding  (value_proj + role/stream/depth/landmark/scalar)
  -> shared pre-LN Transformer block 반복             (동일 parameter 객체 4회 호출)
  -> layer 간 공유하는 native-space readout           (readout_norm + Linear -> H)
```

기본 설정: `d_model = 128`, `heads = 4`, `FFN = 256`, `dropout = 0.1`,
shared pre-LN block 1개를 4회 적용.

loop 수식과 input injection 위치는 다음과 같습니다.

```
x_0     = emb_norm(embed(observations))
x_{k+1} = Block(x_k + inject(x_0))        k = 0 .. 3
V_hat   = readout(readout_norm(x_4))[query cells]
```

- `inject`는 bias 없는 shared `Linear(d_model, d_model)` 하나이며, 매 loop마다 같은
  parameter로 고정 observation embedding `x_0`를 다시 더합니다.
- `blocks`는 tied 설정에서 길이 1인 ModuleList입니다. 독립 block copy 4개를 만들지
  않고 같은 객체를 4회 호출합니다 (`loop1`과 `loop4`의 parameter 수가 동일).
- adaptive halting, 추가 loop sweep, per-loop 전용 module은 넣지 않습니다.
- 단일 `z64` bottleneck을 쓰지 않습니다. 변화 문맥은 sequence hidden states에 남습니다.
- LLM의 layer `d`와 predictor의 loop `k`는 서로 다른 개념입니다. loop 하나가 LLM layer
  하나를 재현한다고 주장하지 않습니다.

### Parameter 수 (실측)

`d_model=128, heads=4, ffn=256` 기준입니다. `core`는 shared block, `input-output`은 공유
input embedding, `heads`는 view별 readout입니다.

| 설정 | model | total | core | input-output | flow head | behavior head |
| --- | --- | --- | --- | --- | --- | --- |
| toy `H=32, L=6, P=4` | `joint` (primary) | 186,147 | 132,480 | 31,872 | 4,384 | 17,411 |
| toy | `joint_untied4` | 583,587 | 529,920 | 31,872 | 4,384 | 17,411 |
| toy | `loop4` (flow only) | 168,736 | 132,480 | 31,872 | 4,384 | - |
| toy | `constant` | 2 | 0 | 0 | - | 2 |
| toy | `raw_change` | 8 | 0 | 0 | - | 8 |
| 예시 `H=2560, L=36, P=17` | `joint` | 1,488,515 | 132,480 | 1,008,128 | 330,496 | 17,411 |
| 예시 | `joint_untied4` | 1,885,955 | 529,920 | 1,008,128 | 330,496 | 17,411 |

`H=2560, L=36`은 크기 감각을 위한 **예시**이며 Qwen3-4B의 확인된 shape가 아닙니다. 실제 값은
서버 extraction에서 확정됩니다 (SERVER_PENDING). `core`와 behavior head는 `H`, `L`에
의존하지 않습니다. behavior head를 추가해도 `joint`는 `loop4`보다 17,411 parameter만 늡니다.

## 6. Rollout

순서는 항상 `normalize -> predict -> inverse normalize -> raw recurrence`입니다.
정규화 좌표에 raw update를 더하지 않습니다.

```
U_hat_variant[d]   = U_original[d] + V_hat[d]                        (raw)
H_hat_variant[d+1] = H_variant_or_pred[d] + sum_c U_hat_variant[d,c]  (raw)
D_hat[d+1]         = H_hat_variant[d+1] - H_original[d+1]
```

- 예측된 state/update를 그대로 다음 prefix에 붙입니다. 최초 cut 이후에는 실제 variant
  future를 다시 사용하지 않습니다.
- horizon 2와 4를 구현합니다. layer 범위를 넘는 step은 만들지 않고, loss와 evaluation
  에서 undefined로 남깁니다.
- 짧은 training rollout의 gradient는 유지합니다 (detach 없음).

## 7. Normalization

scale은 **train originals만으로** 계산해 freeze합니다. 역할별로 따로 보관합니다.

| 이름 | shape | 용도 |
| --- | --- | --- |
| `input_state_scale` | `[L+1]` | predictor 입력 state |
| `input_update_scale` | `[L, 2]` | predictor 입력 update |
| `target_scale` | `[L, 2]` | `V` target |
| `sibling_scale` | `[L, 2]` | `V_a - V_b` |
| `rollout_scale` | `[L+1]` | `D` |

값은 depth별(및 stream별) RMS입니다. `V`는 update와 같은 단위이므로 target/sibling
scale도 original update의 RMS를 씁니다. 이렇게 하면 정규화 오차가 "전형적인 update
크기 대비 상대오차"가 되고, variant 통계를 미리 들여다보지 않게 됩니다.

`floor`(기본 `1e-6`) 아래의 scale은 zero/noise-scale로 보고 **inactive**로 표시해
loss와 metric의 분모에서 제외합니다. 작은 분모로 신호를 과장하지 않습니다.
25/50/100% subset을 쓸 때는 각 training subset 안에서 다시 계산합니다.

## 8. Loss

최종 objective입니다.

```
L = L_behavior + 0.1 * L_flow

L_behavior = BCE(robust logit, robust label)                      # 제공된 label만
           + Huber_0.1(d_hat_ij, signed pair drop)                # 실제 drop만
           [+ Huber_0.1(D_hat_i, panel-only max drop)]            # 독립 target이 있을 때만

L_flow     = L_next + L_within + 0.25 * L_roll
L_next     = normalized MSE(V_hat, V)
L_within   = normalized MSE(V_hat_a - V_hat_b, V_a - V_b)
L_roll     = normalized MSE(D_hat_future, D_future)               # horizon 2 / 4 평균
```

### Behavior 항

- 없는 항은 mask-out하고 각 loss의 valid count를 기록합니다. **label 0은 실제 label이며
  missing으로 오인하지 않습니다.**
- 원문 단위로 같은 가중치를 주고, variants는 group 내부에서 먼저 평균합니다.
- pair drop과 max-drop이 같은 counts에서 파생되면 이중 감독이 되므로 기본은 **pair
  regression만** 켭니다 (`train.use_max_drop: false`). max-drop은 diagnostic으로 보고합니다.
- robust label이 아직 없으면 pair-drop regression이 실제 behavior supervision입니다.
  head를 구현했다는 사실과 실제로 학습했다는 사실을 구분해 보고합니다
  (`trained_heads`, `behavior_supervision_seen`).
- behavior supervision이 전혀 없는 실데이터 실행은 joint 성공으로 보고하지 않습니다
  (summary에 warning이 붙습니다).

### Flow 항

- `a, b`는 같은 original의 서로 다른 variants입니다. **label 조합을 제한하지 않습니다**:
  stable-stable뿐 아니라 stable-failed / failed-failed도 씁니다. variant가 하나인 group은
  within loss만 제외합니다.
- 같은 cut과 common valid cells만 씁니다. `variant_id`가 같은 pair는 collate에서 거부합니다.
- 10% identity example은 flow의 `V = 0` 대조로만 씁니다. identity라고 임의의 robust label을
  만들지 않습니다.
- 원문별로 먼저 평균한 뒤 원문끼리 동일 가중치로 평균합니다.

Clustering / Gram / cosine / TCAV / contrastive concept loss는 넣지 않습니다.

## 9. Training

`AdamW lr 3e-4`, `weight_decay 1e-3`, effective batch 8 originals, gradient clip 1,
최대 100 epochs, validation patience 15. microbatch와 gradient accumulation을
지원하며, 한 forward는 하나의 cut만 다룹니다 (cut별로 묶어 sub-forward를 돕니다).

`task`는 `flow`(legacy/auxiliary) / `behavior` / `joint`(primary) 중 하나입니다. joint에서는
behavior forward와 flow forward의 gradient가 **같은 LoopedCore**에 누적되고, shared
parameter는 단일 optimizer에 한 번만 등록됩니다 (`model.parameters()`가 공유 parameter를
중복 반환하지 않습니다). 메모리를 줄이려고 flow gradient를 detach하지 않습니다.

checkpoint 선택에는 validation만 씁니다. 어떤 지표를 쓰는지는 시작 전에
`train.select_metric`으로 고정합니다 (기본: flow는 `total`, behavior/joint는
`behavior_total`). test나 가장 잘 나온 seed로 모델을 고르지 않습니다.

checkpoint에는 schema/model/task version, config/data/split/stats/label-policy hash,
optimizer/RNG state, head 학습 여부를 함께 저장합니다. 같은 config로 resume하면 끊기지 않은
학습과 같은 값이 나옵니다. v1 flow-only checkpoint는 schema version이 없으므로 **거부**되며
새 supervised 모델로 조용히 해석하지 않습니다.

| version | 값 |
| --- | --- |
| checkpoint schema | `aimo-checkpoint-v2` |
| page store schema | `aimo-page-store-v2` |

## 10. 비교군

### Behavior (primary 비교)

| 이름 | task | 설명 |
| --- | --- | --- |
| `constant` | behavior | constant/prior baseline. 상수 drop과 상수 robust logit만 학습 (2 params). |
| `raw_change` | behavior | 작은 raw-change baseline. scalar 3개(최종 state 차이 크기, update 차이 누적 크기, valid landmark 비율)만 봅니다. |
| `behavior_m0` | behavior | original-only. variant 관측을 original으로 대체하므로 variant state/길이/validity/ID/count가 들어가지 않고 sequence 길이도 variant 수와 무관합니다. |
| `behavior` | behavior | 같은 Looped core, behavior-only. |
| `joint` | joint | behavior + flow. **primary**. |
| `joint_loop1` / `joint_untied4` | joint | shared block 1회 / 독립 block 4개. |

### Flow (legacy / auxiliary)

| 이름 | 설명 |
| --- | --- |
| `persistence` | `V_hat = 0`. parameter 0개. |
| `linear` | 작은 linear conditional baseline (`D[d]`, `U_original[d]`, depth one-hot -> `V_hat[d]`). |
| `m0` | original-only flow. |
| `loop1` / `loop4` / `untied4` | shared block 1회 / 4회 / 독립 4개. |

같은 input/output/data/loss 조건에서 비교합니다. 구조적으로 다른 부분은 명시합니다:
`persistence`와 `constant`는 입력을 보지 않고, `raw_change`는 core가 없으며, `m0` 계열은
variant 관측을 sequence에서 제외합니다.
