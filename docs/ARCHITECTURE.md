## Current candidate backend: Stage-E native representation (`stage_e_v1`)

상태: `IMPLEMENTED / CPU-TOY-TESTED / REAL-MODEL-UNVERIFIED`. 실제 Qwen3.5-4B hook, backward,
수치 허용치, GPU VRAM, 1시간 benchmark는 아직 실행하지 않았다. 코드 구현이 끝난 것과 연구
검증·대회 제출 준비가 끝난 것은 다르다.

새 backend는 legacy 경로와 **분리**되어 있다. LoopedCore / Behavior / E-FLOW-1 / LRT-v1과 그
checkpoint는 비교용으로 그대로 남고, 새 경로는 `aimo native-extract`, `aimo stage-e`,
`aimo submission-*` 명령과 config의 `stage_e` section으로만 선택한다. 공통 좌표를 검증하는
경로에 비선형 reasoning network(LoopedCore)를 넣지 않는다.

### 관측: macro × all-token NativePage (`native_page.py`, `native_extract.py`)

- schema `aimo-native-page-v1`. 실제 raw `H`, prompt의 **모든 token**, macro boundary의 raw
  state `S_{b_g}` [G+1, T, H], macro별 `M_g`, `F_g` [G, T, H], final norm 이후 hidden [T, H],
  token span map(segment + 문제 text 기준 char offset)을 담는다.
- 기존 17-landmark Page는 자동 변환하지 않는다 (`SCHEMA_INCOMPATIBLE`). projection 좌표나
  17 landmark를 native `H`나 all-token으로 취급하지 않는다.
- actual update는 residual 차이로 정의한다: `M_l = h_mid - h_in`, `F_l = h_out - h_mid`.
  `h_mid`는 layout이 정한 module의 입력에서 읽는다 (`model_registry.py`).
  - `pre_norm` (Qwen2/Qwen3/Qwen3.5 hybrid): `post_attention_layernorm` 입력.
    linear-attention(`linear_attn`)과 full-attention(`self_attn`) Mixer를 모두 같은 방식으로
    처리한다.
  - `post_norm` (OLMo-2/3 계열): `mlp` 입력. update는 norm이 적용된 sublayer 출력이다.
  - Mixer/FFN module의 hook 출력은 actual update와의 **일치 audit**에만 쓴다. residual 차이로
    정의한 identity `S_{b_{g+1}} - S_{b_g} ≈ M_g + F_g`는 구조상 거의 자명하므로, 비자명한
    검사는 module disagreement다 (module 밖 gate가 있으면 검출된다).
- raw final boundary는 마지막 layer 출력 hook, final norm 이후 hidden은 backbone `norm` 출력
  hook에서 따로 읽는다. `output_hidden_states[-1]`을 raw boundary로 가정하지 않는다.
- landmark sampling, projection 전 pooling, 무단 truncation이 없다. 길이가 넘치면
  `DATA_LIMIT`. 가변 T는 오른쪽 padding + `attention_mask`로 처리한다.
- `AuditSink`는 offline audit용 완전 관측, `StreamSink`는 제출용으로 update가 생길 때마다
  `E_g`로 투영해 누적하고 raw Page를 보관하지 않는다 (두 경로의 latent 일치를 test로 고정).
- Qwen3.5-4B: 32 layers → `b_g = floor(g·32/8)` = 4개씩 8 macro. 다른 model은 실제
  `num_hidden_layers`를 읽는다 (예: 28층이면 [0,3,7,10,14,17,21,24,28]).
- 알 수 없는 `model_type`은 추측하지 않고 `SCHEMA_INCOMPATIBLE`. `trust_remote_code=False`.

### Stage-E representation (`stage_e.py`)

```
B_g ∈ R^[H×r], B_gᵀB_g = I            (QR, 부호 고정)
L_g ∈ R^[r×r]                          (lower-triangular, 대각 exp → 항상 invertible)
E_g = L_g B_gᵀ
z_state[g,t] = E_g s_out[g,t] + a_g     z_mixer = E_g M_g,  z_ffn = E_g F_g
predicted_logits[g,t] = D z_state[g,t] + b      (모든 stage 공유 linear decoder)
```

- 같은 `E_g`, `a_g`를 macro 입구/출구에 쓰면 `z_out - z_in = z_mixer + z_ffn`. 서로 다른
  stage encoder의 latent 차이는 이 식과 다르다.
- projector는 `B_g B_gᵀ`. 일반 `E_g`에 대해 `E_gᵀE_g`를 projector로 쓰지 않는다.
  native lift는 `E_g⁺ = B_g L_g⁻¹` (`E_g E_g⁺ = I`, `E_g⁺ E_g = B_g B_gᵀ`).
- `L_g`는 gauge다. z는 바뀌지만 projector와 lift된 native 변화는 바뀌지 않는다.
- decoder에는 원문, token ID, attention network가 들어가지 않는다.
- encoder는 stage별로 다르고 모든 문제에 같은 frozen encoder를 쓴다. refitting / test-time
  adaptation 없음.
- 초기 rank grid `4, 8, 16, 32`. rank와 lambda는 discovery Dev에서만 고른다.
- parameter 수: `G·H·r + G·r² + G·r + V·r + V` (`StageE.parameter_report`).
- Factorized Joint는 Stage-E freeze 뒤의 보조 분석으로만 둔다. raw depth×hidden
  factorization을 primary에 강제하지 않는다.

### Teacher와 loss (`npr.py`, `sensitivity.py`)

```
L = L_NPR + λ · L_sensitivity
```

- teacher는 target model의 실제 native next-token distribution이다. behavior label,
  correctness, robustness, max_drop, raw update reconstruction은 discovery loss에 없다.
- `L_NPR`: teacher top-k(k=32) + OTHER의 **coarse categorical KL**. OTHER는 나머지
  vocabulary의 실제 질량이며 top-k를 재정규화하지 않는다. full KL이라고 부르지 않는다.
  teacher/student normalizer는 vocabulary chunk로 계산하고 student 쪽은 chunk마다 gradient
  checkpoint를 써서 T×V activation을 저장하지 않는다. `full_vocab_kl`은 audit 경로다.
- valid-token mask와 문제별 정규화: 문제 안에서 token/stage 평균 후 문제끼리 평균한다.
- `L_sensitivity`: 고정 seed probe `v_j`에 대한 `a_gj = ∂(v_jᵀy)/∂S_g`의 비포착 비율
  `Σ||a - aBBᵀ||² / max(Σ||a||², ε)`. VJP 한 번과 r차원 projection으로 계산하고
  (`||a||² - ||aB||²`), `a`는 detach, `create_graph=False`. target weight는 학습하지 않는다.
  λ=0 baseline을 항상 grid에 둔다. 제출 entry point는 이 module을 import하지 않는다.
- 입력: prompt-only가 기본이고 target model이 생성한 짧은 native prefix를 지원한다. dataset
  제공 solution은 prefix로 쓰지 않는다. 숫자·부호·연산자·조건 contrast probe와 style/format
  control은 continuation audit용이며 정답 annotation이나 수학 relation label이 아니다.

### Response sketch (`sketch.py`)

```
R ∈ R^[V×q] (q=64, 고정 seed), Rᵀ1 = 0
y     = Rᵀ log_softmax(l) = Rᵀ l = (RᵀW_lm) N(h_final) + Rᵀ b_lm
y_hat = (RᵀD) z_state + Rᵀ b
```

- 이름은 **logit-contrast sketch**이며 full NPR가 아니다. folded head는 cached target
  model에서 process 안에서 한 번 만들고 재사용한다. logit softcap 같은 비선형 head는 거부한다.
- 압축 사각지대 audit: 독립 sketch `R'`, full-vocabulary KL, contrast-probe continuation KL.
- CPU toy 관찰(연구 성능 아님): coarse KL이 작아도 sketch 상대 오차가 1을 넘을 수 있었다.
  coarse KL은 OTHER 안의 tail logit을 제약하지 않기 때문이다. server에서는 discovery Dev의
  `sketch_relative_error`를 residual feature 사용 전에 먼저 본다 (held-out으로 고르지 않는다).

### 제출 predictor (`submission.py`, `mp_views.py`, `robust_predictor.py`)

- 기본 feature 3개: Original–MP prompt-end native divergence(full-vocab JS), all-token
  unexplained response-change residual의 max, 같은 residual의 RMS.
  residual = `||Δy - (RᵀD)Δz_g|| / √q` (대응 endpoint 쌍, 모든 stage). raw latent norm은
  instability로 쓰지 않는다. 길이·confidence·alignment coverage는 control feature다.
- MP view는 수식 span 밖 whitespace만 바꾸는 2개(`whitespace_collapse`, `line_reflow`).
  비공백 문자열(semantic signature)이 원문과 같지 않으면 버린다. LLM paraphrase나 긴
  generation은 기본 경로에 없다.
- token index 차이를 쓰지 않는다. edit map과 tokenizer offset으로 **같은 semantic prefix에서
  끝나는 complete-span endpoint**만 대응시키고 coverage와 실패 이유를 기록한다. prompt-end
  anchor는 template suffix token이 같을 때만 쓴다.
- predictor는 작은 L2 logistic이며 JSON artifact(pickle 없음)다. transform, 정규화 강도,
  threshold는 root-grouped CV에서만 고른다. official score로 encoder/rank/feature를 다시
  맞추지 않는다.
- `are_robust(model_id, reasoning_effort, problems) -> list[bool]`: 순서·길이 보존, 실제
  Python bool. registry는 공식 Small Models Track 4개 ID를 그대로 쓰고 adapter/artifact 상태를
  구분한다. `openai/gpt-oss-120b`는 main-only이고 adapter가 없다.
- 시간: 모든 model·문제 합계 3600초, 내부 목표 2700초. import 시점 monotonic clock을 여러
  호출에 걸쳐 공유한다. model load 전과 문제별 forward 전에 비용을 예측하고, 부족하면
  검증된 prior fallback을 쓴다. fallback 사용률과 fallback 포함 accuracy를 따로 보고한다.
  artifact가 없는 model에 fallback한 것을 shared encoder 검증으로 보고하지 않는다.
- frozen forward만 한다: gradient, intervention, fitting, 다운로드, 외부 API 없음.
  같은 model의 effort 호출에서는 load한 model을 재사용한다. artifact의 model revision과
  실제 load된 `_commit_hash`가 다르면 fallback한다.

### 연구 대조군과 평가 (`stage_e_experiment.py`)

| 대조군 | 의미 |
| --- | --- |
| `random_r_trained_decoder` | 고정 random orthonormal B + 학습 decoder |
| `pca_r_trained_decoder` | train split PCA B + 학습 decoder |
| `npr_only_lambda0` | λ=0 |
| `full_hidden_readout` | B=I (rank H) readout, gradient capture 상한 1 |
| `output_compression_reduced_rank` | native head(RᵀW)의 top-r 방향, 출력 압축 기준 |
| `local_fit_oracle_support_query` | 문제별 support token으로 fit, query token에서 평가 (test-time fit 참고값) |

- A: held-out coarse KL, full-vocab KL audit, sketch / 독립 sketch 오차, gradient capture.
- B: topic / difficulty로 root 전체를 떼어낸 group, MP formatting view family.
- C: natural donor(다른 root)의 prompt-end state를 한 macro boundary에 넣은 full patch 효과를
  `E⁺(z_d - z_r) = BBᵀΔs` patch가 회복하는 정도. control: random subspace, complement,
  norm-matched random, rank-matched PCA, site-matched random donor. stage 하나만 patch하며
  여러 stage clamp를 자연적 causal mechanism으로 단정하지 않는다.
- native fidelity, causal use, 공식 robustness 예측 성능은 따로 보고한다. 실패한 held-out을
  보고 rank/feature를 다시 고르는 자동 반복은 없다 (`select_on_dev`는 Dev 결과만 받는다).

### Schema와 호환

| 대상 | schema | 비고 |
| --- | --- | --- |
| native page store | `aimo-native-page-v1` | tensor는 `weights_only=True`, sha256 확인 |
| Stage-E checkpoint | `aimo-stage-e-v1` | legacy Looped(`aimo-checkpoint-v3`) / LRT(`aimo-lrt-v1`)와 서로 load 불가 |
| 제출 artifact | `aimo-submission-artifact-v1` | JSON + tensor, model revision pin 필수 |
| legacy Page | `aimo-page-store-v2` | 그대로 유지, native 경로에서는 `SCHEMA_INCOMPATIBLE` |

`Config.hash()`는 `stage_e` section이 기본값이면 그 section을 빼고 계산한다. 이 section이
생기기 전의 Looped / LRT / FP32 run config hash(resume guard)가 바뀌지 않는다.

## Current primary: Flow representation learning → frozen robustness probe

현재 primary는 **E-FLOW-1**이다. 직전 FP32 behavior regression의 전체 81 valid pairs 중
77개가 zero-drop이었고, known-test 10 originals / 19 pairs에서 zero baseline이 Joint보다
우수했다. 이것은 architecture 자체의 실패를 입증하지 않는다.

- Stage 1: 행동 label 파일을 읽지 않는 Page-only loader. 기존 train의 ID hash로
  flow_train 28 / flow_dev 4를 freeze한다. Flow normalization은 flow_train만 사용한다.
- Shared LoopedCore ×4, d_model128 / heads4 / FFN256 / dropout0.1을 유지한다.
  objective는 L_next + L_within + 0.25 L_roll이며 flow_dev L_flow로 checkpoint를 선택한다.
- Stage 2: encoder 완전 동결, full-page observed valid hidden mask-mean z128,
  panel mean/std256 (population std; singleton std0). 새로운 pair query/head는 없다.
- Zero/train-mean Flow baseline, original-only M0, RawChange, random-init core,
  train original-label shuffle20회와 작은 L2 logistic probe를 비교한다.
- 실데이터에서 검증된 criterion이 없으면 threshold를 만들지 않는다.
  class 또는 criterion 부족은 ROBUSTNESS_PROBE_DATA_LIMIT이며 BCE/AUROC/bootstrap을
  강제로 실행하지 않는다. feature extraction과 Flow validation/diagnostics는 진행한다.
- Flow decodability ≠ robustness; robustness prediction/probe separability ≠ causal mechanism.
  representation stability ≠ robustness. activation patching 이전에는 causal evidence를 주장하지 않는다.
- 아래 Behavior-primary / joint / legacy 지침은 **historical v2 baseline**에 적용한다.
  기존 Behavior head와 loss는 삭제하지 않는다. 이 E-FLOW-1 정의가 현재 primary에 우선한다.

실행:
```sh
python -m aimo flow-representation-experiment --source /path/to/frozen_fp32_run --run-dir /path/to/new_run
CUDA_VISIBLE_DEVICES=0 python -m aimo flow-representation-experiment --run-dir /path/to/new_run --execute-gpu
```
첫 명령은 Page audit와 split/spec freeze이며 GPU inference를 하지 않는다.
둘째 명령은 기존 predictor만 학습한다. 기존 source 결과는 읽기 전용이다.
source/control hashes와 독립 run의 config를 보존하고 automatic push를 하지 않는다.

## Current candidate primary: LRT-v1 (Looped Relational Transport)

상태: `IMPLEMENTED / CPU-VALIDATED / SERVER-UNTESTED`. 아래 historical section은 그대로
보존하며, server 결과가 없는 상태에서 LRT를 validated/successful이라고 쓰지 않는다.

### 좌표: MacroPage-8 (derived view)

Fine Page가 source of truth이고 바뀌지 않는다. MacroPage는 runtime에 deterministic하게
만드는 derived view다 (`src/aimo/macro_page.py`).

```
b_g              = floor(g * L / G),  g = 0..G,  G <= L,  strictly increasing
S[g, p]          = fine.state[b_g, p]                          [G+1, P, H]
U_macro[g, p, c] = sum_{l=b_g}^{b_{g+1}-1} fine.updates[l,p,c]  [G, P, 2, H]
S[g+1]           = S[g] + U_macro[g,:,0] + U_macro[g,:,1]       (source tolerance 안)
```

layer 수를 hard-code하지 않으므로 24 / 32 / 36 / 48 layer 모두 같은 G=8 coordinate를 쓴다.
fingerprint는 `source_page_fingerprint` + `macro_schema` + `n_macro` + `boundaries`를 담고,
같은 Fine Page + 같은 G이면 같은 값이다. source Page는 mutate하지 않는다.

`path_energy[g,p,c] = Σ_l ||fine.update[l,p,c]||₂`는 macro sum의 cancellation을 보는
**diagnostic 전용**이며 encoder input / decoder / training loss에 들어가지 않는다.

### Relation target

```
ΔU[g, p, c] = V.updates[g,p,c] - O.updates[g,p,c]
```

encoder가 보는 primary activation은 ΔU뿐이다. 누적량 `V.state - O.state`는 앞선 macro
relation의 누적이고 held-out macro 정보의 indirect trace가 될 수 있으므로 encoder input으로
쓰지 않는다. absolute original state/update는 decoder context에서만 쓴다.

### Support / Query fold

G=8의 query fold는 연속 macro pair다: `[0,1] [2,3] [4,5] [6,7]`. 나머지 6 macro가 support이고
각 pair에 대해 4 fold를 rotate한다. **future prediction이 아니다** — support에는 query보다
뒤의 macro가 들어갈 수 있으므로 이름은 cross-macro relational transport이며 causal transport라고
쓰지 않는다.

### 모델 (`src/aimo/lrt.py`)

```
A_U : H -> 32   (Mixer/FFN relation과 original update가 공유. stream embedding으로 구분)
A_S : H -> 32   (original state)

relation cell = cell_proj(A_U(ΔU_norm[g,p,c])) + site(depth, landmark, stream, position)
x_{k+1}       = Block(x_k + inject(x_0))          동일 SharedBlock 객체 4회 호출
z_rel         = Linear(128, 16)(LN(valid-mask mean pooling))         -> R^16

a[g,p,c]      = coeff(z_rel, A_S(O.state[g,p]), A_U(O.updates[g,p,c]), site)   -> R^16
ΔU_hat[g,p,M] = B_M(a),   ΔU_hat[g,p,F] = B_F(a)      B_c = Linear(16, H, bias=False)
```

`d_model 128 / heads 4 / FFN 256 / dropout 0.1 / loops 4 / relation_dim 16 / decoder_rank 16`.
site metadata embedding은 std 0.02로 초기화한다 (기본 `nn.Embedding` std 1.0은 adapter를 지난
relation 신호를 약 9배로 압도해 LayerNorm 뒤에 relation 내용이 묻힌다).

low-rank 의미:

```
historical U4-like :  ΔU ≈ P z                      (original computation과 무관한 fixed 변환)
LRT-v1             :  ΔU_hat = B_c g(z_rel, O.state, O.update, site)
```

low-dimensional output subspace를 유지하면서 relation code / state 의존 / site 의존을 허용하되,
full H-space arbitrary nonlinear reconstruction은 허용하지 않는다.

### Loss

```
E_pred = Σ_valid ||ΔU_norm - ΔU_hat_norm||²
E_zero = Σ_valid ||ΔU_norm||²
R      = E_pred / max(E_zero, τ)          Zero predictor는 R = 1

L = L_transport + 0.1 * L_consistency
L_transport   = original-balanced mean R   (variants/folds를 original 안에서 먼저 평균)
L_consistency = 1 - cosine(z1, z2)         (독립 support cell dropout 0.1 두 view)
```

τ(`denominator_floor`)는 server audit의 frozen 값이어야 하고 null이면 real run이 fail-fast한다.
held-out 결과를 보고 조정하지 않는다. behavior loss / Flow next loss / energy loss / InfoNCE /
negative contrastive / recipe classification은 넣지 않는다. InfoNCE를 쓰지 않는 이유는 다른
pair가 실제 negative라는 보장이 없기 때문이다 (서로 다른 문제에 같은 transformation family가
있을 수 있다).

### Normalization과 checkpoint

`MacroNormStats`는 **train originals만**으로 fit한다 (`state_scale [G+1]`,
`update_scale [G,2]`). variant delta 분포를 먼저 보고 scale을 정하지 않고 validation/test Page
통계를 쓰지 않는다. LRT checkpoint schema는 `aimo-lrt-v1`이며 old Flow/Behavior checkpoint를
LRT로 load하지 못한다.

### Baselines와 control

| 이름 | 정의 |
| --- | --- |
| `zero` | `ΔU_hat = 0`, transport ratio 1 |
| `train_mean` | train originals의 macro × stream relation mean |
| `rank4_linear_transport` | train-only rank-4 native basis(SVD) + support 투영 coefficient. **historical U4 재현이 아님** |
| E-FLOW-1 raw Flow | 기존 legacy result/interface 보존 |
| `lrt_v1` | 위 architecture |

support-swap: 평가 pair `i`의 query를 다른 original `j`의 `z_j`로 예측한다. donor는 same pair
금지, 가능하면 same original 금지, outcome/label 기반 선택 금지, eval seed로 deterministic.
`swap_gap = E_swap - E_correct > 0`은 relation representation을 실제로 쓴다는 evidence이며
causal evidence가 아니다.

### Parameter (실측)

`d_model 128 / adapter 32 / relation 16 / decoder_rank 16` 기준.

| component | toy `H=8, P=4` | 예시 `H=2560, P=17` |
| --- | --- | --- |
| update_adapter | 288 | 81,952 |
| state_adapter | 288 | 81,952 |
| relation_embeddings | 23,040 | 24,704 |
| shared_block | 132,480 | 132,480 |
| relation_projection | 2,320 | 2,320 |
| coefficient_network | 14,416 | 14,416 |
| mixer_basis / ffn_basis | 128 / 128 | 40,960 / 40,960 |
| **total** | **173,088** | **419,744** |
| (Flow loop4 total 비교) | 159,752 | 1,470,592 |

Macro8이 parameter를 자동으로 4배 줄이는 것은 **아니다**. Macro는 depth 길이와 relation
sequence 크기, attention compute를 줄인다. real scale에서 total이 줄어드는 것은 low-dimensional
native adapter(H→32)와 rank-16 native basis가 IO parameter를 줄이기 때문이다
(Flow io 1,007,616 → LRT adapter+basis 245,824).

---

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
- 각 항은 **합 + count**로 모으고, effective batch 전체의 항별 global denominator로 한 번만
  나눕니다. 따라서 microbatch 크기가 effective objective와 gradient를 바꾸지 않습니다
  (서로 다른 항의 valid count를 합치지 않습니다). validation도 같은 방식입니다.
- Huber는 training과 evaluation이 같은 helper와 `delta = 0.1`을 씁니다.
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

학습 시작 전에 supervision을 확인합니다. 활성화한 behavior objective에 유효 label이 전혀 없거나
선택한 validation 지표에 유효 label이 없으면 **fail-fast**합니다. 가중치가 0인 항은 objective에
기여하지 않으므로 학습된 head로 표시하지 않습니다.

`run.device`는 실제 model / batch / NormStats에 적용됩니다. 전체 dataset을 accelerator에
올리지 않고 microbatch만 옮기며, 요청한 accelerator가 없으면 CPU로 조용히 내려가지 않고
오류입니다 (실제 CUDA 검증은 SERVER_PENDING).

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
| checkpoint schema | `aimo-checkpoint-v3` |
| page store schema | `aimo-page-store-v2` |
| scorer version | `2` |

`aimo-checkpoint-v3`는 역할별 seed(`data`/`split`/`train`/`sampler`/`eval`), 내용까지 반영한
data hash, Python/NumPy RNG를 포함합니다. version이 없거나 `aimo-checkpoint-v2`인 checkpoint는
명시적으로 거부합니다. CUDA RNG는 저장하지만 재현성은 SERVER_PENDING입니다.

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
