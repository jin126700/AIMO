# AIMO 실험 기록

<!-- run:deepmath_real_20260926T193418Z:start -->
## deepmath_real_20260926T193418Z

- 목적: DeepMath pair-drop prediction prototype.
- 상태: DATA/PROTOCOL_LIMIT
- 원격 SHA: 80f257ded148b3e199769aec6b6e4654d88915c6; local patch hash는 protocol/patch.sha256 참조.
- 데이터/model/policy: protocol/manifest.json 및 calibration/config.json 참조.
- 결과 경로: /data1/HKM/result/AIMO/deepmath_real_20260926T193418Z

```json
{
  "label_valid_originals": 0,
  "label_valid_pairs": 0,
  "main_cohort_started": false,
  "calibration_counts": {
    "C": 4,
    "U_score": 5
  },
  "calibration_started_slots": 9,
  "calibration_planned_slots": 36,
  "main_planned_slots": {
    "small": 624,
    "standard": 960
  },
  "gpu_active_minutes": 21.684410693015284,
  "models_evaluated": [],
  "joint_vs_behavior_only": null,
  "joint_vs_m0": null,
  "pair_specific_signal": null,
  "robust_head": "null/untrained",
  "training_seed_variance": null,
  "judgment": "DATA/PROTOCOL_LIMIT",
  "runtime_estimate_minutes": {
    "528": {
      "one_worker": 4534.777195415646,
      "ideal_four_workers": 1133.6942988539115
    },
    "624": {
      "one_worker": 5359.282140036672,
      "ideal_four_workers": 1339.820535009168
    },
    "960": {
      "one_worker": 8245.049446210265,
      "ideal_four_workers": 2061.2623615525663
    }
  },
  "available_behavior_minutes": 98.31558930698472,
  "calibration_summary": "/data1/HKM/result/AIMO/deepmath_real_20260926T193418Z/calibration/summary.json",
  "blocker": "Even ideal four-worker extrapolation for the 528-slot operational minimum exceeds the remaining behavior budget; no main cohort launched.",
  "calibration_label_summary": {
    "calibration_originals": 6,
    "calibration_pairs": 12,
    "calibration_complete_label_originals": 0,
    "calibration_label_valid_pairs": 0,
    "resolved_slots": 4,
    "planned_slots": 36,
    "unresolved_ratio": 0.8888888888888888,
    "by_topic": {
      "Algebra": {
        "planned_slots": 12,
        "resolved_slots": 3
      },
      "Discrete Mathematics": {
        "planned_slots": 12,
        "resolved_slots": 1
      },
      "Geometry": {
        "planned_slots": 6,
        "resolved_slots": 0
      },
      "Number Theory": {
        "planned_slots": 6,
        "resolved_slots": 0
      }
    },
    "main_label_valid_originals": 0,
    "main_label_valid_pairs": 0,
    "note": "Censoring bounds and finite-sample Wilson intervals are different; calibration is excluded from all model training/evaluation."
  },
  "comparison_table": {
    "constant": {
      "MAE": null,
      "Huber": null,
      "status": "not_trained_insufficient_budget_for_labels"
    },
    "raw_change": {
      "MAE": null,
      "Huber": null,
      "status": "not_trained_insufficient_budget_for_labels"
    },
    "behavior_m0": {
      "MAE": null,
      "Huber": null,
      "status": "not_trained_insufficient_budget_for_labels"
    },
    "behavior": {
      "MAE": null,
      "Huber": null,
      "status": "not_trained_insufficient_budget_for_labels"
    },
    "joint": {
      "MAE": null,
      "Huber": null,
      "status": "not_trained_insufficient_budget_for_labels"
    }
  },
  "cost_gate": {
    "decision": "STOP_DATA_PROTOCOL_LIMIT",
    "decision_basis": "runtime_only_no_outcome_selection",
    "active_seconds_at_decision": 1297.061238833703,
    "parallel_phase_seconds": 1040.8041851329617,
    "workers": 4,
    "startup_and_diagnostic_allowance_seconds_per_worker": 60,
    "optimistic_observed_service_seconds": 3923.216740531847,
    "denominator_all_36_planned_calibration_slots": 36,
    "assumed_remaining_calibration_cost_seconds": 0,
    "optimistic_seconds_per_planned_slot": 108.9782427925513,
    "minimum_label_cohort_slots": 528,
    "ideal_four_worker_minimum_minutes": 239.75213414361284,
    "remaining_behavior_minutes": 98.38231268610495,
    "limitation": "Runtime extrapolation from frozen calibration, not a guarantee for unseen runtimes. Independent-request microbatch 1, FP32 SDPA; unvalidated alternative batching/dtype is not assumed faster.",
    "calibration_status": "partial_due_cost_gate",
    "main_cohort_started": false
  },
  "calibration_page_report": {
    "extracted": 17,
    "skipped": 1,
    "stopped": false,
    "stop_reason": "",
    "errors": [],
    "unique_prompt_pages": 18,
    "calibration_only": true,
    "max_residual_error": 6.103515625e-05,
    "peak_vram_bytes": 16236520448
  },
  "gpu_hours": 1.2298529681864765,
  "main_unstarted_ratio": 1.0,
  "coverage_limitation": "No main labels were generated; calibration-only performance cannot establish pair-specific behavior signal.",
  "code_patch_sha256": "b0f7b51e2d77bd9ac261de173ecbb8bff9d412cfa726cca456f81a5b083e0231",
  "data_manifest_sha256": "219bba3cc8fb5311ed43f24fc3da19bc515f79f3516434a8a432a1dcf7364290",
  "policy_hash": "9bf7e22cece2e173",
  "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
  "calibration_actual": {
    "originals_with_started_slots": 2,
    "pairs_with_started_variant_slots": 3,
    "prompts_with_started_slots": 5,
    "started_slots": 9,
    "originals_with_pages": 6,
    "unique_prompt_pages": 18
  },
  "calibration_planned": {
    "originals": 6,
    "pairs": 12,
    "prompts": 18,
    "slots": 36
  },
  "main_actual": {
    "originals": 0,
    "pairs": 0,
    "prompts": 0,
    "slots": 0
  },
  "termination_breakdown": {
    "eos_scored_correct": 4,
    "wrong": 0,
    "token_cap": 0,
    "scorer_unsupported": 0,
    "infra_error": 0,
    "explicit_stop_diagnostic": 1,
    "cost_gate_interrupted": 4,
    "not_started": 27
  },
  "verification": {
    "full_cpu_suite": "228 passed in 28.39s",
    "later_runtime_tests": "9 passed",
    "real_gpu_seed_repeat": true,
    "real_gpu_inflight_stop": true,
    "real_page_cuda_forward_backward": true,
    "page_checksums_verified": 18,
    "ruff": "not installed; no broad environment upgrade"
  },
  "cleanup": {
    "removed_containers": [
      "hkm-aimo-deepmath"
    ],
    "removed_container_logical_writable_bytes": 16417,
    "actual_disk_reclaimed_bytes": null,
    "deleted_dataset_model_result_bytes": 0,
    "large_data_moved_or_duplicated_bytes": 0,
    "reused_image": "hkm-aimo:rft-paths-20260924",
    "preserved_caches": [
      "/data1/HKM/.cache",
      "/data1/HKM/.nv",
      "/data1/Data/AIMO/Models/Qwen/Qwen3-4B"
    ],
    "preserved_raw_slots": 9,
    "preserved_prompt_pages": 18
  },
  "resume_command": "docker exec -e PYTHONPATH=/data1/HKM/AIMO/src -w /data1/HKM/AIMO hkm-rft-cpu python -m aimo deepmath-real --stage audit-resume --run-dir /data1/HKM/result/AIMO/deepmath_real_20260926T193418Z",
  "limitations": [
    "Calibration is partial; generated outcomes cover only 2 of the 6 frozen originals.",
    "Cost gate uses runtime extrapolation with uncertainty, not a formal bound on unseen problem runtimes.",
    "No main labels, supervised predictors, held-out comparisons, causal claims or official robustness certification.",
    "STOP is retained. Completed/interrupted slots are never replaced; 27 not_started slots remain."
  ]
}
```
<!-- run:deepmath_real_20260926T193418Z:end -->

<!-- run:deepmath_bf16_20260926T202430Z:start -->
## deepmath_bf16_20260926T202430Z

- 판단: **DATA/PROTOCOL_LIMIT**
- 목적: FP32 reference Page로 BF16 behavior의 signed pair-drop을 예측하는 observational prototype.
- 직전 FP32 결과 해석: 행동 측정 비용에 의한 중단; architecture와 robustness 가설 미검증. 이전 C4와 새 BF16 labels는 합치지 않았다.
- 코드: 80f257ded148b3e199769aec6b6e4654d88915c6 + 미push local patch. 정확한 hash/dirty 상태: protocol/manifest.json 및 local.patch.
- Behavior policy: `d023938fbe646945837f281f8f21f595dc301baf5d395664d6ec0c58cd961c2f`; FP32 observation policy: `9bf7e22cece2e173`.
- 공통 checkpoint: `1cfa9a7208912126459214e8b04321603b3df60c`. 데이터·model 파일 hash는 manifest 참조.

| 구분 | planned slots | C | W | X | 운영 중단 | 미시작 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 calibration | 36 | 19 | 3 | 0 | 8 | 6 |
| BF16 main | 624 | 0 | 0 | 0 | 0 | 624 |

- Main planned: 52 originals / 104 pairs / 156 prompts / 624 slots. 실제 시작 slots: 0.
- Label-valid originals: {}; pairs: {'known_original_test': 0, 'train': 0, 'validation': 0}; 부족 originals: {'train': 24, 'validation': 8, 'known_original_test': 12}.
- Planned slots 미확정 비율: 100.00%. 이는 모델 실패율이 아니며 X/채점불가/운영중단/미시작을 별도 원시 집계로 보존했다.
- 256-token 성능 전용 시험: FP32 serial 35.20 tokens/s; BF16 serial 1.21×; BF16 batch12 합산 14.03×. 자연 길이·정답률 표본으로 사용하지 않았다.
- 실제 main stage-inclusive 처리량: 0.0 tokens/s; 자연 생성 길이: {}. Cold start·batch latency·VRAM은 load*.json / *.done.json에 분리했다.

| 비교군 seed0 | pair-drop MAE | Huber δ=.1 | sibling swap MAE 증가 |
|---|---:|---:|---:|
| constant | — | — | — |
| raw_change | — | — | — |
| behavior_m0 | — | — | — |
| behavior | — | — | — |
| joint | — | — | — |

- 학습·평가 상태: 완료되지 않음; 위 최소 label 수 부족 또는 stage 로그의 실행 중단 사유 참조.
- Joint paired 비교: 미실행: 관측 label 부족 또는 학습 미완료.
- Raw-change 입력은 정규화한 final-state 차이 크기, 누적 update 차이 크기, 유효 landmark 비율뿐이다. 8 parameters 중 pair-drop 4개, untrained robust 4개.
- Binary robustness label/Accuracy/AUROC는 없음; robust head null/untrained. Complete-panel max-drop은 diagnostic만 사용.
- Bootstrap 2000회는 original 단위 paired resampling이며 고정 관측 labels에 조건부인 비교다. 4회 sampling의 실제 correctness probability 불확실성이나 training-seed variance를 의미하지 않는다.
- 미확정이 많은 어려운 panel이 평가에서 제외되어 성능이 낙관적일 수 있다. 작은 known-test의 탐색적 결과이며 causal mechanism·공식 robustness 인증이 아니다.
- Page: {}. Calibration은 학습/validation/test에 포함하지 않았다.
- 누적 GPU-active: 59.03분 / 180분; GPU-hours: 3.189. 잔여 120.97분. 새 행동 측정은 누적120분 이후 금지.
- 산출물: `/data1/HKM/result/AIMO/deepmath_bf16_20260926T202430Z`. Raw slots / labels bounds / Page checksum / frozen manifest 보존.
- 재현 이력: `/data1/HKM/result/AIMO/deepmath_bf16_20260926T202430Z/commands.sh`.
- 재개는 동일 policy의 미시작 slot만 가능하다. started-crash, X, W, U_score, infra_error는 새 sample로 교체하지 않는다. 예산 증액은 별도 사용자 승인 없이는 하지 않는다.

```sh
docker exec -d hkm-aimo-bf16 python -m aimo.bf16_pipeline /data1/HKM/result/AIMO/deepmath_bf16_20260926T202430Z
```

- 위 재개 명령은 현재 supervisor가 없고 기존 container가 남아 있을 때 사용한다. 종료 예산을 우회하지 않는다. Backend 성능 시험과 prepare 명령을 재실행하지 않는다.
<!-- run:deepmath_bf16_20260926T202430Z:end -->

<!-- run:deepmath_bf16_bucket_20260926T210413Z:start -->
## deepmath_bf16_bucket_20260926T210413Z

- 판단: **DATA/PROTOCOL_LIMIT**
- 목적: FP32 reference Page로 BF16 behavior의 signed pair-drop을 예측하는 observational prototype.
- 직전 FP32 결과 해석: 행동 측정 비용에 의한 중단; architecture와 robustness 가설 미검증. 이전 C4와 새 BF16 labels는 합치지 않았다.
- 코드: 80f257ded148b3e199769aec6b6e4654d88915c6 + 미push local patch. 정확한 hash/dirty 상태: protocol/manifest.json 및 local.patch.
- Behavior policy: `a38f5bb28eba06de6c51f217387a234158f4df40b5856959614e4fa6f80f0f7d`; FP32 observation policy: `9bf7e22cece2e173`.
- 공통 checkpoint: `1cfa9a7208912126459214e8b04321603b3df60c`. 데이터·model 파일 hash는 manifest 참조.

| 구분 | planned slots | C | W | X | 운영 중단 | 미시작 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 calibration | 36 | 25 | 9 | 2 | 0 | 0 |
| BF16 main | 624 | 110 | 2 | 8 | 0 | 504 |

- Main planned: 52 originals / 104 pairs / 156 prompts / 624 slots. 실제 측정 시작: 12 originals / 18 pairs / 30 prompts / 120 slots.
- Label-valid originals: {'known_original_test': 2, 'validation': 2, 'train': 4}; pairs: {'known_original_test': 4, 'train': 6, 'validation': 3}; 부족 originals: {'train': 20, 'validation': 6, 'known_original_test': 10}.
- Planned slots 미확정 비율: 82.05%. 이는 모델 실패율이 아니며 X/채점불가/운영중단/미시작을 별도 원시 집계로 보존했다.
- 256-token 성능 전용 시험: FP32 serial 35.20 tokens/s; BF16 serial 1.21×; BF16 batch12 합산 14.03×. 자연 길이·정답률 표본으로 사용하지 않았다.
- 실제 main 첫 시작~마지막 종료 관측구간 합산 처리량: 291.8469823628179 tokens/s; 자연 생성 길이: {'p50': 7866.5, 'p90': 14700.300000000003, 'max': 16384.0}. Cold start·batch latency·VRAM은 load*.json / *.done.json에 분리했다.

| 비교군 seed0 | pair-drop MAE | Huber δ=.1 | sibling swap MAE 증가 |
|---|---:|---:|---:|
| constant | — | — | — |
| raw_change | — | — | — |
| behavior_m0 | — | — | — |
| behavior | — | — | — |
| joint | — | — | — |

- 학습·평가 상태: 완료되지 않음; 위 최소 label 수 부족 또는 stage 로그의 실행 중단 사유 참조.
- Joint paired 비교: 미실행: 관측 label 부족 또는 학습 미완료.
- Raw-change 입력은 정규화한 final-state 차이 크기, 누적 update 차이 크기, 유효 landmark 비율뿐이다. 8 parameters 중 pair-drop 4개, untrained robust 4개.
- Binary robustness label/Accuracy/AUROC는 없음; robust head null/untrained. Complete-panel max-drop은 diagnostic만 사용.
- Bootstrap 2000회는 original 단위 paired resampling이며 고정 관측 labels에 조건부인 비교다. 4회 sampling의 실제 correctness probability 불확실성이나 training-seed variance를 의미하지 않는다.
- 미확정이 많은 어려운 panel이 평가에서 제외되어 성능이 낙관적일 수 있다. 작은 known-test의 탐색적 결과이며 causal mechanism·공식 robustness 인증이 아니다.
- Page: {'reused_calibration': 18, 'new_main_pages': 156, 'indexed_pages': 174, 'wall_seconds': 187.08419977873564, 'peak_vram_bytes': 16242251264, 'forward_dtype': 'float32', 'storage_dtype': 'float32', 'model_config_hash': '30b494d677d110e5'}. Calibration은 학습/validation/test에 포함하지 않았다.
- 누적 GPU-active: 129.99분 / 180분; GPU-hours: 7.563. 잔여 50.01분. 새 행동 측정은 누적120분 이후 금지.
- 산출물: `/data1/HKM/result/AIMO/deepmath_bf16_bucket_20260926T210413Z`. Raw slots / labels bounds / Page checksum / frozen manifest 보존.
- 재현 이력: `/data1/HKM/result/AIMO/deepmath_bf16_bucket_20260926T210413Z/commands.sh`.
- 재개는 동일 policy의 미시작 slot만 가능하다. started-crash, X, W, U_score, infra_error는 새 sample로 교체하지 않는다. 예산 증액은 별도 사용자 승인 없이는 하지 않는다.

```sh
sh /data1/HKM/result/AIMO/deepmath_bf16_bucket_20260926T210413Z/runtime/resume.sh
```

- 위 재개 명령은 현재 supervisor가 없고 기존 container가 남아 있을 때 사용한다. 종료 예산을 우회하지 않는다. Backend 성능 시험과 prepare 명령을 재실행하지 않는다.

- 선택 경로: 같은 prompt의 독립4slots를 묶는 BF16 batch4. 8K-cache 성능 전용 decode 시험에서 padding 경로 대비 2.045×; 이 수치는 전체 FP32/BF16 실행의 직접 비교나 정답률 동등성 검증이 아니다.
- 실제 tensor peak / allocator reserved peak: 16.83 / 39.96 GiB. 두 수치는 다르다.
- 관측 pair-drop 분포: {'0.0': 11, '0.25': 2}. 이것만으로 pair-specific predictor 신호를 주장하지 않는다.
- 완료 prompt-batch 30개 기반 전체624slots 비용 외삽: 평균 기준 230.1분, P90 기준 425.7분. CI가 아니며 미완료/미시작 문제는 더 오래 걸릴 수 있다.
- Page config hash 비교 오류는 수정·재검증 완료했다. 초기 오류와 해결 근거는 runtime/post_error.json 및 runtime/post_error_resolution.json에 보존했다.
- 체크섬이 유효한 main Page156개는 CUDA를 숨긴 재개 검사에서 모델 로드 없이 재사용했다. 마지막 생성 batch 시작은 누적119.765분으로120분 제한을 지켰다.
- 총180분 중 남은 시간이 있어도 새 행동 측정120분 cutoff는 소진됐다. 추가 label 생성은 시간 정책의 별도 승인 없이 시작하지 않는다.
<!-- run:deepmath_bf16_bucket_20260926T210413Z:end -->

<!-- run:deepmath_fp32_primary_20260926T224715Z:start -->
## deepmath_fp32_primary_20260926T224715Z

- 상태: EVALUATION_COMPLETED
- 본실험 계획: 52 originals / 104 pairs / 156 prompts / 624 slots.
- 실제 counts: {'C': 541, 'W': 13, 'U_score': 10, 'X': 60}; label-valid originals: {'known_original_test': 10, 'train': 27, 'validation': 6}; pairs: {'known_original_test': 19, 'train': 50, 'validation': 12}.
- unresolved 비율: 0.112; drop 분포: {'0.25': 2, '0.0': 77, '-0.25': 1, '-0.5': 1}.
- Qwen 행동 / Page forward·저장 / predictor forward·backward: FP32. AMP/autocast/TF32 OFF.
- 120/150/175/180분 제한 해제. 5시간은 상한이 아님. cap 16384와 sampling 조건은 유지.
- BF16·calibration label 혼합 없음. 기존 BF16 결과를 보았으므로 untouched dataset 검증이 아님.
- Page: {'new_main_pages': 156, 'reused_pages': 0, 'model_config_hash': '30b494d677d110e5', 'wall_seconds': 181.47146354196593, 'peak_vram_bytes': 16242251264, 'dtype': 'float32'}.

### OOM과 같은 cohort 재개
- GPU2 batch4가 generation14825 tokens에서 allocator90% 한도 내 추가934MiB 할당 실패. allocated33.31GiB, reserved-unused6.08GiB. DynamicCache 성장과 fragmentation/reservation 영향이며 추가 model replica 또는 과대 static cache 증거는 없음.
- 변경: 4 replicas 유지, effective batch1, expandable_segments, allocator80%, sequence 종료 즉시 전체 cache 반환. FP32/policy/seed/cap/cohort 유지. batch shape에 따른 부동소수 연산의 bitwise 동일성은 주장하지 않음.
- smoke peak allocated 18.617GiB / reserved 18.834GiB; single-replica 18.991tokens/s. 이전 OOM batch는4×14825 tokens/3539sec이나 종료된 sequence를 포함해 유효 throughput과 직접 동등 비교할 수 없음.
- 최종 상태: {'counts': {'C': 541, 'W': 13, 'X': 60, 'U_score': 0, 'interrupted': 10, 'not_started': 0}, 'resolved_prompts': 127, 'planned_prompts': 156, 'valid_pairs': 81, 'drop_zero': 77, 'drop_positive': 2, 'drop_negative': 2, 'original_correctness_distribution': {'1.0': 41, '0.75': 1, '0.25': 1}, 'interrupted_replaced': False, 'binary_robust_label': None}. 완료26개와 interrupted10개 원시 bytes 보존 검증 통과.
- 재개 코드 SHA/dirty patch/source hashes: {'git_sha': '80f257ded148b3e199769aec6b6e4654d88915c6', 'dirty_patch_sha256': '3d8d2d7eec2c2bb68151b1219fa594431d89989dd1e366cf3ef20f20e2ca1c24', 'recovery_changes_patch_sha256': '59ca0280335f3f4a1f57f299334f0ae4ec8ce7f869bb0ee254b464547df644d6', 'source_hashes': {'src/aimo/bf16_batch.py': '31292703870a954b01d0f3fad4f119c9c1979dc2eba8c09029c0a22f77beb56b', 'src/aimo/fp32_collect.py': '82fa3784abeb6271d310aa5f4d58267c6aee9279a8b1ab1a319f6931cfbdda74', 'src/aimo/fp32_runtime.py': '63b16ecb9bc2c193db1fd28382c0f6f5d96d8818cfb723c966eeb64067080e71', 'src/aimo/fp32_pipeline.py': '29ad6a42c0c0bdfdf91a43dc980d97600c1425d274029d9be71c503bdbd950d2', 'src/aimo/fp32_recovery.py': '4fa7663ca5446ab3167c4787598dbfc0578e13da228cb1217c35b380727f703c', 'src/aimo/fp32_post.py': '7ad16a4b0d7c0ea91d5dfd4bd7aef62f45deda341eb05b5fc16dc8be784219da', 'src/aimo/fp32_report.py': '2914182f4045e566a4ebc550cd471598bd9caf3b89e9dbcbe9caacca0e1008ed', 'tests/test_fp32_primary.py': '3824353b3abe1f0cc7ad695d1df9463aa806dc25a695ea8bd858512395b72ad8'}, 'memory_settings_sha256': '6ceeeb61a35fc61e05e0f7b0da85fa1fadf4586ceaefe5cff2f2b7774c3779ba', 'baseline': 'source_before plus preserved original executed_source.tar.gz', 'updated_at': 1790489093.7871554}.

| 모델 | Original-balanced MAE | 95% original bootstrap CI | n |
|---|---:|---|---:|
| zero | 0.0375 | [0.0, 0.0875] | 10 |
| constant_fit | 0.04166666666666667 | [0.0041666666666666675, 0.09166666666666666] | 10 |
| constant_seed0 | 0.038331475714221595 | [0.0008314757142215967, 0.08833147571422159] | 10 |
| raw_change_seed0 | 0.9625 | [0.9125, 1.0] | 10 |
| behavior_m0_seed0 | 0.045330767938867214 | [0.009915018239989878, 0.09173345297807826] | 10 |
| behavior_seed0 | 0.04549461946589872 | [0.010070277705672199, 0.091937326999614] | 10 |
| behavior_seed1 | 0.05074001636821777 | [0.017655338447075338, 0.09550161903374828] | 10 |
| behavior_seed2 | 0.06237514173553791 | [0.019361211179057137, 0.11325063047930597] | 10 |
| joint_seed0 | 0.05562471779994667 | [0.013755229766538831, 0.10887504689075285] | 10 |
| joint_seed1 | 0.04714858876832295 | [0.008634420021844563, 0.09850367736478798] | 10 |
| joint_seed2 | 0.06499461580679053 | [0.012728167704190128, 0.12479912247192257] | 10 |

Paired improvement = baseline MAE − joint MAE; 양수일 때 joint 우세.

| joint seed | 비교군 | 개선량 | paired 95% CI |
|---:|---|---:|---|
| 0 | zero | -0.018124717799946666 | [-0.028891777838871348, -0.009112400877347682] |
| 0 | constant_fit | -0.01395805113328 | [-0.024725111172204686, -0.004945734210681021] |
| 0 | constant_seed0 | -0.01729324208572507 | [-0.02806030212464975, -0.008280925163126085] |
| 0 | raw_change_seed0 | 0.9068752822000533 | [0.8017083656566684, 0.9852616635349114] |
| 0 | behavior_m0_seed0 | -0.010293949861079454 | [-0.020717486173671203, -0.0003708207246381959] |
| 0 | behavior_seed0 | -0.010130098334047943 | [-0.020167916573409456, -0.000750996629358271] |
| 1 | zero | -0.009648588768322952 | [-0.01681986664123542, -0.004095376719851629] |
| 1 | constant_fit | -0.0054819221016562895 | [-0.012653199974568758, 7.128994681503519e-05] |
| 1 | constant_seed0 | -0.008817113054101355 | [-0.015988390927013825, -0.0032639010056300325] |
| 1 | raw_change_seed0 | 0.9153514112316771 | [0.8130203396914294, 0.991082086936658] |
| 1 | behavior_m0_seed0 | -0.0018178208294557408 | [-0.008855969431824633, 0.005805577959399669] |
| 1 | behavior_seed1 | 0.0035914275998948143 | [-0.005989326616399922, 0.01331218208404607] |
| 2 | zero | -0.027494615806790534 | [-0.06807086742719548, -0.0041128020340693225] |
| 2 | constant_fit | -0.023327949140123865 | [-0.06390420076052881, 5.386463259734331e-05] |
| 2 | constant_seed0 | -0.026663140092568937 | [-0.06723939171297388, -0.0032813263198477258] |
| 2 | raw_change_seed0 | 0.8975053841932095 | [0.7867495380702894, 0.9860347925371025] |
| 2 | behavior_m0_seed0 | -0.01966384786792332 | [-0.05930371695048962, 0.004691598142344445] |
| 2 | behavior_seed2 | -0.0026194740712526254 | [-0.01754246326065186, 0.007208325280917051] |

- 최종 판정: NO_ESTABLISHED_PAIR_SIGNAL.
- small exploratory result: 권장 최소 original 수에 미달하는 split이 있음. 실제 유효 감독으로 학습하고 한계를 표시했으며 split 재배정은 하지 않음.

- Pair-specific signal: 추가 가치 확인되지 않음. 모든 유효 target의 sibling support-swap MAE 증가: [{'mean': -1.2721121311187745e-05, 'n_originals': 10, 'n_undefined': 2, 'ci95_low': -3.816336393356323e-05, 'ci95_high': 0.0, 'bootstrap': 'original-group resampling (not training-seed variance)'}, {'mean': -4.475098103284836e-05, 'n_originals': 10, 'n_undefined': 2, 'ci95_low': -0.00013425294309854507, 'ci95_high': 0.0, 'bootstrap': 'original-group resampling (not training-seed variance)'}, {'mean': 2.0011540618725122e-05, 'n_originals': 10, 'n_undefined': 2, 'ci95_low': -7.768227369524539e-05, 'ci95_high': 0.00013771689555142075, 'bootstrap': 'original-group resampling (not training-seed variance)'}].
- Flow auxiliary: 추가 가치 확인되지 않음. seed별 joint vs behavior-only 결과를 모두 보고함.
- Training seed variance: {'behavior': {'seed_mae': [0.04549461946589872, 0.05074001636821777, 0.06237514173553791], 'mean': 0.05286992585655146, 'sample_sd': 0.008639467219992016}, 'joint': {'seed_mae': [0.05562471779994667, 0.04714858876832295, 0.06499461580679053], 'mean': 0.05592264079168672, 'sample_sd': 0.008926742902435605}}.
- Flow next/within/2–4step rollout: evaluation/joint_flow.json. test Page는 normalization·pretraining에 미사용.

- 이번 FP32 active wall: 33.168 h; GPU-hours: 112.040.
- 이전 누적 비용: active wall 2.166 h; GPU-hours 7.563; 이번 실행과 별도로 보존.
- 전체 GPU-hours(기존+이번): 119.603. 별도 technical audit 시간은 protocol/technical_cost.json 참조.
- 실행 코드: {'sha': '80f257ded148b3e199769aec6b6e4654d88915c6', 'dirty_patch_sha256': '3d8d2d7eec2c2bb68151b1219fa594431d89989dd1e366cf3ef20f20e2ca1c24', 'source_hashes_sha256': 'f37c0db6fbff97fab5244f50ef26eb9735c28f6d28d8833cc6760e3efc4d846e'}. 상세 data/policy/source hashes: protocol/manifest.json 및 executed_source_hashes.json.
- 네 번의 sampling target은 population correctness의 정확한 값이 아님. original bootstrap은 관측 label에 조건부이고 training seed variance와 별개임.
- Binary robust head null/untrained. 공식 robustness 또는 수학적 causal mechanism 검증으로 해석하지 않음.
- 재개: sh /data1/HKM/result/AIMO/deepmath_fp32_primary_20260926T224715Z/runtime/resume.sh
- 결과: /data1/HKM/result/AIMO/deepmath_fp32_primary_20260926T224715Z
<!-- run:deepmath_fp32_primary_20260926T224715Z:end -->

## flow_rep_v1_20260928T172305Z

# E-FLOW-1

직전 behavior regression에서 81 valid pairs 중 77 pair가 zero-drop이었고, zero baseline이 Joint보다 우수하여, pair-drop direct supervision을 primary에서 제거하고 Flow representation learning + frozen robustness probe로 전환.

Flow 판정: FLOW_REPRESENTATION_NOT_ESTABLISHED
Probe 판정: ROBUSTNESS_PROBE_DATA_LIMIT
81 pairs는 전체 split 합계이며 이전 test는 19 pairs / 10 originals.

## Architecture 및 protocol
Shared LoopedCore ×4, 128/4 heads/256 FFN/dropout0.1. L_next + L_within + 0.25 L_roll.
Stage1 Page-only loader, flow_train 28 / flow_dev 4, ID hash split. 외부 validation/test는 학습 및 checkpoint 선택에 미사용.
behavior_supervision_seen = false. FP32, AMP/autocast/TF32 OFF. 새 Qwen inference 없음.
z: full observed valid cells mask-mean128; panel mean/std256, std ddof0. 새로운 head/query 없음.

## Flow dev 결과 (original-balanced)
| model | next | within | rollout2/4 | total |
|---|---:|---:|---:|---:|
| zero | 0.317225 | 0.393012 | 0.127518 | 0.742116 |
| train_mean | 0.326497 | 0.393012 | 0.133104 | 0.752784 |
| flow | 0.318931 | 0.393012 | 0.12994 | 0.744428 |

학습 epochs: 64; best epoch: 48

## Robustness criterion / controls
{"train": {"robust": 0, "nonrobust": 0, "ambiguous": 0, "unresolved": 32}, "validation": {"robust": 0, "nonrobust": 0, "ambiguous": 0, "unresolved": 8}, "known_test": {"robust": 0, "nonrobust": 0, "ambiguous": 0, "unresolved": 12}}
실데이터 robust_policy가 disabled이고 검증된 binary threshold가 없어 primary robust label을 만들지 않았다. toy threshold를 이전하지 않았다.
Flow/M0/RawChange/Random feature 추출은 완료. BCE, 20회 label-shuffle null, classification bootstrap은 label 부재로 미실행이며 null 상태를 artifact에 명시.
Panel max-drop bounds와 unresolved coverage는 보존. pair label을 panel label로 복사하지 않음.

## 판단과 한계
Flow decodability ≠ robustness; probe separability ≠ causal mechanism. 기존 test를 본 이후의 exploratory 연구다.
다음: Flow gate 실패 시 Page/Flow task 검토. 통과해도 검증된 robustness criterion과 outcome-blind behavior variation 설계가 먼저이며 causal analysis로 진행하지 않음.
PCA는 train에서 fit한 diagnostic only. 클래스 부재로 centroid distance는 정의되지 않음.

Artifacts: /data1/HKM/result/AIMO/flow_rep_v1_20260928T172305Z
Reproduce: python -m aimo flow-representation-experiment --run-dir /data1/HKM/result/AIMO/flow_rep_v1_20260928T172305Z --execute-gpu
Runtime: 1487.0 seconds
Code/provenance: provenance/head.txt, before.patch, implementation.patch, source_hashes.json
