# CLAUDE.md

작업 지침은 [AGENTS.md](AGENTS.md)에 있습니다. 먼저 그 파일을 읽으세요.

상세 문서:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Behavior/Flow view, shape, mask, loop,
  rollout, loss, schema version
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) - dataset, split, baselines, metrics, 실험 순서
- [docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md) - label, counts/bounds, provenance,
  실행 protocol
- [docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) - 환경, 실행 명령, 산출물,
  SERVER_PENDING, 재개 방법
- [docs/FIXES.md](docs/FIXES.md) - 검수에서 고친 문제, 실제 CPU 검증 범위, 서버 남은 항목

짧게 기억할 것:

- 현재 primary는 **behavior supervision + flow auxiliary joint** 학습입니다
  (`L = L_behavior + 0.1 * L_flow`). stable-only flow-only는 legacy baseline입니다.
- Behavior와 Flow는 같은 `LoopedCore` 객체를 공유합니다. Transformer를 두 개 만들지 않습니다.
- Flow model input에 variant future를 넣지 않습니다. reference는 variant 관측을 읽지 않습니다.
- `BehaviorInput`에 label·ID·counts·topic·difficulty를 넣지 않습니다.
- missing label을 0으로 바꾸지 않습니다. label 0은 실제 label입니다.
- loss 항은 합 + count로 모으고 effective batch의 global denominator로 한 번만 나눕니다
  (microbatch 크기가 objective를 바꾸면 안 됩니다).
- 학습되지 않은 head의 원시 score는 canonical 출력에 넣지 않습니다.
- `run.device`는 실제로 적용됩니다. CUDA가 없으면 CPU로 조용히 내려가지 않고 오류입니다.
- 설명은 한국어, 전문용어는 English로 씁니다.
- 로컬은 CPU 전용입니다. GPU 실행에는 `--execute-gpu`가 필요합니다.
- 검증하지 않은 것을 검증 완료라고 쓰지 않습니다.

```bash
.venv/bin/python -m pytest tests/
.venv/bin/ruff check src tests
.venv/bin/aimo check --config configs/toy_behavior.yaml
```
