# CLAUDE.md

작업 지침은 [AGENTS.md](AGENTS.md)에 있습니다. 먼저 그 파일을 읽으세요.

상세 문서:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - shape, indexing, input/output, mask,
  loop, rollout, loss
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) - dataset, split, screening, baselines,
  metrics, 실험 순서
- [docs/SERVER_HANDOFF.md](docs/SERVER_HANDOFF.md) - 환경, 실행 명령, 산출물,
  SERVER_PENDING, 재개 방법

짧게 기억할 것:

- 확정 architecture를 구현합니다. 새 architecture 탐색으로 범위를 바꾸지 않습니다.
- variant future를 model input에 넣지 않습니다. reference는 variant prefix를 읽지 않습니다.
- 설명은 한국어, 전문용어는 English로 씁니다.
- 로컬은 CPU 전용입니다. GPU 실행에는 `--execute-gpu`가 필요합니다.
- 검증하지 않은 것을 검증 완료라고 쓰지 않습니다.

```bash
.venv/bin/python -m pytest tests/
.venv/bin/ruff check src tests
.venv/bin/aimo check --config configs/toy.yaml
```
