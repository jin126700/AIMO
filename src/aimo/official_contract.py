"""AIMO Interpretability Challenge 공식 제출 계약의 기록.

아래 값은 공식 자료를 직접 읽고 옮긴 것입니다. 자료가 바뀌면 이 기록을 다시 확인합니다.

- 공식 FAQ: https://aimo-interp.github.io/#faq (2026-09-30 확인)
- 공식 starter: https://github.com/aimo-interp/getting-started
  commit de794053debe75a711400696e66b60937cebbb1b
- ingestion: components/ingestion_program/ingestion.py
  sha256 47ce02f9eed36ee2f4f9ab28d07401b4b079dba3b4ed5b42d32dcd1de83cfbfd
- scoring: components/scoring_program/scoring.py
  sha256 df10d76c79129220aba37f0a23300df8d220c46af5dabec4dd20ad09fdda6191
- README.md sha256 e823924799ca1d67094c217555f65f4ff05c4b71ccbe03cf11f1392f83c4e4f5
- Dockerfile.competition sha256 d6aa3e87b0d5b4308d5479b64d7c56116b63691775b8934ccf5e17020ee5f357
"""

from __future__ import annotations

STARTER_REPO = "https://github.com/aimo-interp/getting-started"
STARTER_COMMIT = "de794053debe75a711400696e66b60937cebbb1b"
FAQ_URL = "https://aimo-interp.github.io/#faq"
SOURCE_SHA256 = {
    "components/ingestion_program/ingestion.py": (
        "47ce02f9eed36ee2f4f9ab28d07401b4b079dba3b4ed5b42d32dcd1de83cfbfd"
    ),
    "components/scoring_program/scoring.py": (
        "df10d76c79129220aba37f0a23300df8d220c46af5dabec4dd20ad09fdda6191"
    ),
    "README.md": "e823924799ca1d67094c217555f65f4ff05c4b71ccbe03cf11f1392f83c4e4f5",
    "Dockerfile.competition": "d6aa3e87b0d5b4308d5479b64d7c56116b63691775b8934ccf5e17020ee5f357",
}

# ingestion은 `reasoning_effort` keyword를 받는 are_robust를 이렇게 부릅니다:
#   solution.are_robust(model_id, reasoning_effort=effort, problems=problems)
# 결과는 `type(results) is list`, 길이 일치, 모든 원소 `type(x) is bool`이어야 valid입니다.
ENTRY_POINT = "are_robust"
REASONING_EFFORTS = ("default", "low", "medium")
DEFAULT_EFFORT = "default"

# 모든 model batch와 문제를 합친 prediction run 전체 제한 (초).
TIME_LIMIT_SECONDS = 3600.0
# 내부 목표. model load, folding, tokenizer, MP, feature, 반환까지 포함합니다.
INTERNAL_TARGET_SECONDS = 2700.0

# README의 Competition phase model 표. are_robust는 이 ID를 그대로 받습니다.
SMALL_TRACK_MODELS = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
)
MAIN_ONLY_MODELS = ("openai/gpt-oss-120b",)

# small track bundle은 root에 small.txt가 있어야 합니다.
SMALL_TRACK_MARKER = "small.txt"

# 공식 runtime (Dockerfile.competition). 로컬 .venv와 다르므로 artifact는 pickle을 쓰지 않습니다.
RUNTIME_VERSIONS = {
    "torch": "2.12.1",
    "transformers": "5.13.0",
    "accelerate": "1.13.0",
    "safetensors": "0.8.0",
    "scikit-learn": "1.8.0",
    "tokenizers": "0.22.2",
}

# 공개 dev data(aimo-interp/val-sample)의 row field. 이 이름 밖의 field를 추측하지 않습니다.
OFFICIAL_ROW_FIELDS = (
    "model_id",
    "dataset_id",
    "problem_id",
    "original_problem",
    "reasoning_effort",
    "model_is_robust",
)
