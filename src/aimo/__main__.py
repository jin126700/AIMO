"""python -m aimo 도 aimo CLI와 같은 진입점을 씁니다."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
