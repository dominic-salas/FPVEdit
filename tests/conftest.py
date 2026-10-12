from __future__ import annotations

import sys
from pathlib import Path

# Make the fixture generator importable as a plain module from the tests directory.
sys.path.insert(0, str(Path(__file__).parent))
