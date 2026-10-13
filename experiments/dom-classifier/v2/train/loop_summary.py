# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""One line per loop_eval.json: which classes meet the goal and the best precision/recall per class (#2307)."""

import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    r = json.loads(Path(path).read_text())
    cfg = r["config"]
    cells = []
    for name, block in r["perClass"].items():
        b = block["bestAtGoalPrecision"]
        cells.append(f"{name}={'%.2f@%.2f' % (b['precision'], b['recall']) if b else '-'}")
    print(f"{Path(path).parent.name:34s} acc={r['accuracy']:.3f} met={r['goalsMet']} | " + " ".join(cells))
