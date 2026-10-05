from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, json
sys.path.insert(0, __HERE__); sys.path.insert(0, __ROOT__)
from screen import make, score, LAYOUTS, X, X_START
C = {
  "cross (r18), r18 goals": ("cross (r18)", [(1, 5), (2, 6), (1, 10)]),
  "cross, goals (0,6) (1,4) (2,2)": ("cross (r18)", [(0, 6), (1, 4), (2, 2)]),
  "mirror, goals (0,2) (0,8) (1,6)": ("mirror", [(0, 2), (0, 8), (1, 6)]),
  "twin-stubs, goals (0,8) (1,4) (2,2)": ("twin-stubs", [(0, 8), (1, 4), (2, 2)]),
}
out = {}
for name, (lay, goals) in C.items():
    rows, st = LAYOUTS[lay]; out[name] = score(make(rows, goals, st))
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
rows, st = LAYOUTS["mirror"]
cells = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != "." and (r, c) not in st]
out["mirror, goal = any non-start cell"] = score(make(rows, cells, st))
print("mirror, goal = any non-start cell", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["mirror, goal = any non-start cell"].items()})
json.dump(out, open(__HERE__ + "screen_confirm.json", "w"), indent=1)
