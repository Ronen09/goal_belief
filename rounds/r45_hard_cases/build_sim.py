import json, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
LAB = {"r23": "Reward-trained policy", "targeted": "Reward + hard-case supervision", "control": "Reward + ordinary supervision"}
vs, maze = [], None
for name in ["r23", "targeted", "control"]:
    f = HERE / f"sim_{name}_s0.json"
    if not os.path.exists(f): continue
    d = json.load(open(f)); maze = d["maze"]
    vs.append(dict(id=name, label=LAB[name], summary=d["summary"], episodes=d["episodes"]))
html = open(HERE / "sim_template.html").read().replace("/*DATA*/null", json.dumps(dict(maze=maze, variants=vs), separators=(",", ":")))
open(HERE / "aliased_corridor.html", "w").write(html); print("variants", [v["id"] for v in vs], "bytes", len(html))
