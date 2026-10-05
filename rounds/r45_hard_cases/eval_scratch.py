from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[2])
import sys, json, statistics as st, torch
from pathlib import Path
SP = __HERE__ + ""
exec(open(SP + "finetune.py").read().split("res = {}")[0])
from goalgeo import mazeaux as AX
res = {}
for arm in ("targeted", "control"):
    for seed in range(10):
        ck = sorted((Path(SP) / "runs" / arm / f"seed{seed}" / "ckpt").glob("u*.pt"))[-1]
        net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        res.setdefault(f"seed{seed}", {})[arm] = evaluate(net, seed)
ft = json.load(open(SP + "finetune_results.json"))
for s in res: res[s]["baseline (r23)"] = ft[s]["before"]
json.dump(res, open(SP + "scratch_results.json", "w"))
for arm in ("baseline (r23)", "targeted", "control"):
    g = lambda f: round(st.median([f(res[s][arm]) for s in res]), 4)
    print(f"{arm:15s} uns nat {g(lambda x: x['uns']['nat'])} add {g(lambda x: x['uns']['add'])} gap {g(lambda x: x['uns']['nat'] - x['uns']['add'])} | solv {g(lambda x: x['solv']['nat'])} | nondep {g(lambda x: x['nondep']['nat'])} | addshare {g(lambda x: x['additive_share'])} | regret {g(lambda x: x['greedy_regret'])}")
    print("    per-seed regret", [round(res[s][arm]['greedy_regret'], 4) for s in res], "uns nat", [round(res[s][arm]['uns']['nat'], 3) for s in res])
