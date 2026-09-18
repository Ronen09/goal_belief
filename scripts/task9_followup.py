"""TASK9 follow-up: which cut carries the cue in the transformer? Patch the layer-1 residual
from a B-branch sequence into an A-branch sequence at the cue position t, at t+1, and at
both (the complete cut between the cue token and the prediction at t+1)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import hmm4 as H4, prominence as Pr, tfm as Tf, tfm_measure as Tm, invariants as Iv

out = Path("results9"); models = sorted(p.stem for p in (out / "models").glob("*.pt"))
m = H4.make_hmm4(1.0, 0.4, 1); ev = Pr.make_eval(m, n=2000, T=48); rng = np.random.default_rng(0)
P, aA, jB = Tm.anchors_of(m, ev["Z"], 4, 200, rng); X = torch.as_tensor(ev["X"]); i, t = aA[:, 0], aA[:, 1]; j, tj = jB[:, 0], jB[:, 1]
ar = torch.arange(len(i)); tt = torch.as_tensor(t)
res = {}
for name in models:
    label = name.rsplit("_s", 1)[0]; seed = int(name.rsplit("_s", 1)[1])
    gain = float(label.split("gain")[1].split("_")[0]) if label.startswith("gain") else None
    if not (label.startswith(("base", "lr", "gain"))): continue
    net = Tf.CausalTransformer(n_vocab=H4.V, final_ln=not label.endswith("noln"), seed=seed, gain=gain)
    net.load_state_dict(torch.load(out / "models" / f"{name}.pt")); net.eval()
    with torch.no_grad():
        R1 = net.residuals(X)[net.cut]
        base = torch.softmax(net.run_from(R1[i], net.cut), -1)[ar, tt + 1]
        def patched(pos_mask):
            R = R1[i].clone()
            for off in pos_mask:
                R[ar, tt + off] = R1[j, tj + off]
            return torch.softmax(net.run_from(R, net.cut), -1)[ar, tt + 1]
        js = {k: float(Iv.js_divergence(patched(v).numpy(), base.numpy()).mean()) for k, v in (("t", (0,)), ("t+1", (1,)), ("both", (0, 1)))}
        # attention of layer-2 at t+1 to position t vs t+1 (route weights), averaged over heads and anchors
        x = net.blocks[1].ln1(R1[i]); mask = net._mask(x.shape[1], x.device)
        _, w = net.blocks[1].attn(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=True)
        js["attn_t"] = float(w[ar, tt + 1, tt].mean()); js["attn_t1"] = float(w[ar, tt + 1, tt + 1].mean())
        logp = torch.log_softmax(net.forward_all(X[:500]), -1)[:, :-1].numpy(); Y = ev["Y"][:500, :-1]
        js["kl"] = float((Y * (np.log(Y + 1e-12) - logp)).sum(-1).mean())
    res[name] = js; print(name, {k: round(v, 3) for k, v in js.items()}, flush=True)
json.dump(res, open(out / "followup_cuts.json", "w"), indent=1)
conv = {k: v for k, v in res.items() if v["kl"] < 0.003}
for key in ("t", "t+1", "both"):
    v = np.array([r[key] for r in conv.values()]); print(f"JS patch {key}: mean {v.mean():.3f} CV {v.std() / v.mean():.3f} n={len(v)}")
