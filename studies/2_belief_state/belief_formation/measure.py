"""Belief formation: where the posterior's probabilities appear along the reward-bandit transformer, what each
component computes of the belief, block-0 attention, and online patches at the decision token. Measures and rule:
PLAN.md. The reward-bandit experiment's trained transformers and their untrained checkpoints; no training.

    .venv/bin/python studies/2_belief_state/belief_formation/measure.py              # writes results.json, tables.md
    .venv/bin/python studies/2_belief_state/belief_formation/measure.py --tables     # tables.md from results.json
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bandit as BD, beliefprobe as BP

HERE = Path(__file__).resolve().parent
RB = HERE.parent / "reward_bandit"
sys.path.insert(0, str(RB))
import measure as ME                                                 # noqa: E402
import train as TR                                                   # noqa: E402

SITES = ("res0", "attn0", "mid0", "mlp0", "res1", "attn1", "mid1", "mlp1", "res2")
COMPS = ("attn0", "mlp0", "attn1", "mlp1")


@torch.no_grad()
def record(net, tok, pos):
    """Every site at the decision positions [N*T, d] (double, cpu), and block attention patterns from the decision
    positions [N, T, heads, L]."""
    out = {k: [] for k in SITES}
    pat = {0: [], 1: []}
    for i in range(0, len(tok), 2048):
        tk = tok[i:i + 2048]
        _, _, rec = net(tk, record=True)
        vals = dict(res0=rec["resid"][0], attn0=rec["attn"][0], mlp0=rec["mlp"][0], res1=rec["resid"][1], attn1=rec["attn"][1], mlp1=rec["mlp"][1], res2=rec["resid"][2])
        vals["mid0"] = vals["res0"] + vals["attn0"]; vals["mid1"] = vals["res1"] + vals["attn1"]
        for k in SITES:
            out[k].append(vals[k][:, pos].flatten(0, 1).double().cpu().numpy())
        x = net.embed(tk)
        for l, b in enumerate(net.blocks):
            a, A, _, _ = b.attn(x, want=True)
            pat[l].append(A[:, :, pos].permute(0, 2, 1, 3).cpu().numpy())                      # [n, T, h, L]
            x = x + a; x = x + b.mlp(b.ln2(x))
    return {k: np.concatenate(v) for k, v in out.items()}, {l: np.concatenate(v) for l, v in pat.items()}


def fits(X, Y, fold):
    return BP.r2(Y, BP.cv_pred(X, Y, fold))


def site_probes(H, D, counts):
    nd = D.nondeg
    return dict(counts_r2=fits(H, counts, D.fold), IID_r2y=fits(H[nd], D.y[nd], D.fold[nd]), IID_r2b=fits(H, D.b, D.fold),
                EXT_r2y=BP.r2(D.y[D.ext_test], BP.probe_pred(H, D.y, D.ext_fit, D.ext_test)), EXT_r2b=BP.r2(D.b[D.ext_test], BP.probe_pred(H, D.b, D.ext_fit, D.ext_test)))


def component_fits(Z, D):
    """What function of the belief a component output is: R² of affine fits from y, from b, quadratic in y, and the
    node table (the ceiling)."""
    y, b = D.y, D.b[:, :2]
    quad = np.c_[y, y ** 2, y[:, :1] * y[:, 1:]]
    out = dict(from_y=fits(y, Z, D.fold), from_b=fits(b, Z, D.fold), from_y_quadratic=fits(quad, Z, D.fold), from_step=fits(np.eye(D.T)[D.step], Z, D.fold))
    _, inv = np.unique(D.node, return_inverse=True)
    means = np.zeros((inv.max() + 1, Z.shape[1])); cnt = np.zeros(inv.max() + 1)
    np.add.at(means, inv, Z); np.add.at(cnt, inv, 1); means /= cnt[:, None]
    out["node_table"] = float(1 - ((Z - means[inv]) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum())
    return out


def attention_summary(pat, t):
    """pat [N, T, h, L]: from decision positions, mass on BOS / cue / event tokens (over the visible prefix) and the
    evenness of the mass over the history tokens, per head."""
    N, T, h, L = pat.shape
    rows = []
    for head in range(h):
        bos = cue = evt = even = 0.0; n = 0
        for s in range(T):
            p = t.n_cue + s
            A = pat[:, s, head, : p + 1]
            bos += A[:, 0].mean(); cue += A[:, 1: 1 + t.n_cue].sum(1).mean(); evt += A[:, 1 + t.n_cue: p + 1].sum(1).mean()
            hist = A[:, 1: p + 1]; hist = hist / np.clip(hist.sum(1, keepdims=True), 1e-9, None)
            ent = -(hist * np.log(np.clip(hist, 1e-12, None))).sum(1).mean()
            even += ent / np.log(p); n += 1
        rows.append({k: float(x) for k, x in dict(bos=bos / n, cue=cue / n, event=evt / n, history=(cue + evt) / n, evenness=even / n).items()})
    return rows


PATCHES = {"mlp0 mean": ("mean", ["mlp0"]), "mlp1 mean": ("mean", ["mlp1"]), "attn0 mean": ("mean", ["attn0"]), "attn1 mean": ("mean", ["attn1"]),
           "mlp0 affine in y": ("affine_y", ["mlp0"]), "mlp1 affine in y": ("affine_y", ["mlp1"]), "both MLPs affine in y": ("affine_y", ["mlp0", "mlp1"]),
           "mlp0 affine in b": ("affine_b", ["mlp0"]), "mlp1 affine in b": ("affine_b", ["mlp1"]), "both MLPs affine in b": ("affine_b", ["mlp0", "mlp1"]),
           # post hoc, not registered: block 1's attention cannot import a probability code made at earlier positions
           "attn1 mean + mlp0 affine in y": ({"attn1": "mean", "mlp0": "affine_y"}, ["attn1", "mlp0"]),
           "attn1 mean + both MLPs affine in y": ({"attn1": "mean", "mlp0": "affine_y", "mlp1": "affine_y"}, ["attn1", "mlp0", "mlp1"]),
           "attn1 mean + both MLPs affine in b": ({"attn1": "mean", "mlp0": "affine_b", "mlp1": "affine_b"}, ["attn1", "mlp0", "mlp1"])}


class Patch:
    """A component's output at the last position replaced by: its mean at that step, or its affine fit from the exact
    log-odds / probabilities of the current state (fitted on natural runs)."""

    def __init__(self, kind, comps, fit, t):
        """kind: one kind for every component in comps, or a dict component -> kind."""
        self.kinds = {c: (kind[c] if isinstance(kind, dict) else kind) for c in comps}
        self.comps, self.fit, self.t = comps, fit, t
        self.env, self.step = None, None

    def value(self, comp):
        f, kind = self.fit[comp], self.kinds[comp]
        if kind == "mean":
            return f["mean_by_step"][self.step]
        b = self.env.belief.double()
        if kind == "affine_b":
            X = torch.cat([b[:, :2], torch.ones(len(b), 1, dtype=b.dtype, device=b.device)], 1)
            return (X @ f["W_b"]).float()
        y = torch.log(b[:, :2].clamp(min=1e-300)) - torch.log(b[:, 2:].clamp(min=1e-300))
        X = torch.cat([y, torch.ones(len(y), 1, dtype=y.dtype, device=y.device)], 1)
        return (X @ f["W_y"]).float()

    def patches(self):
        def make(comp):
            def fn(z):
                z = z.clone(); z[:, -1] = self.value(comp); return z
            return fn
        return {(c[:-1], int(c[-1])): make(c) for c in self.comps}


def fit_component_maps(sites, D, dev):
    out = {}
    for c in COMPS:
        Z = sites[c]
        y, b = D.y, D.b[:, :2]
        Wy = BP._fit(y, Z); Wb = BP._fit(b, Z)
        out[c] = dict(W_y=torch.tensor(Wy, device=dev), W_b=torch.tensor(Wb, device=dev),
                      mean_by_step=torch.tensor(np.stack([Z[D.step == s].mean(0) for s in range(D.T)]), dtype=torch.float32, device=dev))
    return out


@torch.no_grad()
def patched_run(net, t, goal, cues, u, patch: Patch):
    """The policy run with the patch at every decision; return, regret, and res2 at decision positions recorded
    step by step under the same patch."""
    env = BD.Env(t, len(goal), None, goal, cues, u)
    patch.env = env
    N, T = len(goal), t.T
    reg = torch.zeros(N, T, device=t.dev); rew = torch.zeros(N, T, device=t.dev); res2 = []
    store = {}
    for s in range(T):
        patch.step = s
        p = env.pos
        pt = patch.patches(); pt[("resid", net.nl)] = lambda x: (store.__setitem__("x", x), x)[1]
        lg, _ = net(env.tok[:, : p + 1], patch=pt)
        a = lg[:, p].argmax(-1)
        res2.append(store["x"][:, p].double().cpu().numpy())
        q = env.q_star(); reg[:, s] = q.max(1).values - q.gather(1, a[:, None]).squeeze(1)
        rew[:, s] = env.step(a)
    return dict(ret=float(rew.sum(1).mean()), regret=float(reg.sum(1).mean())), np.stack(res2, 1).reshape(N * T, -1)


def measure_model(net, t, goal, cues, u, seed, dev, with_patches):
    r = BD.rollout(net, t, len(goal), None, greedy=True, goal=goal, cues=cues, u=u)
    D = ME.Decisions(t, r, seed)
    pos = torch.as_tensor(D.step[: D.T] + t.n_cue, device=dev)
    sites, pat = record(net, r.tok, pos)
    cue_counts = torch.nn.functional.one_hot(r.tok[:, 1: 1 + t.n_cue, BD.F_SYM] - 1, t.M).sum(1)                 # [N, M]
    oc = t.g.out[r.oi.cpu().numpy()]                                                                         # [N, T, 2A]
    counts = np.c_[np.repeat(cue_counts.cpu().numpy(), D.T, 0), oc.reshape(-1, oc.shape[-1])].astype(float)
    row = dict(behaviour=BD.summarize(r, t)[0], sites={k: site_probes(H, D, counts) for k, H in sites.items()},
               components={c: component_fits(sites[c], D) for c in COMPS}, attention={l: attention_summary(pat[l], t) for l in pat})
    if with_patches:
        maps = fit_component_maps(sites, D, dev)
        row["patches"] = {}
        for name, (kind, comps) in PATCHES.items():
            beh, res2 = patched_run(net, t, goal, cues, u, Patch(kind, comps, maps, t))
            row["patches"][name] = dict(**beh)
            row["patches"][name]["_res2"] = res2
        # the patched runs' beliefs differ from the natural run's; recompute Decisions per patched run
        for name in list(row["patches"]):
            res2 = row["patches"][name].pop("_res2")
            kind, comps = PATCHES[name]
            Dp = decisions_of_patched(net, t, goal, cues, u, Patch(kind, comps, maps, t), seed)
            row["patches"][name].update(EXT_r2b=BP.r2(Dp.b[Dp.ext_test], BP.probe_pred(res2, Dp.b, Dp.ext_fit, Dp.ext_test)),
                                        EXT_r2y=BP.r2(Dp.y[Dp.ext_test], BP.probe_pred(res2, Dp.y, Dp.ext_fit, Dp.ext_test)), IID_r2b=fits(res2, Dp.b, Dp.fold))
    return row


@torch.no_grad()
def decisions_of_patched(net, t, goal, cues, u, patch, seed):
    """The belief states the patched policy visits (its own greedy episodes)."""
    env = BD.Env(t, len(goal), None, goal, cues, u); patch.env = env
    N, T = len(goal), t.T
    oi = torch.zeros(N, T, dtype=torch.long, device=t.dev)
    for s in range(T):
        patch.step = s; p = env.pos
        a = net(env.tok[:, : p + 1], patch=patch.patches())[0][:, p].argmax(-1)
        oi[:, s] = env.oi; env.step(a)
    r = BD.Batch(env.tok, None, torch.zeros(N, T, dtype=torch.long), None, None, None, None, None, goal, env.ci, oi, None)
    return ME.Decisions(t, r, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE / ("_smoke" if a.quick else ".")
    out.mkdir(parents=True, exist_ok=True)
    if not a.tables:
        torch.set_grad_enabled(False)
        dev, t0 = a.device, time.time()
        s = BD.Spec(**TR.TASK); t = BD.Sim(BD.Graph(s), dev)
        n_eval = 1024 if a.quick else ME.N_EVAL
        gen = torch.Generator(device=dev); gen.manual_seed(78)
        e = BD.Env(t, n_eval, gen); goal, cues, u = e.goal, e.cues, e.u
        args = json.load(open(RB / "runs" / "tfm" / "task.json"))["args"]
        res = dict(runs={})
        for seed in a.seeds[:1] if a.quick else a.seeds:
            d = RB / "runs" / "tfm" / f"seed{seed}" / "ckpt"
            for which, ck in (("trained", sorted(d.glob("u*.pt"))[-1]), ("untrained", d / "u000000.pt")):
                net = BD.build("tfm", s, args.get("d", 128), args.get("layers", 2))
                net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
                res["runs"][f"seed{seed}/{which}"] = measure_model(net, t, goal, cues, u, seed, dev, with_patches=(which == "trained"))
                r_ = res["runs"][f"seed{seed}/{which}"]
                print(f"seed{seed} {which}: EXT b by site " + " ".join(f"{k} {r_['sites'][k]['EXT_r2b']:.2f}" for k in SITES) + " | counts attn0 %.2f res0 %.2f" % (r_["sites"]["attn0"]["counts_r2"], r_["sites"]["res0"]["counts_r2"])
                      + (" | patches: " + " ".join(f"{k}: regret {v['regret']:.3f} EXTb {v['EXT_r2b']:.2f}" for k, v in r_["patches"].items()) if "patches" in r_ else "") + f" | {time.time() - t0:.0f}s", flush=True)
                (out / "results.json").write_text(json.dumps(res))
    tables(out)


def tables(out):
    R = json.load(open(out / "results.json"))
    keys = {w: sorted(k for k in R["runs"] if k.endswith("/" + w)) for w in ("trained", "untrained")}
    v = lambda w, *p: np.array([np.nan if (x := ME_get(R["runs"][k], p)) is None else x for k in keys[w]], dtype=float)
    f = lambda x, d=2: "—" if x != x else f"{x:.{d}f}"
    mr = lambda xs, d=2: f"{f(np.median(xs), d)} ({f(np.min(xs), d)}–{f(np.max(xs), d)})"
    row = lambda *c: "| " + " | ".join(str(x) for x in c) + " |"
    L = ["# Belief formation tables", "", "Generated by `measure.py`. The reward-bandit experiment's six trained transformers (and their untrained checkpoints) at decision positions of 16 384 evaluation episodes; median (range).", "",
         "## A. Where the belief appears (affine probes)", "", row("site", "counts R² (IID)", "y R² (IID)", "b R² (IID)", "**EXT R² y**", "**EXT R² b**", "untrained: counts", "untrained: EXT b"), "|---|---|---|---|---|---|---|---|"]
    for k in SITES:
        L.append(row(f"**{k}**" if k in ("res1", "res2") else k, *[mr(v("trained", "sites", k, m)) for m in ("counts_r2", "IID_r2y", "IID_r2b", "EXT_r2y", "EXT_r2b")], mr(v("untrained", "sites", k, "counts_r2")), mr(v("untrained", "sites", k, "EXT_r2b"))))
    L += ["", "## B. What each component computes (IID R² of its output from…)", "", row("component", "affine in y", "affine in b", "quadratic in y", "the step alone", "the belief-node table (ceiling)"), "|---|---|---|---|---|---|"]
    for c in COMPS:
        L.append(row(c, *[mr(v("trained", "components", c, m), 3) for m in ("from_y", "from_b", "from_y_quadratic", "from_step", "node_table")]))
    L += ["", "## C. Attention from the decision position", "", row("block, head", "mass on BOS", "on cue tokens", "on event tokens", "evenness over history tokens (1 = uniform)"), "|---|---|---|---|---|"]
    for l in ("0", "1"):
        for h in range(4):
            L.append(row(f"block {l}, head {h}", *[mr(np.array([R["runs"][k]["attention"][l][h][m] for k in keys["trained"]])) for m in ("bos", "cue", "event", "evenness")]))
    L += ["", "## D. Patches at the decision token, online", "", row("patch", "return", "exact regret", "EXT R² b at res2", "EXT R² y at res2", "IID R² b at res2"), "|---|---|---|---|---|---|"]
    nat = v("trained", "behaviour", "regret")
    L.append(row("none", mr(v("trained", "behaviour", "ret"), 3), mr(nat, 3), mr(v("trained", "sites", "res2", "EXT_r2b")), mr(v("trained", "sites", "res2", "EXT_r2y")), mr(v("trained", "sites", "res2", "IID_r2b"))))
    for name in R["runs"][keys["trained"][0]]["patches"]:
        L.append(row(f"**{name}**" if "both" in name else name, mr(v("trained", "patches", name, "ret"), 3), mr(v("trained", "patches", name, "regret"), 3), mr(v("trained", "patches", name, "EXT_r2b")), mr(v("trained", "patches", name, "EXT_r2y")), mr(v("trained", "patches", name, "IID_r2b"))))
    yes = lambda b: "**yes**" if b else "no"
    gather = np.median(v("trained", "sites", "attn0", "counts_r2")) >= 0.9 and np.median(v("trained", "sites", "res0", "counts_r2")) <= 0.3
    first = next((k for k in SITES if np.median(v("trained", "sites", k, "EXT_r2b")) >= 0.6), None)
    form = first in ("mlp0", "res1", "mlp1", "res2")
    ry, rb = v("trained", "patches", "both MLPs affine in y", "regret"), v("trained", "patches", "both MLPs affine in b", "regret")
    ey = v("trained", "patches", "both MLPs affine in y", "EXT_r2b")
    nonlin = np.median(ry) >= 0.15 and np.median(ey) <= 0.5 and np.median(rb) <= 0.10
    L += ["", "## Decision rule (registered)", "", row("", "criterion", "value", "held"), "|---|---|---|---|",
          row("**GATHER**", "attn0 holds the counts (≥ 0.9), res0 does not (≤ 0.3)", f"{f(np.median(v('trained', 'sites', 'attn0', 'counts_r2')))} / {f(np.median(v('trained', 'sites', 'res0', 'counts_r2')))}", yes(gather)),
          row("**FORM**", "the first site with EXT R² b ≥ 0.6 is an MLP output or the residual after one", f"{first}", yes(form)),
          row("**NONLIN**", "both MLPs affine in y: regret ≥ 0.15 and EXT b ≤ 0.5; affine in b: regret ≤ 0.10", f"{f(np.median(ry), 3)}, {f(np.median(ey))}; {f(np.median(rb), 3)}", yes(nonlin))]
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def ME_get(d, p):
    for k in p:
        if isinstance(d, list):
            d = d[int(k)]
        elif isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


if __name__ == "__main__":
    main()
