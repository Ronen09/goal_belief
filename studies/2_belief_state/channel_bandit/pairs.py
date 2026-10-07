"""The channel-bandit experiment: matched-evidence counterfactuals. Each evaluation episode's cues and a random
permutation of them have the same counts and, under the channel, different exact posteriors; does the policy, and
does each site's decoded belief, separate the two orders in the Bayes direction? Measures and rule: PLAN.md.

    .venv/bin/python studies/2_belief_state/channel_bandit/pairs.py              # writes pairs.json, tables.md
    .venv/bin/python studies/2_belief_state/channel_bandit/pairs.py --tables     # tables.md from pairs.json and results.json
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from goalgeo import bandit as BD, beliefprobe as BP

HERE = Path(__file__).resolve().parent
RB = HERE.parent / "reward_bandit"
_load = lambda name, path: (lambda sp: (sys.modules.__setitem__(name, m := importlib.util.module_from_spec(sp)), sp.loader.exec_module(m), m)[2])(importlib.util.spec_from_file_location(name, path))
ME = _load("measure", RB / "measure.py")
BF = _load("bf_measure", HERE.parent / "belief_formation" / "measure.py")
SITES = BF.SITES
ARMS = ("channel", "iid")


def js(p, q):
    return ME.js(p, q)


@torch.no_grad()
def pairs_test(net, t, goal, cues, u, seed, dev):
    """On the model's own decisions: affine probes per site; then the matched pairs at the first decision."""
    r = BD.rollout(net, t, len(goal), None, greedy=True, goal=goal, cues=cues, u=u)
    D = ME.Decisions(t, r, seed)
    pos = torch.as_tensor(D.step[: D.T] + t.n_cue, device=dev)
    sites, _ = BF.record(net, r.tok, pos)
    probes = {k: BP._fit(H, D.b) for k, H in sites.items()}                                    # affine decoders of b, all decisions
    # the matched pairs: a random permutation of each episode's cues
    gen = torch.Generator(device=dev); gen.manual_seed(500 + seed)
    perm = torch.argsort(torch.rand(len(goal), t.n_cue, device=dev, generator=gen), 1)
    cues_p = torch.gather(cues, 1, perm)
    keep = (cues_p != cues).any(1)
    b, bp = BD.channel_posterior(t, cues), BD.channel_posterior(t, cues_p)
    _, Q = BD.episode_tables(t, b); _, Qp = BD.episode_tables(t, bp)
    a_star, a_star_p = Q[:, t.g.start].argmax(-1), Qp[:, t.g.start].argmax(-1)
    tok, tok_p = r.tok[:, : t.n_cue + 1].clone(), r.tok[:, : t.n_cue + 1].clone()
    tok_p[:, 1: 1 + t.n_cue, BD.F_SYM] = cues_p + 1
    p0 = t.n_cue
    out_sites, out_sites_p = {}, {}
    pa, pp = [], []
    for i in range(0, len(goal), 4096):
        lg, _, rec = net(tok[i:i + 4096], record=True); lgp, _, recp = net(tok_p[i:i + 4096], record=True)
        pa.append(torch.softmax(lg[:, p0], -1).double().cpu().numpy()); pp.append(torch.softmax(lgp[:, p0], -1).double().cpu().numpy())
        for which, rc, store in (("a", rec, out_sites), ("p", recp, out_sites_p)):
            vals = dict(res0=rc["resid"][0], attn0=rc["attn"][0], mlp0=rc["mlp"][0], res1=rc["resid"][1], attn1=rc["attn"][1], mlp1=rc["mlp"][1], res2=rc["resid"][2])
            vals["mid0"] = vals["res0"] + vals["attn0"]; vals["mid1"] = vals["res1"] + vals["attn1"]
            for k in SITES:
                store.setdefault(k, []).append(vals[k][:, p0].double().cpu().numpy())
    pa, pp = np.concatenate(pa), np.concatenate(pp)
    k_ = keep.cpu().numpy()
    b_, bp_ = b.double().cpu().numpy(), bp.double().cpu().numpy()
    db = np.abs(b_ - bp_).sum(1)
    d_model = js(pa, pp)
    diff_opt = (a_star != a_star_p).cpu().numpy()
    ga, gp = pa.argmax(1), pp.argmax(1)
    res = dict(behaviour=ME.BD.summarize(r, t)[0], pairs=int(k_.sum()), delta_b_median=float(np.median(db[k_])), share_delta_ge_02=float((db[k_] >= 0.2).mean()),
               different_optimal_action_share=float(diff_opt[k_].mean()))
    for name, m in (("all", k_), ("delta_ge_02", k_ & (db >= 0.2))):
        rho = spearmanr(d_model[m], db[m]).correlation
        dm = m & diff_opt
        res[name] = dict(n=int(m.sum()), js_model=float(d_model[m].mean()), spearman_js_vs_delta_b=float(rho),
                         greedy_differs_where_optimal_differs=float((ga != gp)[dm].mean()) if dm.any() else float("nan"),
                         right_on_both_where_optimal_differs=float(((ga == a_star.cpu().numpy()) & (gp == a_star_p.cpu().numpy()))[dm].mean()) if dm.any() else float("nan"),
                         greedy_differs_where_optimal_same=float((ga != gp)[m & ~diff_opt].mean()), sites={})
        for k in SITES:
            W = probes[k]
            ha, hp = np.concatenate(out_sites[k]), np.concatenate(out_sites_p[k])
            dec_a, dec_p = BP._pred(ha, W), BP._pred(hp, W)
            dd = dec_p - dec_a; true = bp_ - b_
            X = np.c_[dd[m], np.ones(m.sum())]
            Wl, *_ = np.linalg.lstsq(X, true[m], rcond=None)
            res[name]["sites"][k] = dict(r2_delta_b_from_decoded=BP.r2(true[m], X @ Wl), spearman_magnitudes=float(spearmanr(np.abs(dd[m]).sum(1), db[m]).correlation),
                                         decoded_delta_mean=float(np.abs(dd[m]).sum(1).mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if not a.tables:
        torch.set_grad_enabled(False)
        dev, t0 = a.device, time.time()
        task = json.load(open(HERE / "runs" / "channel" / "task.json"))["task"]
        s = BD.Spec(**task); t = BD.Sim(BD.Graph(s), dev)
        gen = torch.Generator(device=dev); gen.manual_seed(78)
        e = BD.Env(t, ME.N_EVAL, gen); goal, cues, u = e.goal, e.cues, e.u                      # channel-generated cues for both arms
        res = dict(task=task, runs={})
        for arm in ARMS:
            args = json.load(open(HERE / "runs" / arm / "task.json"))["args"]
            for seed in a.seeds:
                d = HERE / "runs" / arm / f"seed{seed}" / "ckpt"
                for which, ck in (("trained", sorted(d.glob("u*.pt"))[-1]), ("untrained", d / "u000000.pt")):
                    net = BD.build("tfm", s, args.get("d", 128), args.get("layers", 2))
                    net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
                    res["runs"][f"{arm}/seed{seed}/{which}"] = pairs_test(net, t, goal, cues, u, seed, dev)
                    r_ = res["runs"][f"{arm}/seed{seed}/{which}"]
                    print(f"{arm} seed{seed} {which}: regret {r_['behaviour']['regret']:.3f} | pairs {r_['pairs']} | spearman JS~Δb {r_['all']['spearman_js_vs_delta_b']:.2f} | greedy differs where optimal differs {r_['all']['greedy_differs_where_optimal_differs']:.2f} "
                          f"(where same {r_['all']['greedy_differs_where_optimal_same']:.2f}) | R² Δb from decoded: " + " ".join(f"{k} {r_['all']['sites'][k]['r2_delta_b_from_decoded']:.2f}" for k in SITES) + f" | {time.time() - t0:.0f}s", flush=True)
                    (HERE / "pairs.json").write_text(json.dumps(res))
    tables()


def tables():
    R = json.load(open(HERE / "pairs.json"))
    try:
        S = json.load(open(HERE / "results.json"))
    except FileNotFoundError:
        S = None
    keys = lambda arm, w: sorted(k for k in R["runs"] if k.startswith(arm + "/") and k.endswith("/" + w))
    get = lambda d, p: (get(d[p[0]], p[1:]) if p and isinstance(d, dict) and p[0] in d else (d if not p else None))
    v = lambda arm, w, *p: np.array([np.nan if (x := get(R["runs"][k], list(p))) is None else x for k in keys(arm, w)], dtype=float)
    f = lambda x, d=2: "—" if x != x else f"{x:.{d}f}"
    mr = lambda xs, d=2: f"{f(np.median(xs), d)} ({f(np.min(xs), d)}–{f(np.max(xs), d)})"
    row = lambda *c: "| " + " | ".join(str(x) for x in c) + " |"
    cols = [("channel", "trained"), ("channel", "untrained"), ("iid", "trained"), ("iid", "untrained")]
    L = ["# Channel bandit tables", "", "Generated by `pairs.py`. Six transformers per arm, both measured on the same channel-generated evaluation cues (16 384 episodes) against the channel posterior; median (range). "
         f"Matched pairs: each episode's cues and a random permutation (pairs with different tokens: {int(np.median(v('channel', 'trained', 'pairs')))}); Δb = ‖b − b′‖₁ between the exact posteriors of the two orders "
         f"(median {f(np.median(v('channel', 'trained', 'delta_b_median')), 3)}; share ≥ 0.2: {f(np.median(v('channel', 'trained', 'share_delta_ge_02')))}; the optimal first action differs in {f(np.median(v('channel', 'trained', 'different_optimal_action_share')), 3)}).", "",
         "## Behaviour", "", row("arm", "return", "exact regret", "optimal actions", "where information pays: optimal action"), "|---|---|---|---|---|"]
    for arm, w in cols:
        L.append(row(f"{arm} {w}", mr(v(arm, w, "behaviour", "ret"), 3), mr(v(arm, w, "behaviour", "regret"), 3), mr(v(arm, w, "behaviour", "opt_rate")), mr(v(arm, w, "behaviour", "opt_where_pays"))))
    if S:
        for k, kl in (("optimal", "optimal"), ("myopic", "myopic"), ("constant", "evidence-blind"), ("random", "random")):
            d = S["references"][k]; L.append(row(kl, f(d["ret"], 3), f(d["regret"], 3), f(d["opt_rate"]), f(d["opt_where_pays"])))
    for name, title in (("all", "## Matched pairs: behaviour (all pairs)"), ("delta_ge_02", "## Matched pairs: behaviour (pairs with Δb ≥ 0.2)")):
        L += ["", title, "", row("", *[f"{a} {w}" for a, w in cols]), "|---|---|---|---|---|"]
        for k, kl in (("js_model", "JS divergence between the two orders' action distributions"), ("spearman_js_vs_delta_b", "**Spearman(JS, Δb)** across pairs"),
                      ("greedy_differs_where_optimal_differs", "**the greedy action differs between the orders, where the optimal action differs**"), ("right_on_both_where_optimal_differs", "there: right on both orders"),
                      ("greedy_differs_where_optimal_same", "the greedy action differs where the optimal action is the same")):
            L.append(row(kl, *[mr(v(a, w, name, k)) for a, w in cols]))
    L += ["", "## Matched pairs: representation (all pairs). R² of Δb from the decoded belief difference at each site", "", row("site", *[f"{a} {w}" for a, w in cols]), "|---|---|---|---|---|"]
    for k in SITES:
        L.append(row(f"**{k}**" if k in ("mlp0", "res2") else k, *[mr(v(a, w, "all", "sites", k, "r2_delta_b_from_decoded")) for a, w in cols]))
    L += ["", "Pairs with Δb ≥ 0.2:", "", row("site", *[f"{a} {w}" for a, w in cols]), "|---|---|---|---|---|"]
    for k in SITES:
        L.append(row(k, *[mr(v(a, w, "delta_ge_02", "sites", k, "r2_delta_b_from_decoded")) for a, w in cols]))
    yes = lambda b: "**yes**" if b else "no"
    reg = v("channel", "trained", "behaviour", "regret")
    blind = S["references"]["constant"]["regret"] if S else float("nan")
    rho, gd, gd_iid = v("channel", "trained", "all", "spearman_js_vs_delta_b"), v("channel", "trained", "all", "greedy_differs_where_optimal_differs"), v("iid", "trained", "all", "greedy_differs_where_optimal_differs")
    r2c, r2i = v("channel", "trained", "all", "sites", "mlp0", "r2_delta_b_from_decoded"), v("iid", "trained", "all", "sites", "mlp0", "r2_delta_b_from_decoded")
    L += ["", "## Decision rule (registered)", "", row("", "criterion", "value", "held"), "|---|---|---|---|",
          row("**LEARN**", "channel arm: regret ≤ a quarter of the evidence-blind policy's", f"{f(np.median(reg), 3)} against {f(0.25 * blind, 3)}", yes(np.median(reg) <= 0.25 * blind)),
          row("**ORDER-B**", "Spearman(JS, Δb) ≥ 0.5 and the greedy action differs where it should in ≥ 0.5; i.i.d. arm ≤ 0.1", f"{f(np.median(rho))}; {f(np.median(gd))} (i.i.d. {f(np.median(gd_iid))})", yes(np.median(rho) >= 0.5 and np.median(gd) >= 0.5 and np.median(gd_iid) <= 0.1)),
          row("**ORDER-R**", "mlp0: R² of Δb from the decoded difference ≥ 0.5; i.i.d. arm ≤ 0.1", f"{f(np.median(r2c))} (i.i.d. {f(np.median(r2i))})", yes(np.median(r2c) >= 0.5 and np.median(r2i) <= 0.1))]
    (HERE / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
