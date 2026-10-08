"""Pair terms: H on decisions where the belief is split between two cells, against five readings of the pair term
(mixture, robust, commit, disambiguate, lookahead). Plan: PLAN.md. Reuses the H-simplex experiment's decisions and fits.

    .venv/bin/python studies/7_information_seeking/pair_terms/run.py                        # writes results.json
    .venv/bin/python studies/7_information_seeking/pair_terms/run.py --untrained --seeds 0 --out DIR     # smoke test
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "H_simplex")); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as HS, posthoc2 as P2, measure as MS, train as TR       # noqa: E402

MASS, MINOR, CERTAIN, MIN_CELL = 0.85, 0.15, 0.9, 20


def opt_moves(t):
    dn = t.dist[:, t.nxt_cell]                                                                 # [K, n, 4]
    return dn <= dn.min(-1, keepdim=True).values + 1e-6


def lookahead_values(t, b, k, goals):
    """Goal-averaged and per-goal one-step-lookahead values [N, 4] / [N, K, 4] for beliefs b [N, n] with k [N] moves
    left (the maze10 experiment's qmdp2, from the belief alone)."""
    N, K = len(b), len(t.goal_cell)
    b = b.float()
    per = torch.zeros(N, K, 4, device=b.device)
    for a in range(4):
        p = torch.zeros_like(b).scatter_add_(1, t.nxt_cell[:, a][None].expand(N, -1), b)
        for g in range(K):
            gc = t.goal_cell[g]
            r = p[:, gc]
            pg = p.clone(); pg[:, gc] = 0
            w = pg[:, None, :] * t.E.T[None]                                                   # [N, sym, n]
            Q1 = t.q_mdp(torch.full((N,), g, device=b.device), k - 1)                           # [N, n, 4]
            per[:, g, a] = r + t.gamma * torch.einsum("nos,nsb->nob", w, Q1).max(-1).values.sum(-1)
    return per.mean(1).double(), per.double()


def two_cell(d, t, H_len, h_cells, popt, opt):
    """The two-cell decisions of a decision dict, with every rule's profile."""
    b = d["b"]
    top = b.topk(2, 1)
    sel = (top.values.sum(1) >= MASS) & (top.values[:, 1] >= MINOR)
    idx = torch.nonzero(sel).squeeze(1)
    s1, s2 = top.indices[idx, 0], top.indices[idx, 1]
    w = top.values[idx, 0] / top.values[idx].sum(1)
    K = opt.shape[0]
    both = (opt[:, s1] & opt[:, s2]).double()                                                  # [K, m, 4]
    out = dict(idx=idx, s1=s1, s2=s2, w=w, H=d["H"][idx], st=d["st"][idx], goal=d["goal"][idx], b=b[idx],
               conflict=(both.sum(-1) == 0).double().mean(0),
               mixture=w[:, None] * h_cells[s1] + (1 - w[:, None]) * h_cells[s2],
               mixture_popt=w[:, None] * popt[s1] + (1 - w[:, None]) * popt[s2],
               robust=both.mean(0), commit=popt[s1], disambiguate=P2.eig_of(t, {"b": b[idx], "ent": d["ent"][idx]}),
               covered=(d["cover"][s1] & d["cover"][s2]) if "cover" in d else None)
    k = (H_len - d["st"][idx]).long()
    out["lookahead"], out["lookahead_goal"] = lookahead_values(t, b[idx], k, None)
    # per-goal rules for behaviour
    Q = torch.stack([t.q_mdp(torch.full((len(idx),), g, device=b.device), k) for g in range(K)], 1)   # [m, K, n, 4]
    out["qmdp_goal"] = torch.einsum("ms,mksa->mka", b[idx].float(), Q).double()
    out["both_goal"], out["likelier_goal"] = both.permute(1, 0, 2), opt[:, s1].double().permute(1, 0, 2)
    return out


def cell_profiles(d, n):
    """h_s: the model's mean H on decisions where b(s) >= CERTAIN; cover[s]: at least MIN_CELL such decisions."""
    top = d["b"].max(1)
    h = torch.zeros(n, 4, dtype=torch.float64, device=d["b"].device); c = torch.zeros(n, dtype=torch.float64, device=d["b"].device)
    m = top.values >= CERTAIN
    h.index_add_(0, top.indices[m], d["H"][m]); c.index_add_(0, top.indices[m], torch.ones(int(m.sum()), dtype=torch.float64, device=h.device))
    return h / c.clamp(min=1)[:, None], c >= MIN_CELL, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    out_dir = Path(a.out) if a.out else HERE; out_dir.mkdir(parents=True, exist_ok=True)
    runs = HS.M10 / "runs" / "ppo"
    cfg = json.load(open(runs / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len, n = len(t.goal_cell), t.H, t.n
    n_ep = 512 if a.untrained else HS.N_EP
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, n_ep, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, n_ep, gen); fgoal, fcell = e.goal, e.cell
    popt, reach = HS.cell_candidates(t); opt = opt_moves(t)
    res = dict(task=cfg["task"], n_episodes=n_ep, runs={})
    RULES = ["mixture", "mixture_popt", "robust", "commit", "disambiguate", "lookahead"]
    for s in a.seeds:
        d = runs / f"seed{s}" / "ckpt"
        ck = d / "u000000.pt" if a.untrained else sorted(d.glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = HS.decisions(t, net, rf, K), HS.decisions(t, net, r, K)
        h_cells, cover, counts = cell_profiles(df, n)
        for dd in (df, dt): dd["cover"] = cover
        tf, tt = two_cell(df, t, H_len, h_cells, popt, opt), two_cell(dt, t, H_len, h_cells, popt, opt)
        row = dict(checkpoint=ck.name, decisions_test=len(dt["st"]), two_cell_test=len(tt["idx"]), two_cell_fit=len(tf["idx"]),
                   two_cell_share=len(tt["idx"]) / len(dt["st"]), cells_covered=int(cover.sum()), covered_share_test=float(tt["covered"].double().mean()))
        # A. departure from the mixture by conflict (covered pairs only)
        cv = tt["covered"]
        dep = ((tt["H"] - tt["mixture"]) ** 2).sum(1) / (tt["H"] ** 2).sum(1).clamp(min=1e-9)
        bins = {"0": tt["conflict"] == 0, "(0,0.5]": (tt["conflict"] > 0) & (tt["conflict"] <= 0.5), "(0.5,1]": tt["conflict"] > 0.5}
        row["departure_by_conflict"] = {k: (float(dep[m & cv].median()) if (m & cv).any() else None) for k, m in bins.items()}
        row["decisions_by_conflict"] = {k: int((m & cv).sum()) for k, m in bins.items()}
        row["conflict_mean"] = float(tt["conflict"].mean())
        # B. prediction on two-cell decisions
        S = lambda z: HS.step_onehot(z["st"], H_len)
        Y, Yt, ym = tf["H"], tt["H"], tf["H"].mean(0)
        va = df["ep"][tf["idx"]] % 10 == 0
        fit = lambda f: HS.r2(HS.Ridge(f(tf), Y, va)(f(tt)), Yt, ym)
        r2 = {"step": fit(lambda z: S(z))}
        for k in RULES:
            r2[k] = fit(lambda z, k=k: torch.cat([S(z), z[k]], 1))
        for k in RULES[2:]:
            r2["mixture+" + k] = fit(lambda z, k=k: torch.cat([S(z), z["mixture"], z[k]], 1))
        r2["mixture+all"] = fit(lambda z: torch.cat([S(z)] + [z[k] for k in RULES], 1))
        r2["mixture_raw"] = HS.r2(tt["mixture"], Yt, ym)                                        # the mixture in H's own units, no map
        r2["mlp_b"] = HS.r2(HS.fit_mlp(torch.cat([S(tf), tf["b"]], 1), Y, va)(torch.cat([S(tt), tt["b"]], 1)), Yt, ym) if len(Y) >= 500 else None
        aff = HS.Ridge(torch.cat([HS.step_onehot(df["st"], H_len), df["b"]], 1), df["H"], df["ep"] % 10 == 0)
        r2["affine_all_decisions"] = HS.r2(aff(torch.cat([S(tt), tt["b"]], 1)), Yt, ym)
        row["r2"] = r2
        # C. top action against the mixture's
        topH = tt["H"].argmax(1); topM = tt["mixture"].argmax(1)
        uniq = lambda P: (P == P.max(1, keepdim=True).values).sum(1) == 1
        row["top"] = {}
        for k in RULES[2:]:
            P = tt[k]; tr_ = P.argmax(1)
            m = cv & uniq(P) & uniq(tt["mixture"]) & (tr_ != topM)
            row["top"][k] = dict(decisions=int(m.sum()), rule=float((topH == tr_)[m].double().mean()) if m.any() else None,
                                 mixture=float((topH == topM)[m].double().mean()) if m.any() else None)
        # D. behaviour under the episode's goal
        mv = dt["lc"][tt["idx"]].argmax(-1)[torch.arange(len(tt["idx"]), device=dev), tt["goal"]]   # the model's move (greedy, own goal)
        gi = tt["goal"]; ar = torch.arange(len(gi), device=dev)
        both, lik = tt["both_goal"][ar, gi], tt["likelier_goal"][ar, gi]
        qm, q2, eig = tt["qmdp_goal"][ar, gi], tt["lookahead_goal"][ar, gi], tt["disambiguate"]
        inb = lambda M, mv_: M.gather(1, mv_[:, None]).squeeze(1) > 0
        ex = both.sum(1) > 0
        differ = ex & ((both * lik).sum(1) == 0)                                                 # a common move exists and none of them is a likelier-cell move
        row["behaviour"] = dict(common_exists_share=float(ex.double().mean()), common_differs_from_likelier=int(differ.sum()),
                                takes_common=float(inb(both, mv)[differ].double().mean()) if differ.any() else None,
                                takes_likelier=float(inb(lik, mv)[differ].double().mean()) if differ.any() else None)
        none = ~ex
        amax = lambda M: M.argmax(1)
        d3 = none & (amax(q2) != amax(eig)) & (amax(q2) != amax(lik)) | none & (amax(eig) != amax(lik))
        row["behaviour"]["no_common"] = dict(decisions=int(none.sum()), takes_likelier=float(inb(lik, mv)[none].double().mean()) if none.any() else None,
                                             takes_eig_top=float((mv == amax(eig))[none].double().mean()) if none.any() else None,
                                             takes_qmdp2_top=float((mv == amax(q2))[none].double().mean()) if none.any() else None,
                                             takes_qmdp_top=float((mv == amax(qm))[none].double().mean()) if none.any() else None)
        res["runs"][f"seed{s}"] = row
        print(f"seed{s} {ck.name} two-cell {row['two_cell_share']:.2f} of {row['decisions_test']} (covered {row['covered_share_test']:.2f}, conflict {row['conflict_mean']:.2f}) | departure "
              + " ".join(f"{k} {v:.2f}" for k, v in row['departure_by_conflict'].items() if v is not None) + " | R2 " + " ".join(f"{k} {v:.3f}" for k, v in r2.items())
              + " | top " + " ".join(f"{k} {v['rule']:.2f}/{v['mixture']:.2f}" for k, v in row['top'].items() if v['rule'] is not None)
              + f" | beh common {row['behaviour']['takes_common']} likelier {row['behaviour']['takes_likelier']} | no-common " + " ".join(f"{k} {v}" for k, v in row['behaviour']['no_common'].items())
              + f" | {time.time() - t0:.0f}s", flush=True)
        (out_dir / ("smoke.json" if a.untrained else "results.json")).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
