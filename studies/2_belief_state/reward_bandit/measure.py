"""The reward-bandit experiment: behaviour against the exact references, the posterior's decodability with held-out
splits, equal-belief pairs, the belief-node table, transplants along the probe at the state the heads read, and the
within-action ladder. Measures and decision rule: PLAN.md.

    .venv/bin/python studies/2_belief_state/reward_bandit/measure.py                       # writes results.json
    .venv/bin/python studies/2_belief_state/reward_bandit/measure.py --untrained --seeds 0   # smoke test (initial checkpoints)
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bandit as BD, beliefcausal as BC, beliefprobe as BP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as TR                                                   # noqa: E402

N_EVAL, EXT_FIT, EXT_TEST, NONDEG = 16384, 2.0, 3.0, 1e-6


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(np.clip(a, 1e-300, None)) - np.log(np.clip(b, 1e-300, None)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def constant(t):
    """The evidence-blind reference: the action with the best prior-mean reward, always."""
    a = int(t.R.mean(0).argmax())
    return lambda env: torch.full((env.N,), a, device=t.dev)


@torch.no_grad()
def sites(net, tok, pos, arch):
    """Representation sites at the decision positions, flattened: {name: [N*T, d]} (double, cpu)."""
    out = {}
    names = ["res0", "res1", "res2"][: net.nl + 1] if arch == "tfm" else ["h"]
    for i in range(0, len(tok), 4096):
        _, _, rec = net(tok[i:i + 4096], record=True)
        for k, R in zip(names, rec["resid"]):
            out.setdefault(k, []).append(R[:, pos].flatten(0, 1).double().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def heads(net, h):
    """Action distribution the heads give to final states h [m, d] (numpy)."""
    x = torch.as_tensor(h, dtype=torch.float32, device=next(net.parameters()).device)
    return torch.softmax(net.pi(net.ln(x)), -1).double().cpu().numpy()


class Decisions:
    """The evaluation episodes' decisions: beliefs, log-odds, splits, optimal actions, node ids."""

    def __init__(self, t, r, seed=0):
        N, T = r.act.shape
        self.N, self.T = N, T
        self.b = r.b.flatten(0, 1).double().cpu().numpy()                                     # [N*T, K]
        self.q = r.q.flatten(0, 1).double().cpu().numpy()                                     # [N*T, A]
        self.v = self.q.max(1)
        self.y = np.log(np.clip(self.b[:, :-1], 1e-300, None)) - np.log(np.clip(self.b[:, -1:], 1e-300, None))
        self.nondeg = self.b.min(1) > NONDEG
        ymax = np.abs(self.y).max(1)
        self.step = np.tile(np.arange(T), N)
        later = self.step >= 1                                                                  # the extrapolation split leaves out the first decision (all its states are in the fit region)
        self.ext_fit, self.ext_test = self.nondeg & later & (ymax < EXT_FIT), self.nondeg & later & (ymax >= EXT_TEST)
        self.action = self.q.argmax(1)
        self.ep = np.repeat(np.arange(N), T)
        self.node = (r.cue_id[:, None] * t.B.shape[1] + r.oi).flatten().cpu().numpy()         # the belief state: cue state × outcome counts
        self.fold = np.random.default_rng(seed).permutation(N)[self.ep] % 5


def probes(H, D: Decisions):
    """Decodability of y (log-odds) and b (probabilities): 5-fold in distribution on the non-degenerate states, the
    extrapolation split, the optimal action, V* and Q*; within-action R² of y (the ladder)."""
    H = np.asarray(H, np.float64)
    nd = D.nondeg
    out = {}
    Py = BP.cv_pred(H[nd], D.y[nd], D.fold[nd])
    out["IID_r2y"] = BP.r2(D.y[nd], Py)
    out["IID_r2b"] = BP.r2(D.b, BP.cv_pred(H, D.b, D.fold))
    out["EXT_r2y"] = BP.r2(D.y[D.ext_test], BP.probe_pred(H, D.y, D.ext_fit, D.ext_test))
    out["EXT_r2b"] = BP.r2(D.b[D.ext_test], BP.probe_pred(H, D.b, D.ext_fit, D.ext_test))
    onehot = np.eye(D.q.shape[1])[D.action]
    out["action_acc"] = float((BP.cv_pred(H, onehot, D.fold).argmax(1) == D.action).mean())
    out["IID_r2v"] = BP.r2(D.v[:, None], BP.cv_pred(H, D.v[:, None], D.fold))
    out["IID_r2q"] = BP.r2(D.q, BP.cv_pred(H, D.q, D.fold))
    sse = sst = 0.0
    for a in np.unique(D.action):
        m = nd & (D.action == a)
        if m.sum() < 200:
            continue
        P = BP.cv_pred(H[m], D.y[m], D.fold[m])
        sse += ((D.y[m] - P) ** 2).sum(); sst += ((D.y[m] - D.y[m].mean(0)) ** 2).sum()
    out["r2y_within_action"] = float(1 - sse / sst) if sst > 0 else float("nan")
    return out


def node_table(lg, D: Decisions):
    """How much of the centred logits' variance a table over belief nodes explains (the rest is history dependence
    beyond the belief)."""
    lg = lg - lg.mean(1, keepdims=True)
    _, inv = np.unique(D.node, return_inverse=True)
    means = np.zeros((inv.max() + 1, lg.shape[1])); cnt = np.zeros(inv.max() + 1)
    np.add.at(means, inv, lg); np.add.at(cnt, inv, 1)
    means /= cnt[:, None]
    within = ((lg - means[inv]) ** 2).sum()
    total = ((lg - lg.mean(0)) ** 2).sum()
    multi = cnt[inv] > 1
    return dict(r2=float(1 - within / total), nodes=int(len(cnt)), decisions_in_shared_nodes=float(multi.mean()))


@torch.no_grad()
def equal_belief(net, t, r, gen):
    """Permute the cue tokens and the event tokens of every history (the counts, so the belief, are unchanged): the
    divergence of the model's action distribution, against a random history of the same step."""
    tok, N, T = r.tok, r.act.shape[0], r.act.shape[1]
    dev = tok.device
    out = {"equal_js": [], "equal_flip": [], "random_js": [], "random_flip": [], "changed": []}
    lg_all = torch.cat([net(tok[i:i + 4096])[0] for i in range(0, N, 4096)])
    for s in range(T):
        p = t.n_cue + s
        tk = tok[:, : p + 1].clone()
        perm = torch.argsort(torch.rand(N, t.n_cue, device=dev, generator=gen), 1)
        tk[:, 1: 1 + t.n_cue] = torch.gather(tok[:, 1: 1 + t.n_cue], 1, perm[..., None].expand(-1, -1, BD.NF))
        if s > 0:
            perm = torch.argsort(torch.rand(N, s, device=dev, generator=gen), 1)
            tk[:, t.n_cue + 1: p + 1] = torch.gather(tok[:, t.n_cue + 1: p + 1], 1, perm[..., None].expand(-1, -1, BD.NF))
        changed = (tk != tok[:, : p + 1]).flatten(1).any(1)
        pa = torch.softmax(lg_all[:, p], -1).double().cpu().numpy()
        pb = torch.softmax(torch.cat([net(tk[i:i + 4096])[0][:, p] for i in range(0, N, 4096)]), -1).double().cpu().numpy()
        pr = np.roll(pa, 1, 0)
        c = changed.cpu().numpy()
        out["equal_js"].append(js(pa, pb)[c]); out["equal_flip"].append((pa.argmax(1) != pb.argmax(1))[c])
        out["random_js"].append(js(pa, pr)); out["random_flip"].append(pa.argmax(1) != pr.argmax(1)); out["changed"].append(c)
    cat = {k: np.concatenate(v) for k, v in out.items()}
    res = {k: float(v.mean()) for k, v in cat.items()}
    res["js_ratio"] = res["equal_js"] / res["random_js"]
    return res


def transplant(net, H, D: Decisions, rng):
    """At the state the heads read: fit the probe (decoder and encoder) on the non-degenerate states; for pairs (A, B)
    of decisions at the same step, move A's state to B's log-odds along the encoder (probe_e) or the decoder's
    pseudo-inverse (probe_d), or by a random direction of the same length; compare the heads' action distribution
    with the model's own on B."""
    nd = np.nonzero(D.nondeg)[0]
    bmap = BC.BeliefMap(H[nd], D.y[nd])
    perm = rng.permutation(len(nd))
    A, B = nd, nd[perm]
    same_step = D.step[A] == D.step[B]
    A, B = A[same_step], B[same_step]
    hA, hB, yB = H[A], H[B], D.y[B]
    pA, pB = heads(net, hA), heads(net, hB)
    he = bmap.probe_e(hA, yB)
    states = {"probe_e": he, "probe_d": bmap.probe_d(hA, yB), "rand": bmap.rand_like(hA, he - hA, rng)}
    base = js(pA, pB)
    out = dict(pairs=int(len(A)), none_js=float(base.mean()), opt_agree_A_on_B=float((pA.argmax(1) == D.action[B]).mean()),
               model_B_opt=float((pB.argmax(1) == D.action[B]).mean()))
    for k, h in states.items():
        p = heads(net, h)
        out[f"{k}_gap_closed"] = float(1 - js(p, pB).mean() / base.mean())
        out[f"{k}_same_action_as_B"] = float((p.argmax(1) == pB.argmax(1)).mean())
        out[f"{k}_opt_action_at_B"] = float((p.argmax(1) == D.action[B]).mean())
        out[f"{k}_decoded_r2"] = BP.r2(yB, bmap.z(h))
    return out


def measure_model(net, arch, t, goal, cues, u, seed, dev):
    r = BD.rollout(net, t, len(goal), None, greedy=True, goal=goal, cues=cues, u=u)
    row = dict(behaviour=BD.summarize(r, t)[0])
    D = Decisions(t, r, seed)
    row["decisions"] = dict(total=int(len(D.b)), nondegenerate=float(D.nondeg.mean()), ext_fit=float(D.ext_fit.mean()), ext_test=float(D.ext_test.mean()))
    S = sites(net, r.tok, torch.as_tensor(D.step[: D.T] + t.n_cue, device=dev), arch)
    row["probes"] = {k: probes(H, D) for k, H in S.items()}
    with torch.no_grad():
        lg = torch.cat([net(r.tok[i:i + 4096])[0] for i in range(0, len(goal), 4096)])[:, t.n_cue: t.n_cue + t.T].flatten(0, 1).double().cpu().numpy()
    row["node_table"] = node_table(lg, D)
    gen = torch.Generator(device=dev); gen.manual_seed(300 + seed)
    row["equal_belief"] = equal_belief(net, t, r, gen)
    final = "res2" if arch == "tfm" else "h"
    final = final if final in S else list(S)[-1]
    row["transplant"] = transplant(net, S[final], D, np.random.default_rng(400 + seed))
    row["final_site"] = final
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["tfm", "gru"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--untrained", action="store_true", help="smoke test: initial checkpoints only, fewer episodes")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    task = json.load(open(Path(a.runs) / a.archs[0] / "task.json"))["task"]
    s = BD.Spec(**task)
    t = BD.Sim(BD.Graph(s), dev)
    n_eval = 2048 if a.untrained else N_EVAL
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BD.Env(t, n_eval, gen)
    goal, cues, u = e.goal, e.cues, e.u
    res = dict(task=task, states=int(t.B.shape[0] * t.B.shape[1]), references={}, runs={})
    for name, pol in (("optimal", BD.optimal), ("myopic", BD.myopic), ("constant", lambda g: constant(t)), ("random", BD.uniform)):
        gen.manual_seed(79)
        res["references"][name] = BD.summarize(BD.rollout(None, t, n_eval, gen, behaviour=pol(gen), goal=goal, cues=cues, u=u), t)[0]
    print("references: " + " ".join(f"{k} {v['ret']:.3f} (regret {v['regret']:.3f})" for k, v in res["references"].items()), flush=True)
    out = Path(a.out) if a.out else HERE / ("smoke.json" if a.untrained else "results.json")
    for arch in a.archs:
        runs = Path(a.runs) / arch
        info = json.load(open(runs / "task.json")); args = info["args"]
        model_arch = info.get("arch", arch)                           # the run directory may be named after the arm (channel / iid), not the architecture
        for seed in a.seeds:
            d = runs / f"seed{seed}" / "ckpt"
            rows = {}
            for which, ck in (("trained", sorted(d.glob("u*.pt"))[-1]), ("untrained", d / "u000000.pt")):
                if a.untrained and which == "trained":
                    continue
                net = BD.build(model_arch, s, args.get("d", 128), args.get("layers", 2))
                net.load_state_dict(torch.load(ck, map_location=dev))
                net = net.to(dev).eval()
                rows[which] = dict(checkpoint=ck.name, **measure_model(net, model_arch, t, goal, cues, u, seed, dev))
            res["runs"][f"{arch}/seed{seed}"] = rows
            k = "trained" if "trained" in rows else "untrained"
            b_, p_, tr_, eq_ = rows[k]["behaviour"], rows[k]["probes"][rows[k]["final_site"]], rows[k]["transplant"], rows[k]["equal_belief"]
            print(f"{arch} seed{seed} {rows[k]['checkpoint']} regret {b_['regret']:.3f} return {b_['ret']:.3f} opt-where-pays {b_['opt_where_pays']:.2f} | "
                  f"final site IID r2y {p_['IID_r2y']:.3f} r2b {p_['IID_r2b']:.3f} EXT r2y {p_['EXT_r2y']:.3f} r2b {p_['EXT_r2b']:.3f} within-action {p_['r2y_within_action']:.3f} | "
                  f"node table {rows[k]['node_table']['r2']:.3f} | equal-belief js ratio {eq_['js_ratio']:.3f} flip {eq_['equal_flip']:.3f} (random {eq_['random_flip']:.3f}) | "
                  f"transplant probe_e {tr_['probe_e_gap_closed']:.2f} probe_d {tr_['probe_d_gap_closed']:.2f} rand {tr_['rand_gap_closed']:.2f} | {time.time() - t0:.0f}s", flush=True)
            out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
