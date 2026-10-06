"""The interior-goals experiment, collect arm: behaviour, first target, information seeking before the first pickup,
and the additive policy over goal states, run in the environment. Measures and decision rule: PLAN.md.

    .venv/bin/python studies/7_information_seeking/interior_goals/measure_collect.py                 # writes results_collect.json
    .venv/bin/python studies/7_information_seeking/interior_goals/measure_collect.py --untrained --seeds 0   # smoke test

Goal state: the set of goals still to collect. Index 0..P-1: a pair (nothing collected yet); P + y: goal y alone
(the other one collected). For a history, the alternatives are the pairs that contain every collected goal and no goal
cell the agent has walked over without a pickup. H(h) = mean over alternatives of the centred logits with the goal
tokens replaced; G[goal state, step] = the mean deviation from H on separate episodes (step: moves made before the
first pickup, moves since it afterwards).
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

from goalgeo import bigcollect as BC, bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as TR                                                   # noqa: E402

N_EVAL, SB, TOL = 4096, 25, 1e-6


class Ctx:
    """What the measurement tracks beside the environment: goal cells walked over, moves since the last pickup."""

    def __init__(self, env):
        t = env.t
        self.visited = torch.zeros(env.N, dtype=torch.long, device=t.dev)                        # bit mask of goal cells entered
        self.since = torch.zeros(env.N, dtype=torch.long, device=t.dev)
        self.first = torch.full((env.N,), -1, dtype=torch.long, device=t.dev)                    # the first goal collected

    def update(self, env, hit, live):
        t = env.t
        gj = t.goal_of_cell[env.cell]
        self.visited = torch.where(live & (gj >= 0), self.visited | (1 << gj.clamp(min=0)), self.visited)
        self.since = torch.where(hit, torch.zeros_like(self.since), self.since + 1)
        self.first = torch.where(hit & (self.first < 0), gj, self.first)

    def state(self, env, s):
        """Per episode: collected mask, valid alternatives [N, P], goal-state id of every alternative [N, P], of the
        true goals [N], step bin [N]."""
        t = env.t
        P = len(t.pairs)
        coll = t.pair_mask[env.pair] & ~env.mask                                                 # collected bits
        walked = self.visited & ~coll
        pm = t.pair_mask[None]
        valid = ((pm & coll[:, None]) == coll[:, None]) & ((pm & walked[:, None]) == 0)
        rest = pm & ~coll[:, None]                                                               # goals left under each alternative
        one = coll != 0
        y = torch.log2(rest.clamp(min=1).float()).round().long().clamp(max=t.K - 1)                                 # the single goal left (when one is collected)
        ids = torch.where(one[:, None], P + y, torch.arange(P, device=t.dev)[None].expand(env.N, -1))
        true = ids.gather(1, env.pair[:, None]).squeeze(1)
        sb = torch.where(one, self.since, torch.full_like(self.since, s)).clamp(max=SB - 1)
        return valid, ids, true, sb, one


def logits_alts(net, env, s):
    """Centred logits at the current decision under every pair of goals: [N, P, 4]."""
    t = env.t
    P = len(t.pairs)
    tok = env.tok[:, : s + t.m + 1].repeat(P, 1, 1)
    for j in range(t.m):
        tok[:, 1 + j, MM.F_GOAL] = t.pairs[:, j].repeat_interleave(env.N) + 1
    lg = net(tok)[0][:, s + t.m].view(P, env.N, 4).transpose(0, 1).double()
    return lg - lg.mean(-1, keepdim=True)


def natural(net):
    def f(env, s, ctx):
        return net(env.tok[:, : s + env.t.m + 1])[0][:, s + env.t.m].argmax(-1), None
    return f


def probe(net):
    """The natural policy, also returning what the additive code needs at this decision."""
    def f(env, s, ctx):
        lc = logits_alts(net, env, s)
        valid, ids, true, sb, one = ctx.state(env, s)
        n = torch.arange(env.N, device=lc.device)
        return lc[n, env.pair].argmax(-1), dict(lc=lc, valid=valid, ids=ids, true=true, sb=sb, one=one)
    return f


def additive(net, G, kind):
    def f(env, s, ctx):
        valid, ids, true, sb, one = ctx.state(env, s)
        g = G[true, sb]
        if kind == "history_blind":
            return g.argmax(-1), None
        lc = logits_alts(net, env, s)
        w = valid.double()[..., None]
        H = (lc * w).sum(1) / w.sum(1).clamp(min=1)
        return (H + g if kind == "additive" else H).argmax(-1), None
    return f


def behaviour(fn):
    return lambda env, s, ctx: (fn(env), None)


@torch.no_grad()
def episodes(t, policy, pair, cell, seed, full=False):
    gen = torch.Generator(device=t.dev); gen.manual_seed(seed)
    env = BC.Env(t, len(pair), gen, pair, cell)
    ctx = Ctx(env)
    N, H = env.N, t.H
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=t.dev)
    act, alive, rew, before = z(dt=torch.long), z(dt=torch.bool), z(), z(dt=torch.bool)
    qv, ee, en = (z(4), z(4), z()) if full else (None, None, None)
    extra = []
    spawn = env.cell.clone()
    for s in range(H):
        live = ~env.done
        alive[:, s], before[:, s] = live, env.mask == t.pair_mask[env.pair]
        if full:
            qv[:, s], ee[:, s], en[:, s] = BC.qmdp_values(env), BC.expected_entropy(env), BM.entropy(env.belief)
        a, x = policy(env, s, ctx)
        if x is not None:
            extra.append(x)
        act[:, s] = a
        hit = env.step(a).bool()
        rew[:, s] = hit.float()
        ctx.update(env, hit, live)
    disc = t.gamma ** torch.arange(H, device=t.dev)
    c = rew.sum(1)
    d = t.dist[t.pairs[pair].T, spawn[None]]                                                 # [m, N] path from the spawn to each active goal
    nearer = t.pairs[pair].gather(1, d.argmin(0)[:, None]).squeeze(1)
    clear = (c >= 1) & (d[0] != d[1])
    out = dict(ret=float((rew * disc).sum(1).mean()), collected=float(c.mean()), success=float((c >= t.m).float().mean()), first=float((c >= 1).float().mean()),
               length=float(alive.sum(1).float().mean()), first_is_nearer=float((ctx.first == nearer)[clear].float().mean()))
    rec = dict(act=act, alive=alive, before=before, qv=qv, ee=ee, en=en, first=ctx.first, spawn=spawn)
    if extra:
        rec.update({k: torch.stack([e[k] for e in extra], 1) for k in extra[0]})
    return out, rec


def seeking(r):
    """Deviations from QMDP and their expected information gain, on decisions before the first pickup."""
    a, m = r["act"][..., None], r["alive"] & r["before"]
    best = r["qv"].max(-1, keepdim=True).values
    dev = m & (r["qv"].gather(2, a).squeeze(2) < best.squeeze(2) - TOL)
    ig = r["en"][..., None] - r["ee"]
    ig_q = ig.gather(2, r["qv"].argmax(-1, keepdim=True)).squeeze(2)
    ig_a = ig.gather(2, a).squeeze(2)
    nonbest = r["qv"] < best - TOL
    ig_rand = (ig * nonbest).sum(-1) / nonbest.sum(-1).clamp(min=1)
    f = lambda x, w: float(x[w].mean()) if w.any() else None
    return dict(deviation_rate=f(dev.float(), m), ig_advantage=f(ig_a - ig_q, dev), ig_advantage_random_same_state=f(ig_rand - ig_q, dev),
                more_informative=f((ig_a > ig_q + TOL).float(), dev), qmdp_value_given_up=f(best.squeeze(2) - r["qv"].gather(2, a).squeeze(2), dev),
                entropy_mean=f(r["en"], m), entropy_by_step=[f(r["en"][:, s], m[:, s]) for s in (0, 5, 10, 20)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoints")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    runs = HERE / "runs" / "collect"
    task = json.load(open(runs / "task.json"))["task"]
    t, _, build = TR.setup("collect", task, dev)
    P, K = len(t.pairs), t.K
    n_eval = 512 if a.untrained else N_EVAL
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BC.Env(t, n_eval, gen)
    pair, cell = e.pair, e.cell
    gen.manual_seed(178)
    e = BC.Env(t, n_eval, gen)
    fpair, fcell = e.pair, e.cell
    res = dict(task=task, cells=t.n, pairs=P, references={}, runs={})
    rgen = torch.Generator(device=dev); rgen.manual_seed(5)
    rnd = lambda env: torch.randint(4, (env.N,), device=dev, generator=rgen)
    for name, fn in (("oracle", BC.oracle), ("qmdp2", BC.qmdp2), ("qmdp", BC.qmdp), ("mls", BC.mls), ("random", rnd)):
        s_, r = episodes(t, behaviour(fn), pair, cell, 79, full=True)
        res["references"][name] = dict(**s_, **seeking(r))
    print("references: " + " ".join(f"{k} {v['ret']:.3f} (nearer first {v['first_is_nearer']:.2f})" for k, v in res["references"].items()), flush=True)
    out = HERE / ("smoke_collect.json" if a.untrained else "results_collect.json")
    for s in a.seeds:
        d = runs / f"seed{s}" / "ckpt"
        ck = d / "u000000.pt" if a.untrained else sorted(d.glob("u*.pt"))[-1]
        net = build()
        net.load_state_dict(torch.load(ck, map_location=dev))
        net = net.to(dev).eval()
        row = dict(checkpoint=ck.name)
        s_, r = episodes(t, natural(net), pair, cell, 79, full=True)
        row["natural"] = dict(**s_, **seeking(r))
        # G on separate episodes of the natural policy
        _, rf = episodes(t, probe(net), fpair, fcell, 179)
        w = (rf["valid"] & rf["alive"][..., None]).double()                                      # [N, H, P]
        Hf = (rf["lc"] * w[..., None]).sum(2) / w.sum(2).clamp(min=1)[..., None]
        dv = rf["lc"] - Hf[:, :, None]
        key = (rf["ids"] * SB + rf["sb"][..., None]).flatten()
        Gs = torch.zeros((P + K) * SB, 4, dtype=torch.float64, device=dev).index_add_(0, key, (dv * w[..., None]).flatten(0, 2))
        Gc = torch.zeros((P + K) * SB, dtype=torch.float64, device=dev).index_add_(0, key, w.flatten())
        G = (Gs / Gc.clamp(min=1)[:, None]).view(P + K, SB, 4)
        # offline, on the evaluation episodes
        _, rt = episodes(t, probe(net), pair, cell, 79)
        w = rt["valid"] & rt["alive"][..., None]
        wd = w.double()
        Ht = (rt["lc"] * wd[..., None]).sum(2) / wd.sum(2).clamp(min=1)[..., None]
        Gt = G[rt["ids"], rt["sb"][..., None].expand_as(rt["ids"])]                              # [N, H, P, 4]
        nat_a, add_a = rt["lc"].argmax(-1), (Ht[:, :, None] + Gt).argmax(-1)
        ref_a = nat_a.gather(2, pair[:, None, None].expand(-1, t.H, 1))
        matters = ((nat_a != ref_a) & w).any(-1, keepdim=True) & w                               # the model's move depends on the goal state
        dvt = rt["lc"] - Ht[:, :, None]
        Gm = (Gt * wd[..., None]).sum(2, keepdim=True) / wd.sum(2).clamp(min=1)[..., None, None]
        off = {}
        for name, ph in (("before", ~rt["one"]), ("after", rt["one"]), ("all", torch.ones_like(rt["one"]))):
            m = matters & ph[..., None]
            wm = w & ph[..., None]
            off[name] = dict(goal_matters_share=float(matters.any(-1)[ph & rt["alive"]].float().mean()) if (ph & rt["alive"]).any() else None,
                             same_action_goal_matters=float((nat_a == add_a)[m].float().mean()) if m.any() else None,
                             same_action_goal_blind=float((nat_a == Ht.argmax(-1)[..., None])[m].float().mean()) if m.any() else None,
                             additive_share=float(1 - (((dvt - (Gt - Gm)) ** 2).sum(-1)[wm]).sum() / ((dvt ** 2).sum(-1)[wm]).sum()) if wm.any() else None)
        row["additive_offline"] = off
        row["online"] = {k: episodes(t, additive(net, G, k), pair, cell, 79)[0] for k in ("additive", "goal_blind", "history_blind")}
        res["runs"][f"seed{s}"] = row
        nt, o = row["natural"], row["online"]
        rec = (o["additive"]["ret"] - o["goal_blind"]["ret"]) / (nt["ret"] - o["goal_blind"]["ret"]) if nt["ret"] != o["goal_blind"]["ret"] else float("nan")
        print(f"seed{s} {ck.name} return {nt['ret']:.3f} (both {nt['success']:.2f}, nearer first {nt['first_is_nearer']:.2f}) | deviations {nt['deviation_rate']:.3f} "
              f"IG adv {nt['ig_advantage']:.4f} | additive {o['additive']['ret']:.3f} goal-blind {o['goal_blind']['ret']:.3f} history-blind {o['history_blind']['ret']:.3f} "
              f"recovery {rec:.2f} | same action before {off['before']['same_action_goal_matters']} after {off['after']['same_action_goal_matters']} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
