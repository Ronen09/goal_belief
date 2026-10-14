"""The additive decomposition of a trained policy on the four-goal bump maze (PLAN.md, part A): with the history
fixed, the centred logits under every goal L(h, g) = H(h) + G(g, step) + I(h, g); variance shares; the additive and
goal-blind policies offline (against the exact optimal sets under every goal and the ceiling) and online (recovery of
the goal-directed return).

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_goals/additive.py --model llm --run runs/llm_40000
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze")); sys.path.insert(0, str(HERE))
from design import worlds_and_solutions, K                                           # noqa: E402
from train_goals import LLMFormatGoals, LLMPolicy, SmallPolicy, test_starts, N_TEST   # noqa: E402
from train import StepTransformer, StepGRU                                            # noqa: E402

N_DEC = 4096


def load_policy(kind, run, worlds, device):
    if kind == "llm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(run / "model"); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(run / "model", dtype=torch.bfloat16, device_map=device).eval()
        return LLMPolicy(model, LLMFormatGoals(tok, worlds), device)
    H = worlds[0].H; model = StepTransformer(H) if kind == "tfm" else StepGRU(H)
    model.tok = torch.nn.Embedding(7 + K, model.tok.embedding_dim); model.load_state_dict(torch.load(run / "model.pt", map_location=device)); model = model.to(device).eval()
    return SmallPolicy(model, H, device)


def decisions_with_all_goals(worlds, sols, n_per_goal, seed):
    """Decisions along the optimum under each goal, with the optimal set under every goal whose filtered belief is
    non-empty (as in the design probe) and the goal-free set's size."""
    out = []; key = lambda m: np.packbits(m).tobytes()
    for gi, (w, sol) in enumerate(zip(worlds, sols)):
        gen = torch.Generator().manual_seed(seed + gi); e = BMZ.Env(w, n_per_goal, gen); alive_hist = []
        for t in range(w.H):
            alive_hist.append(~e.done.clone()); e.step(torch.as_tensor(BMZ.optimal_action(w, sol, e.filter.mask.numpy())[0]))
            if e.done.all(): break
        common = BMZ.Filter(w, n_per_goal, remove_goal=False); filters = [BMZ.Filter(w2, n_per_goal) for w2 in worlds]
        for t in range(len(e.acts)):
            cm = common.mask.numpy(); masks = [f.mask.numpy() for f in filters]
            for i in torch.nonzero(alive_hist[t]).squeeze(1).tolist():
                opts = {}
                for g2 in range(K):
                    m = masks[g2][i]
                    if m.any(): q = sols[g2]["Q"][sols[g2]["sets"][key(m)]]; opts[g2] = q >= q.max() - 1e-9
                if len(opts) == K:
                    out.append(dict(acts=[int(x[i]) for x in e.acts[:t]], obss=[int(x[i]) for x in e.obss[:t]], step=t, goal=gi, opts=opts, size=int(cm[i].sum())))
            common.update(e.acts[t], e.obss[t])
            for f in filters: f.update(e.acts[t], e.obss[t])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["llm", "tfm", "gru"], default="llm"); ap.add_argument("--run", default="runs/llm_40000")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--n_dec", type=int, default=N_DEC); ap.add_argument("--n_online", type=int, default=1024); a = ap.parse_args()
    run = HERE / a.run; out = Path(a.out) if a.out else run / "additive.json"; t0 = time.time()
    free, goals, worlds, sols = worlds_and_solutions(); H = worlds[0].H; policy = load_policy(a.model, run, worlds, a.device)
    design = json.loads((HERE / "design.json").read_text()); ceil = design["ceiling"]
    dec = decisions_with_all_goals(worlds, sols, 256, 500); rng = np.random.default_rng(0); pick = rng.choice(len(dec), min(a.n_dec, len(dec)), replace=False); dec = [dec[i] for i in pick]
    print(f"{len(dec)} decisions {time.time() - t0:.0f}s", flush=True)
    # logits under every goal, history fixed
    L = torch.stack([policy.logits([(d["acts"], d["obss"], g) for d in dec]) for g in range(K)], 1)      # [n, K, 3]
    Hh = L.mean(1); step = torch.tensor([min(d["step"], H - 1) for d in dec])
    G = torch.zeros(K, H, 3, dtype=L.dtype)
    for t in range(H):
        m = step == t
        if m.any(): G[:, t] = (L[m] - Hh[m][:, None]).mean(0)
    I = L - Hh[:, None] - G[:, step].transpose(0, 1)
    tot = ((L - L.mean((0, 1))) ** 2).sum()
    shares = dict(history=float(((Hh - Hh.mean(0)) ** 2).sum() * K / tot), goal=float(((G[:, step].transpose(0, 1) - G[:, step].transpose(0, 1).mean((0, 1))) ** 2).sum() / tot), interaction=float((I ** 2).sum() / tot))
    size = np.array([d["size"] for d in dec]); inter_by = {k: float((I[torch.as_tensor(m)] ** 2).sum() / ((L[torch.as_tensor(m)] - L.mean((0, 1))) ** 2).sum()) for k, m in (("1", size == 1), ("2-4", (size >= 2) & (size <= 4)), ("5-16", (size >= 5) & (size <= 16)), (">16", size > 16))}
    # the model's, additive and goal-blind actions against the optimal sets under every goal
    opts = torch.tensor(np.array([[d["opts"][g] for g in range(K)] for d in dec]))                        # [n, K, 3]
    acts = dict(model=L.argmax(-1), additive=(Hh[:, None] + G[:, step].transpose(0, 1)).argmax(-1), goal_blind=Hh.argmax(-1)[:, None].expand(-1, K), goal_blind_ceiling=None)
    offline = {k: float(opts.gather(2, v[..., None]).double().mean()) for k, v in acts.items() if v is not None}
    offline_by_size = {k: {b: float(opts[torch.as_tensor(m)].gather(2, v[torch.as_tensor(m)][..., None]).double().mean()) for b, m in (("1", size == 1), ("2-4", (size >= 2) & (size <= 4)), ("5-16", (size >= 5) & (size <= 16)), (">16", size > 16))} for k, v in acts.items() if v is not None}
    goal_dep = float((L.argmax(-1) != L.argmax(-1)[:, :1]).any(1).double().mean()); opt_goal_dep = float((opts != opts[:, :1]).any((1, 2)).double().mean())
    print(f"shares H {shares['history']:.3f} G {shares['goal']:.3f} I {shares['interaction']:.3f}; offline optimal: model {offline['model']:.3f} additive {offline['additive']:.3f} goal-blind {offline['goal_blind']:.3f} (ceiling {ceil['additive']:.3f} / {ceil['goal_blind']:.3f}); goal-dependent decisions model {goal_dep:.2f} optimum {opt_goal_dep:.2f} {time.time() - t0:.0f}s", flush=True)
    # online: the additive and goal-blind policies run in the environment
    class Additive:
        def __init__(self, blind): self.blind = blind
        def act(self, hist):
            z = torch.stack([policy.logits([(a_, o_, g) for a_, o_, _ in hist]) for g in range(K)], 1).mean(1)
            if not self.blind: z = z + G[torch.tensor([g for _, _, g in hist]), torch.tensor([min(len(a_), H - 1) for a_, _, _ in hist])]
            return z.argmax(1).tolist()
    online = {}
    for name, pol in (("model", policy), ("additive", Additive(False)), ("goal_blind", Additive(True))):
        rows = []
        for gi, (w, sol) in enumerate(zip(worlds, sols)):
            starts = test_starts(w, gi)[: a.n_online]; n = len(starts); e = BMZ.Env(w, n, torch.Generator().manual_seed(0), start=starts); ret = torch.zeros(n)
            for t in range(w.H):
                alive = torch.nonzero(~e.done).squeeze(1).tolist()
                if not alive: break
                ch = pol.act([([int(x[i]) for x in e.acts], [int(x[i]) for x in e.obss], gi) for i in alive])
                act = torch.zeros(n, dtype=torch.long); act[alive] = torch.tensor(ch); r, _ = e.step(act); ret += r
            rows.append(dict(ret=float(ret.mean()), success=float((ret > 0).float().mean())))
        online[name] = dict(per_goal=rows, ret=float(np.mean([r["ret"] for r in rows])), success=float(np.mean([r["success"] for r in rows])))
        print(f"online {name}: return {online[name]['ret']:.3f} success {online[name]['success']:.3f} {time.time() - t0:.0f}s", flush=True)
    rec = (online["additive"]["ret"] - online["goal_blind"]["ret"]) / max(online["model"]["ret"] - online["goal_blind"]["ret"], 1e-9)
    res = dict(model=a.model, run=a.run, n=len(dec), shares=shares, interaction_by_size=inter_by, offline=offline, offline_by_size=offline_by_size, goal_dependent=dict(model=goal_dep, optimum=opt_goal_dep), ceiling=ceil, online=online, recovery=rec,
               optimum_ret=float(np.mean([r["optimal"]["ret"] for r in design["per_goal"]])))
    print(f"recovery of the goal-directed return by the additive policy: {rec:.3f}")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
