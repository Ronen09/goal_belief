"""Reward training on the single-goal bump maze (PLAN.md): PPO on reward only for the small transformer and the GRU of
the single-goal experiment (an actor-critic head on the same trunk), one seed per process. Evaluation greedy on the
single-goal experiment's 2 048 fixed test starts. Writes runs/<kind>/seed<k>/{model.pt, log.json, train.log}.

    PYTHONPATH=/workspace/goal_belief .venv/bin/python studies/8_language_models/bump_reward/train_reward.py --kind tfm --seed 0
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from goalgeo import bumpmaze as BMZ
from goalgeo.mazeppo import gae

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze"))
from zero_shot import SEED, SIZE, LOOPS                     # noqa: E402
from train import StepTransformer, StepGRU, test_starts     # noqa: E402


class ActorCritic(nn.Module):
    """The single-goal experiment's StepTransformer or StepGRU with a value head on the same trunk. forward returns
    logits [N, L, 3] and values [N, L]; record=True adds the residuals entering each block and the final (transformer:
    layers + 1 tensors; GRU: the hidden states); patch: list of (layer, positions, tensor [N, d]) replacing the residual
    entering that layer at those positions (transformer only)."""
    def __init__(self, H, kind, d=128, layers=4, heads=4):
        super().__init__(); self.kind, self.H = kind, H
        self.body = StepTransformer(H, d, layers, heads) if kind == "tfm" else StepGRU(H, d); self.value = nn.Linear(d, 1)

    def forward(self, tokens, record=False, patch=None):
        b_ = self.body
        if self.kind == "gru":
            h, _ = b_.gru(b_.tok(tokens)); lg, v = b_.head(h), self.value(h).squeeze(-1)
            return (lg, v, [h]) if record else (lg, v)
        x = b_.tok(tokens) + b_.pos.weight[: tokens.shape[1]]; rec = []
        for l, blk in enumerate(b_.blocks):
            if patch:
                for (pl, pos, val) in patch:
                    if pl == l: x = x.clone(); x[torch.arange(x.shape[0], device=x.device), pos] = val.to(x.dtype)
            rec.append(x)
            a, _, _, _ = blk.attn(x); mid = x + a; x = mid + blk.mlp(blk.ln2(mid))
        if patch:
            for (pl, pos, val) in patch:
                if pl == len(b_.blocks): x = x.clone(); x[torch.arange(x.shape[0], device=x.device), pos] = val.to(x.dtype)
        rec.append(x); h = b_.ln(x); lg, v = b_.head(h), self.value(h).squeeze(-1)
        return (lg, v, rec) if record else (lg, v)


@torch.no_grad()
def rollout(model, w, N, gen, device, greedy=False, starts=None):
    """N episodes; tokens [N, H + 1] (0 start, 1 + 2a + o), per step: action, log-prob, value, reward (1 at the goal),
    alive. The task's discounted return is sum_s gamma^s r_s."""
    env = BMZ.Env(w, N, gen, device=device, start=starts); H = w.H
    tok = torch.zeros(N, H + 1, dtype=torch.long, device=device); z = lambda dt=torch.float32: torch.zeros(N, H, dtype=dt, device=device)
    act, logp, val, rew, alive = z(torch.long), z(), z(), z(), z(torch.bool)
    for s in range(H):
        live = ~env.done
        if not live.any(): break
        lg, v = model(tok[:, : s + 1]); lg, v = lg[:, s], v[:, s]
        a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
        logp[:, s], val[:, s], act[:, s], alive[:, s] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1), v, a, live
        r, o = env.step(a); rew[:, s] = (r > 0).float(); tok[:, s + 1] = torch.where(live, 1 + 2 * a + o, torch.zeros_like(a))
    return dict(tok=tok, act=act, logp=logp, val=val, rew=rew, alive=alive)


def summarize(b, gamma):
    H = b["rew"].shape[1]; disc = gamma ** torch.arange(H, device=b["rew"].device, dtype=torch.float32)
    ret = (b["rew"] * disc).sum(1); return dict(ret=float(ret.mean()), success=float((b["rew"].sum(1) > 0).float().mean()), length=float(b["alive"].sum(1).float().mean()))


def ppo_update(model, opt, b, gamma, lam, epochs, minibatches, clip, vf, ent_coef, max_grad):
    adv, target = gae(b["rew"], b["val"], b["alive"], gamma, lam); m = b["alive"].float(); cnt = m.sum().clamp(min=1)
    mean = (adv * m).sum() / cnt; std = (((adv - mean) ** 2 * m).sum() / cnt).sqrt(); adv = (adv - mean) / (std + 1e-8)
    N, H = b["act"].shape; stats = []
    for _ in range(epochs):
        for idx in torch.randperm(N, device=b["act"].device).chunk(minibatches):
            lg, v = model(b["tok"][idx]); lg, v = lg[:, :H], v[:, :H]; lp = torch.log_softmax(lg, -1); mm = m[idx]
            per = lambda x: (x * mm).sum() / mm.sum().clamp(min=1)
            ratio = (lp.gather(2, b["act"][idx][..., None]).squeeze(2) - b["logp"][idx]).exp(); a = adv[idx]
            pl = -per(torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)); vl = per((v - target[idx]) ** 2); ent = per(-(lp.exp() * lp).sum(-1))
            loss = pl + vf * vl - ent_coef * ent; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_grad); opt.step()
            stats.append((float(pl), float(vl), float(ent)))
    return np.mean(stats, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=["tfm", "gru"], default="tfm"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--n_env", type=int, default=4096); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03); ap.add_argument("--ent_final", type=float, default=0.003); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=50); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = Path(a.out) if a.out else HERE / "runs" / a.kind / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True); log = open(out / "train.log", "a")
    def P(*x):
        s = " ".join(str(y) for y in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    w = BMZ.make(SEED, SIZE, LOOPS); dev = a.device; torch.manual_seed(a.seed); model = ActorCritic(w.H, a.kind).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5); gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seed); egen = torch.Generator(device=dev)
    starts = test_starts(w).to(dev); rows = []; t0 = time.time(); inter = 0
    def evaluate(u):
        egen.manual_seed(5); model.eval(); r = summarize(rollout(model, w, len(starts), egen, dev, greedy=True, starts=starts), w.gamma); model.train()
        rows.append(dict(update=u, interactions=inter, seconds=time.time() - t0, **r)); (out / "log.json").write_text(json.dumps(dict(kind=a.kind, seed=a.seed, args=vars(a), log=rows), indent=1))
        P(f"u={u} {time.time() - t0:.0f}s greedy return {r['ret']:.3f} success {r['success']:.3f} length {r['length']:.1f}")
    for u in range(a.updates + 1):
        if u % a.eval_every == 0 or u == a.updates: evaluate(u)
        if u == a.updates: break
        for g_ in opt.param_groups: g_["lr"] = a.lr * min(1, (u + 1) / 50)
        b = rollout(model, w, a.n_env, gen, dev); inter += int(b["alive"].sum())
        pl, vl, ent = ppo_update(model, opt, b, w.gamma, 0.95, a.epochs, a.minibatches, 0.2, 0.5, a.ent + (a.ent_final - a.ent) * u / a.updates, 0.5)
        if u % a.eval_every == 0: P(f"  u={u} sampled return {summarize(b, w.gamma)['ret']:.3f} policy loss {pl:.3f} value loss {vl:.3f} entropy {ent:.3f}")
    torch.save(model.state_dict(), out / "model.pt"); P(f"saved {out / 'model.pt'} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
