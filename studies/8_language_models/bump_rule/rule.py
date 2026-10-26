"""What policy the reward-trained GRUs compute (the bump reward experiment's runs/gru/seed<k>/model.pt), against the
supervised GRU and the exact optimum put through the same analysis. No plan was registered: run on request after
the reward experiment's battery showed the policy is not a function of the belief set.

Collects 20 000 greedy episodes per model with the exact filter alongside, then
  1. memory: a lookup from the last k tokens (plus step / bumps / steps-since-bump features) to the greedy action, fitted
     on half the episodes and scored on the other half; token patches at distance d from the decision;
  2. the window rule run in the environment (backoff to shorter windows), return and success on the 2 048 test starts;
  3. an automaton from the hidden state: k-means with K clusters, action purity, transition determinism, run in the
     environment as a finite-state machine;
  4. the agent against the references on its own decisions: optimality, QMDP, always forward, forward-else-right-after-a-
     bump, the value given up at deviations and the information gain of its moves, localisation by step, openings.
Writes rule_<name>.json; `--tables` writes tables.md from the jsons.

    PYTHONPATH=/workspace/goal_belief .venv/bin/python studies/8_language_models/bump_rule/rule.py --name ppo_gru_seed2 --model ../bump_reward/runs/gru/seed2/model.pt
"""

from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent; S8 = HERE.parent
for d in ("bump_reward", "bump_policy", "bump_maze"): sys.path.insert(0, str(S8 / d))
from train_reward import ActorCritic, rollout, summarize          # noqa: E402
from battery import load_model                                     # noqa: E402
from common import world                                           # noqa: E402
from train import test_starts                                      # noqa: E402

TOKS = (1, 3, 5, 6)              # left, right, forward -> moved, forward -> bumped (1 + 2a + o)
N_EP, SPLIT_SEED = 20000, 7
WINDOWS = list(range(0, 13)) + [16, 20, 40]
KS = (2, 4, 8, 16, 32, 64, 128, 256)


def ent(p):
    p = np.asarray(p, float); p = p[p > 0]; return float(-(p * np.log2(p)).sum())


class Sets:
    """Per reachable set: Q*, V*, the optimal actions, size, P(bump | forward), P(goal | forward), the information gain of
    forward (the entropy of its outcome; a turn's is 0), QMDP's action."""
    def __init__(self, w, sol):
        self.w, self.sol = w, sol; self.Q = sol["Q"]; self.V = sol["V"]; self.opt = self.Q >= self.V[:, None] - 1e-9
        M = np.stack(sol["masks"]); n = M.sum(1).clip(min=1); goalahead = w.at_goal[w.nxt[:, 2]] & ~w.bump
        self.size = M.sum(1); self.pb = M[:, w.bump].sum(1) / n; self.pg = M[:, goalahead].sum(1) / n
        self.ig = np.array([ent([b, g, 1 - b - g]) for b, g in zip(self.pb, self.pg)])
        self.qmdp = ((M / n[:, None]) @ w.q_known).argmax(1)
        self.known = self.size == 1

    def index(self, masks):
        pk = np.packbits(masks.cpu().numpy(), axis=1); return np.array([self.sol["sets"].get(r.tobytes(), -1) for r in pk])


@torch.no_grad()
def collect(w, sets, policy, N, seed, dev, starts=None):
    """policy(tok [N, s + 1], env, idx [N], s) -> actions [N]. Flat arrays over live decisions: episode, step, set index,
    action, true state; the token matrix and the return per episode."""
    gen = torch.Generator(device=dev); gen.manual_seed(seed); env = BMZ.Env(w, N, gen, device=dev, start=starts); H = w.H
    tok = torch.zeros(N, H + 1, dtype=torch.long, device=dev); ret = torch.zeros(N, device=dev); rows = []
    for s in range(H):
        live = ~env.done
        if not live.any(): break
        idx = sets.index(env.filter.mask); a = policy(tok[:, : s + 1], env, idx, s); li = live.cpu().numpy()
        rows.append((np.nonzero(li)[0], np.full(int(li.sum()), s), idx[li], a[live].cpu().numpy(), env.s[live].cpu().numpy()))
        r, o = env.step(a); ret += r; tok[:, s + 1] = torch.where(live, 1 + 2 * a + o, torch.zeros_like(a))
    ep, st, si, ac, ss = (np.concatenate(x) for x in zip(*rows)); order = np.lexsort((st, ep))
    return dict(tok=tok.cpu(), ret=ret.cpu().numpy(), ep=ep[order], step=st[order], set=si[order], act=ac[order], state=ss[order])


def gru_policy(model, batch=4096):
    return lambda tok, env, idx, s: torch.cat([model(tok[b:b + batch])[0][:, s].argmax(-1) for b in range(0, tok.shape[0], batch)])


def opt_policy(sets, dev):
    return lambda tok, env, idx, s: torch.as_tensor(sets.Q[np.maximum(idx, 0)].argmax(1), device=dev)


@torch.no_grad()
def hidden(model, tok, ep, step, dev, batch=4096):
    out = []
    for b in range(0, tok.shape[0], batch):
        _, _, (h,) = model(tok[b:b + batch].to(dev), record=True); out.append(h.cpu())
    return torch.cat(out)[ep, step].float()


# ------------------------------------------------------------------ 1. memory

def features(D, H):
    """Per decision: step, bumps so far, steps since the last bump (step + 1 if none), forward moves so far."""
    T = D["tok"].numpy(); ep, st = D["ep"], D["step"]; n = len(ep)
    bumps = np.zeros(n, dtype=np.int64); since = np.zeros(n, dtype=np.int64); moved = np.zeros(n, dtype=np.int64)
    cb = np.cumsum(T == 6, 1); cm = np.cumsum(T == 5, 1)                                   # cumulative over positions 0..p
    bumps = cb[ep, st]; moved = cm[ep, st]                                                 # tokens at positions 1..step
    last = np.full(T.shape, -1); pos = np.arange(T.shape[1])[None].repeat(T.shape[0], 0); last = np.where(T == 6, pos, -1); last = np.maximum.accumulate(last, 1)
    lb = last[ep, st]; since = np.where(lb >= 0, st + 1 - lb, st + 1)
    return dict(step=st, bumps=bumps, since=since, moved=moved)


def window_keys(D, k, extra=None):
    """Rows [n, k (+1)]: the last k tokens before the decision (positions step - k + 1 .. step; -1 before the start)."""
    T = D["tok"].numpy(); ep, st = D["ep"], D["step"]; cols = []
    for j in range(k):
        p = st - j; cols.append(np.where(p >= 0, T[ep, np.maximum(p, 0)], -1))
    if extra is not None: cols.append(extra)
    R = np.stack(cols, 1) if cols else np.zeros((len(ep), 0), dtype=np.int64); return np.ascontiguousarray(R.astype(np.int64))


class Lookup:
    """Majority action per key with backoff to shorter windows (dropping the oldest token; the extra feature kept)."""
    def __init__(self, rows, act, k, has_extra):
        self.k, self.has_extra = k, has_extra; self.levels = []
        for kk in range(k, -1, -1):
            sub = self._sub(rows, kk); u, inv = np.unique(sub, axis=0, return_inverse=True); inv = inv.ravel()
            cnt = np.zeros((len(u), 3), dtype=np.int64); np.add.at(cnt, (inv, act), 1); self.levels.append((kk, u, cnt.argmax(1)))
        self.n_keys = len(self.levels[0][1])

    def _sub(self, rows, kk):
        cols = list(range(kk)) + ([self.k] if self.has_extra else []); return np.ascontiguousarray(rows[:, cols])

    def predict(self, rows):
        n = len(rows); out = np.full(n, -1); level = np.full(n, -1)
        for kk, u, maj in self.levels:
            todo = np.nonzero(out < 0)[0]
            if len(todo) == 0: break
            sub = self._sub(rows[todo], kk); U, inv = np.unique(np.concatenate([u, sub]), axis=0, return_inverse=True); inv = inv.ravel()
            m = np.full(len(U), -1); m[inv[:len(u)]] = maj; pred = m[inv[len(u):]]; hit = pred >= 0
            out[todo[hit]] = pred[hit]; level[todo[hit]] = kk
        return out, level


def memory(D, F, train, test, name):
    """Agreement of window lookups with the greedy action on the held-out episodes, by window and feature."""
    res = {}; act = D["act"]
    for feat in (None, "step", "bumps", "since", "moved"):
        row = {}
        for k in WINDOWS:
            rows = window_keys(D, k, None if feat is None else F[feat]); L = Lookup(rows[train], act[train], k, feat is not None)
            pred, level = L.predict(rows[test]); row[k] = dict(agree=float((pred == act[test]).mean()), exact=float((level == k).mean()), keys=L.n_keys)
            print(f"  {name} window {k:2d} {feat or '-':6s}: agree {row[k]['agree']:.3f} (exact key {row[k]['exact']:.3f}, {row[k]['keys']} keys)", flush=True)
        res[feat or "window"] = row
    return res


@torch.no_grad()
def token_patch(model, D, test, dev, n=20000, dmax=16, seed=0):
    """Replace the token at distance d from the decision (1 = the latest) with a random other token; the share of greedy
    decisions that change, on held-out decisions with step >= d."""
    rng = np.random.default_rng(seed); T = D["tok"]; ep, st, act = D["ep"][test], D["step"][test], D["act"][test]; out = {}
    for d in range(1, dmax + 1):
        ok = np.nonzero(st >= d)[0]
        if len(ok) < 100: break
        pick = rng.choice(ok, min(n, len(ok)), replace=False); rows = T[ep[pick]].clone(); pos = st[pick] - d + 1; orig = rows[np.arange(len(pick)), pos]
        new = torch.as_tensor([TOKS[rng.integers(4)] for _ in pick]); same = new == orig
        while same.any(): new[same] = torch.as_tensor([TOKS[rng.integers(4)] for _ in range(int(same.sum()))]); same = new == orig
        rows[np.arange(len(pick)), pos] = new; lg = torch.cat([model(rows[b:b + 4096].to(dev))[0].cpu() for b in range(0, len(pick), 4096)]); a2 = lg[np.arange(len(pick)), st[pick]].argmax(-1).numpy()
        out[d] = dict(changed=float((a2 != act[pick]).mean()), n=len(pick))
    print("  token patch: " + " ".join(f"d{d} {v['changed']:.2f}" for d, v in out.items()), flush=True); return out


# ------------------------------------------------------------------ 2. the window rule in the environment

def run_lookup_policy(w, sets, L, k, feat, dev, starts):
    def pol(tok, env, idx, s):
        T = tok.cpu().numpy(); n = T.shape[0]; cols = [np.where(s - j >= 0, T[:, max(s - j, 0)], -1) for j in range(k)]
        if feat is not None:
            if feat == "step": cols.append(np.full(n, s))
            elif feat == "bumps": cols.append((T[:, 1:s + 1] == 6).sum(1))
            elif feat == "moved": cols.append((T[:, 1:s + 1] == 5).sum(1))
            elif feat == "since":
                last = np.where(T[:, :s + 1] == 6, np.arange(s + 1)[None], -1).max(1); cols.append(np.where(last >= 0, s + 1 - last, s + 1))
        rows = np.ascontiguousarray(np.stack(cols, 1).astype(np.int64)) if cols else np.zeros((n, 0), dtype=np.int64); pred, _ = L.predict(rows); return torch.as_tensor(pred, device=dev)
    D = collect(w, sets, pol, len(starts), 5, dev, starts); return dict(ret=float(D["ret"].mean()), success=float((D["ret"] > 0).mean()), length=float(np.bincount(D["ep"]).mean()))


# ------------------------------------------------------------------ 3. the automaton from the hidden state

def kmeans(X, K, iters=40, seed=0):
    g = torch.Generator(device=X.device); g.manual_seed(seed); C = X[torch.randperm(len(X), generator=g, device=X.device)[:K]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1); newC = torch.zeros_like(C); newC.index_add_(0, a, X); cnt = torch.bincount(a, minlength=K).float()
        empty = cnt == 0; C = torch.where(empty[:, None], C, newC / cnt.clamp(min=1)[:, None])
    return C, torch.cdist(X, C).argmin(1)


def automaton(w, sets, Hs, D, train, test, dev, starts, name):
    """For each K: action purity on held-out decisions, transition determinism, labels, and the machine run in the environment."""
    T = D["tok"].numpy(); ep, st, act, si = D["ep"], D["step"], D["act"], D["set"]; X = Hs.to(dev); res = {}
    nxt_tok = T[ep, st + 1]                                                      # the token produced by this decision
    nxt_i = np.arange(len(ep)) + 1; has_next = np.zeros(len(ep), dtype=bool); has_next[:-1] = (ep[1:] == ep[:-1])
    for K in KS:
        C, lab = kmeans(X[torch.as_tensor(train, device=dev)], K); lab_all = torch.cdist(X, C).argmin(1).cpu().numpy(); lt = lab_all[train]
        cnt = np.zeros((K, 3), dtype=np.int64); np.add.at(cnt, (lt, act[train]), 1); maj = cnt.argmax(1); purity_test = float((maj[lab_all[test]] == act[test]).mean())
        tr = np.nonzero(has_next)[0]; trt = tr[np.isin(tr, train)]; trs = tr[np.isin(tr, test)]
        tab = np.zeros((K, 7, K), dtype=np.int64); np.add.at(tab, (lab_all[trt], nxt_tok[trt], lab_all[trt + 1]), 1); nxt_maj = tab.argmax(2); seen = tab.sum(2) > 0
        det_train = float((tab.max(2).sum() / max(tab.sum(), 1))); det_test = float((nxt_maj[lab_all[trs], nxt_tok[trs]] == lab_all[trs + 1]).mean())
        # labels: per cluster, mean log2 set size, share known, mean P(bump), share of decisions optimal
        labels = {int(c): dict(n=int((lt == c).sum()), action=int(maj[c]), log_size=float(np.log2(sets.size[si[train][lt == c]]).mean()) if (lt == c).any() else None,
                               known=float(sets.known[si[train][lt == c]].mean()) if (lt == c).any() else None, step=float(st[train][lt == c].mean()) if (lt == c).any() else None) for c in range(K)}
        # run the machine: start at the cluster of the step-0 state
        c0 = int(lab_all[np.nonzero(st == 0)[0][0]]); fallback = tab.sum(1).argmax(1)
        def pol(tok, env, idx, s, state={}):
            if s == 0: state["c"] = np.full(tok.shape[0], c0)
            else:
                t = tok[:, s].cpu().numpy(); c = state["c"]; nc = np.where(seen[c, t], nxt_maj[c, t], fallback[c]); state["c"] = nc
            return torch.as_tensor(maj[state["c"]], device=dev)
        E = collect(w, sets, pol, len(starts), 5, dev, starts); run = dict(ret=float(E["ret"].mean()), success=float((E["ret"] > 0).mean()), length=float(np.bincount(E["ep"]).mean()))
        res[K] = dict(purity=purity_test, det_train=det_train, det_test=det_test, run=run, labels=labels, used=int((cnt.sum(1) > 0).sum()))
        print(f"  {name} K={K:3d}: action purity {purity_test:.3f}, transition determinism {det_train:.3f} / {det_test:.3f}, machine return {run['ret']:.3f} success {run['success']:.3f}", flush=True)
    return res


# ------------------------------------------------------------------ 4. references, deviations, information, openings

def references(w, sets, D, F, name):
    act, si, st = D["act"], D["set"], D["step"]; T = D["tok"].numpy(); n = len(act); ar = np.arange(n)
    last = np.where(st >= 1, T[D["ep"], np.maximum(st, 0)], -1)                # the latest token (position step); the start token 0 at step 0
    rule_fr = np.where(last == 6, 1, 2)                                        # forward, right after a bump
    optimal = sets.opt[si, act]; qmdp_eq = act == sets.qmdp[si]
    given = (sets.V[si] - sets.Q[si, act]) / np.maximum(sets.V[si], 1e-9)
    ig_a = np.where(act == 2, sets.ig[si], 0.0); ig_opt = np.where(sets.opt[si, 2], sets.ig[si], 0.0); ig_max = sets.ig[si]
    dev_ = ~optimal
    res = dict(decisions=n, optimal=float(optimal.mean()), optimal_known=float(optimal[sets.known[si]].mean()), optimal_unknown=float(optimal[~sets.known[si]].mean()),
               known_share=float(sets.known[si].mean()), qmdp=float(qmdp_eq.mean()), forward=float((act == 2).mean()), forward_right_after_bump=float((act == rule_fr).mean()),
               action_freq=np.bincount(act, minlength=3).astype(float).__truediv__(n).tolist(),
               deviations=dict(share=float(dev_.mean()), value_given_up=float(given[dev_].mean()) if dev_.any() else None, ig_minus_opt=float((ig_a - ig_opt)[dev_].mean()) if dev_.any() else None,
                               more_informative=float((ig_a > ig_opt + 1e-9)[dev_].mean()) if dev_.any() else None, less_informative=float((ig_a < ig_opt - 1e-9)[dev_].mean()) if dev_.any() else None,
                               turn_where_forward_optimal=float(((act < 2) & sets.opt[si, 2])[dev_].mean()) if dev_.any() else None, forward_where_turn_optimal=float(((act == 2) & ~sets.opt[si, 2])[dev_].mean()) if dev_.any() else None),
               ig_mean=float(ig_a.mean()), ig_mean_opt_action=float(ig_opt.mean()), ig_mean_max=float(ig_max.mean()))
    # by step: alive share, mean log2 set size, share known, share optimal, action frequencies
    by = []
    for s in range(w.H):
        m = st == s
        if not m.any(): break
        by.append(dict(step=s, alive=float(m.sum() / N_EP), log_size=float(np.log2(sets.size[si[m]]).mean()), known=float(sets.known[si[m]].mean()), optimal=float(optimal[m].mean()), freq=(np.bincount(act[m], minlength=3) / m.sum()).tolist()))
    res["by_step"] = by
    # openings: the first 8 actions of each episode
    firsts = {}
    for i, (e, s, a) in enumerate(zip(D["ep"], st, act)):
        if s < 8: firsts.setdefault(int(e), []).append("LRF"[a])
    c = Counter("".join(v) for v in firsts.values()); res["openings"] = [(k, v / N_EP) for k, v in c.most_common(8)]; res["distinct_openings"] = len(c)
    # after a bump: the next action(s)
    bump_dec = (last == 6); res["after_bump_freq"] = (np.bincount(act[bump_dec], minlength=3) / max(bump_dec.sum(), 1)).tolist()
    moved_dec = (last == 5); res["after_moved_freq"] = (np.bincount(act[moved_dec], minlength=3) / max(moved_dec.sum(), 1)).tolist()
    for lt, lab in ((1, "after_left_freq"), (3, "after_right_freq")): m = last == lt; res[lab] = (np.bincount(act[m], minlength=3) / max(m.sum(), 1)).tolist()
    # the two-token context: action after each pair of last two tokens
    two = {}
    T2 = np.where(st >= 2, T[D["ep"], np.maximum(st - 1, 0)], -1)
    for a_ in (-1, 1, 3, 5, 6):
        for b_ in (1, 3, 5, 6):
            m = (T2 == a_) & (last == b_)
            if m.sum() >= 200: two[f"{a_},{b_}"] = dict(n=int(m.sum()), freq=(np.bincount(act[m], minlength=3) / m.sum()).tolist())
    res["two_token"] = two
    print(f"  {name}: optimal {res['optimal']:.3f} (known {res['optimal_known']:.3f}, unknown {res['optimal_unknown']:.3f}); = QMDP {res['qmdp']:.3f}; forward {res['forward']:.3f}; forward-else-right-after-bump {res['forward_right_after_bump']:.3f}; "
          f"deviations {res['deviations']['share']:.3f}: value given up {res['deviations']['value_given_up']}, IG − opt {res['deviations']['ig_minus_opt']}; openings {res['openings'][:3]}", flush=True)
    return res


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--model", default=None); ap.add_argument("--kind", default="gru"); ap.add_argument("--device", default="cuda"); ap.add_argument("--tables", action="store_true"); a = ap.parse_args()
    if a.tables: return tables()
    t0 = time.time(); w, sol = world(); sets = Sets(w, sol); dev = a.device; H = w.H; starts = test_starts(w).to(dev)
    model = None
    if a.model:
        mp = Path(a.model); mp = mp if mp.exists() else (HERE / a.model).resolve(); model = load_model(a.kind, mp, H, dev); pol = gru_policy(model)
    else: pol = opt_policy(sets, dev)
    D = collect(w, sets, pol, N_EP, SPLIT_SEED, dev); n = len(D["act"]); F = features(D, H)
    online = dict(ret=float(D["ret"].mean()), success=float((D["ret"] > 0).mean()), length=float(np.bincount(D["ep"]).mean()))
    print(f"{a.name}: {N_EP} greedy episodes, {n} decisions, return {online['ret']:.3f} success {online['success']:.3f} length {online['length']:.1f} {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0); ep_train = rng.random(N_EP) < 0.5; train = np.nonzero(ep_train[D["ep"]])[0]; test = np.nonzero(~ep_train[D["ep"]])[0]
    res = dict(name=a.name, online=online, decisions=n, distinct_sets=int(len(np.unique(D["set"]))))
    res["references"] = references(w, sets, D, F, a.name)
    res["memory"] = memory(D, F, train, test, a.name)
    if model is not None: res["token_patch"] = token_patch(model, D, test, dev)
    # the window rule in the environment: window only and window + since-bump, k = 1..8, 12, 20
    env_runs = {}
    for feat in (None, "since", "bumps"):
        for k in (0, 1, 2, 3, 4, 6, 8, 12, 20, 40):
            rows = window_keys(D, k, None if feat is None else F[feat]); L = Lookup(rows[train], D["act"][train], k, feat is not None)
            r = run_lookup_policy(w, sets, L, k, feat, dev, starts); env_runs[f"{feat or 'window'}:{k}"] = r
            print(f"  {a.name} rule window {k:2d} {feat or '-':6s} in the environment: return {r['ret']:.3f} success {r['success']:.3f} length {r['length']:.1f} {time.time() - t0:.0f}s", flush=True)
    res["env_runs"] = env_runs
    if model is not None:
        Hs = hidden(model, D["tok"], D["ep"], D["step"], dev); res["automaton"] = automaton(w, sets, Hs, D, train, test, dev, starts, a.name)
    (HERE / f"rule_{a.name}.json").write_text(json.dumps(res, indent=1)); print(f"wrote rule_{a.name}.json {time.time() - t0:.0f}s")


def tables():
    names = ["optimum", "sup_gru_100000"] + [f"ppo_gru_seed{s}" for s in range(5)]; R = {n: json.loads((HERE / f"rule_{n}.json").read_text()) for n in names if (HERE / f"rule_{n}.json").exists()}
    L = ["# Bump rule: tables", "", "Generated by `rule.py --tables`. 20 000 greedy episodes per model from random starts; lookups fitted on half the episodes and scored on the other half; rules run in the environment on the 2 048 fixed test starts.", ""]
    L += ["## Online, and the agent on its own decisions against the references", "", "| model | return | success | length | distinct sets | optimal | at known states | at unknown | known share | = QMDP | forward | forward, right after a bump | deviations | value given up at deviations | IG − optimum's IG at deviations | more / less informative at deviations | mean IG of its move / of an optimal / max |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in R.items():
        o, f, d = r["online"], r["references"], r["references"]["deviations"]
        L.append(f"| {n} | {o['ret']:.3f} | {o['success']:.3f} | {o['length']:.1f} | {r['distinct_sets']} | {f['optimal']:.3f} | {f['optimal_known']:.3f} | {f['optimal_unknown']:.3f} | {f['known_share']:.3f} | {f['qmdp']:.3f} | {f['forward']:.3f} | {f['forward_right_after_bump']:.3f} | {d['share']:.3f} | {fmt(d['value_given_up'])} | {fmt(d['ig_minus_opt'])} | {fmt(d['more_informative'])} / {fmt(d['less_informative'])} | {f['ig_mean']:.3f} / {f['ig_mean_opt_action']:.3f} / {f['ig_mean_max']:.3f} |")
    L += ["", "## Openings (the first 8 actions; share of episodes) and the action after the latest token", "", "| model | distinct openings | top openings | after a bump: L / R / F | after moved | after left | after right |", "|---|---|---|---|---|---|---|"]
    for n, r in R.items():
        f = r["references"]; L.append(f"| {n} | {f['distinct_openings']} | " + ", ".join(f"{k} {v:.2f}" for k, v in f["openings"][:5]) + f" | {fr(f['after_bump_freq'])} | {fr(f['after_moved_freq'])} | {fr(f['after_left_freq'])} | {fr(f['after_right_freq'])} |")
    L += ["", "## Localisation by step (mean log2 set size among alive episodes; share with the state known; share alive)", "", "| model | " + " | ".join(f"step {s}" for s in (0, 2, 4, 6, 8, 10, 15, 20, 30)) + " |", "|---|" + "---|" * 9]
    for n, r in R.items():
        by = {b["step"]: b for b in r["references"]["by_step"]}; L.append(f"| {n} | " + " | ".join(f"{by[s]['log_size']:.2f} / {by[s]['known']:.2f} / {by[s]['alive']:.2f}" if s in by else "–" for s in (0, 2, 4, 6, 8, 10, 15, 20, 30)) + " |")
    L += ["", "## Memory: agreement of a lookup from the last k tokens with the greedy action on held-out episodes (exact-key share in brackets)", ""]
    for feat, lab in (("window", "window only"), ("step", "+ step"), ("bumps", "+ bumps so far"), ("since", "+ steps since the last bump"), ("moved", "+ forward moves so far")):
        L += [f"### {lab}", "", "| model | " + " | ".join(f"k={k}" for k in WINDOWS) + " |", "|---|" + "---|" * len(WINDOWS)]
        for n, r in R.items():
            m = r["memory"][feat]; L.append(f"| {n} | " + " | ".join(f"{m[str(k)]['agree']:.3f} ({m[str(k)]['exact']:.2f})" for k in WINDOWS) + " |")
        L.append("")
    L += ["## Token patches: share of held-out greedy decisions changed by replacing the token at distance d (1 = the latest) with a random other token", "", "| model | " + " | ".join(f"d={d}" for d in range(1, 17)) + " |", "|---|" + "---|" * 16]
    for n, r in R.items():
        if "token_patch" in r: tp = r["token_patch"]; L.append(f"| {n} | " + " | ".join(f"{tp[str(d)]['changed']:.2f}" if str(d) in tp else "–" for d in range(1, 17)) + " |")
    ks = (0, 1, 2, 3, 4, 6, 8, 12, 20, 40)
    L += ["", "## The window rule run in the environment (return / success on the test starts)", ""]
    for feat, lab in (("window", "window only"), ("since", "+ steps since the last bump"), ("bumps", "+ bumps so far")):
        L += [f"### {lab}", "", "| model | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for n, r in R.items():
            e = r["env_runs"]; L.append(f"| {n} | " + " | ".join(f"{e[f'{feat}:{k}']['ret']:.3f} / {e[f'{feat}:{k}']['success']:.2f}" for k in ks) + " |")
        L.append("")
    L += ["## The automaton from the hidden state (K clusters): action purity on held-out decisions, transition determinism (fit / held-out), the machine's return / success in the environment", "", "| model | " + " | ".join(f"K={K}" for K in KS) + " |", "|---|" + "---|" * len(KS)]
    for n, r in R.items():
        if "automaton" in r: A = r["automaton"]; L.append(f"| {n} | " + " | ".join(f"{A[str(K)]['purity']:.2f}, {A[str(K)]['det_train']:.2f} / {A[str(K)]['det_test']:.2f}, {A[str(K)]['run']['ret']:.3f} / {A[str(K)]['run']['success']:.2f}" for K in KS) + " |")
    Dv = {n: json.loads((HERE / f"deviations_{n}.json").read_text()) for n in names if (HERE / f"deviations_{n}.json").exists()}
    if Dv:
        L += ["", "## Where the deviations are (`deviations.py`): per set the agent's majority action; systematic = at a set whose majority action is wrong, occasional = at a set whose majority is optimal; mixed = sets with ≥ 20 decisions and more than one action, not a tie = one of them suboptimal", "",
              "| model | deviations | systematic / occasional | sets wrong by majority (of visited) | sets carrying half / 90 % of deviations | mixed sets, not a tie (share of decisions) | known states: share of decisions / deviation share / states wrong by majority | wrong known states (cell x,y heading: agent vs optimal, cost in moves) |", "|---|---|---|---|---|---|---|---|"]
        for n, d in Dv.items():
            k = d["known"]; mv = lambda g: {0.03: 1, 0.06: 2, 0.11: 4}.get(round(g, 2), round(np.log(1 - g) / np.log(0.97)) if g < 1 else "-")
            L.append(f"| {n} | {d['deviation_share']:.3f} | {d['systematic_share']:.2f} / {d['occasional_share']:.2f} | {d['sets_with_wrong_majority']} of {d['sets_visited']} | {d['sets_for_half_of_deviations']} / {d['sets_for_90pct']} | {d['big_sets_mixed_not_tie']} ({d['decisions_in_mixed_not_tie']:.3f}) | {k['decisions'] / d['decisions']:.2f} / {k['deviation_share']:.3f} / {len(d['known_wrong'])} | " + "; ".join(f"({r['cell'][0]},{r['cell'][1]}) {r['heading'][0].upper()}: {r['agent']} vs {r['optimal']}, {mv(r['gap'])}" for r in d["known_wrong"][:8]) + " |")
        L += ["", "Maps: cells with a known state whose majority move is wrong, by model", ""]
        for n, d in Dv.items(): L += [f"{n}:", "```", d["map"], "```", ""]
    (HERE / "tables.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def fmt(x): return "–" if x is None else f"{x:.3f}"
def fr(f): return " / ".join(f"{x:.2f}" for x in f)


if __name__ == "__main__":
    main()
