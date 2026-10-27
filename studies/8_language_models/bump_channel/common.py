"""Shared pieces of the bump channel study (PLAN.md): the sites and registered ranks, same-step history pairs with
different goal-free sets, the registered rank from a list of pooled causal ranks, the goal / history / readout bases
and the projectors built from them, through ratios with cluster bootstraps and the known / uncertain split; (part 2)
the from-scratch four-goal models with recording, patching and Jacobians at the decision position; (part 3) the
fine-tuned model with patched recording and Jacobians, and the coordinates study's decision pool and states.
Scripts: subspace_small.py, channel.py, tables.py.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_channel .venv/bin/python studies/8_language_models/bump_channel/<script>.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
COORDS, GOALS_DIR, MAZE_DIR = HERE.parent / "bump_coords", HERE.parent / "bump_goals", HERE.parent / "bump_maze"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


CC = _load_by_path("coords_common", COORDS / "common.py")            # bump_coords/common.py, which is also called common.py

SITES_LLM = (15, 19, 21, 23)
K_LLM = {15: 128, 19: 64, 21: 8, 23: 4}                                # PLAN.md's registered ranks: at 23 and 21 the median pooled causal rank of bump_coords/rank.json rounded up (4, 6 → 8); at 19 and 15 the plan's numbers (the per-pair medians; the pooled medians there are 128 and 128)
KS_LLM = (1, 2, 4, 8, 16, 32, 64, 128, 896)
KS_SMALL = (1, 2, 4, 8, 16, 32, 64, 128)
SITES_TFM = (1, 2, 3, 4)                                                # residual entering block l at the decision position; 4 = the final residual (entering the head's layer norm)
SITES_GRU = (-2, -1, 0)                                                 # the hidden state at position t + s (0 = at the decision; below position 0: the zero initial state)
N_PAIRS, D_SMALL, FLOOR = 512, 128, 0.10
K, H = CC.K, CC.H


# ------------------------------------------------------------------ pairs and ranks
def set_key(w, acts, obss):
    """The goal-free belief after a history, as bytes."""
    from goalgeo import bumpmaze as BMZ
    return np.packbits(BMZ.common_mask(w, acts, obss)).tobytes()


def draw_pairs(steps, keys, n_pairs=N_PAIRS, seed=1):
    """Same-step pairs (i, j) with different keys, drawn as the goal-swap study draws its pairs: i uniform, j uniform
    among i's step; histories may repeat across pairs."""
    steps = np.asarray(steps); rng = np.random.default_rng(seed); by_step = {}
    for i, s in enumerate(steps): by_step.setdefault(int(s), []).append(i)
    pairs, tries = [], 0
    while len(pairs) < n_pairs:
        i = int(rng.integers(len(steps))); c = by_step[int(steps[i])]; j = c[int(rng.integers(len(c)))]; tries += 1
        if j != i and keys[j] != keys[i]: pairs.append((i, j))
        if tries > 1_000_000: raise RuntimeError("no same-step pairs with different sets")
    return pairs


def registered_k(kc_list, ks):
    """The median of the defined pooled causal ranks, rounded up to the grid; None if the list is empty, if more than
    six entries are undefined, or if none is defined."""
    kc_list = list(kc_list); defined = [k for k in kc_list if k is not None]
    if not kc_list or len(kc_list) - len(defined) > 6 or not defined: return None
    m = float(np.median(defined))
    for k in ks:
        if k >= m: return int(k)
    return int(ks[-1])


# ------------------------------------------------------------------ bases and projectors
def _pairs_of(k=K):
    return [(A, B) for A in range(k) for B in range(k) if A != B]


def goal_basis(X_fit, steps_fit):
    """The pooled per-step-centred goal differences of all ordered pairs over the fit histories: a full orthonormal
    basis [D, D] (singular value descending) and the pooled spectrum's ranks."""
    pooled = torch.cat([CC._centre(torch.as_tensor(X_fit[B]).double() - torch.as_tensor(X_fit[A]).double(), steps_fit) for (A, B) in _pairs_of()])
    return CC.basis(pooled, None), CC.spectrum_ranks(pooled, None)


def history_basis(X_fit, steps_fit, keys_fit, n_pairs=N_PAIRS, seed=2):
    """The belief effect's own principal directions: same-step fit pairs with different sets, x(h2, g) − x(h1, g) per
    goal, per-step centred, pooled over goals."""
    pairs = draw_pairs(steps_fit, keys_fit, n_pairs, seed); i = torch.tensor([p[0] for p in pairs]); j = torch.tensor([p[1] for p in pairs]); st = torch.as_tensor(steps_fit)[i]
    pooled = torch.cat([CC._centre(torch.as_tensor(X_fit[g]).double()[j] - torch.as_tensor(X_fit[g]).double()[i], st) for g in range(K)])
    return CC.basis(pooled, None)


def readout_basis(J):
    """J [n, 3, D], the Jacobian of the three centred logits: the basis of the stacked centred rows (a weight matrix,
    not centred as data)."""
    J = torch.as_tensor(J).double(); rows = (J - J.mean(1, keepdim=True)).reshape(-1, J.shape[-1]); return CC.basis(rows, None, centre=False)


def projector(Vh, k):
    return CC.projector(Vh, slice(0, k))


def projectors(Vh, k, d, seed=0):
    """goal (top-k of Vh), comp (its complement), bottom (the last k directions), random0..2 (k random directions)."""
    Pg = projector(Vh, k)
    return dict(goal=Pg, comp=torch.eye(d, dtype=torch.float64) - Pg, bottom=CC.projector(Vh, slice(d - k, None)), **{f"random{i}": CC.random_projector(d, k, seed + i) for i in range(3)})


def candidate(x1, delta, P):
    """x1 + δ P: the change restricted to P's subspace."""
    return torch.as_tensor(x1).double() + torch.as_tensor(delta).double() @ torch.as_tensor(P).double()


def norm_share(delta, P):
    """Mean over rows of ‖δ P‖² / ‖δ‖²."""
    delta = torch.as_tensor(delta).double(); proj = delta @ torch.as_tensor(P).double()
    return float(((proj ** 2).sum(1) / (delta ** 2).sum(1).clamp(min=1e-12)).mean())


def overlap(P1, P2, k):
    return CC.subspace_overlap(P1, P2, k)


# ------------------------------------------------------------------ bins and caches
def dep_share_by_bin(known_pair, cells_p, k=K):
    """The share of a bin's cells (its pairs × k goals) that are history-dependent: known (set size 1) and uncertain."""
    known_pair, cells_p = np.asarray(known_pair, dtype=bool), np.asarray(cells_p); n = dict(known=int(known_pair.sum()), uncertain=int((~known_pair).sum()))
    share = lambda m, nb: float(m.sum() / (k * nb)) if nb else float("nan")
    return dict(known=share(known_pair[cells_p], n["known"]), uncertain=share(~known_pair[cells_p], n["uncertain"]), n_pairs=n)


def resolve_cache(kind, quick, cache_arg, out):
    """The cache holding states_<kind>.pt / subspaces_<kind>.pt: the --cache argument; in quick mode the smoke
    layouts (reproduce.py's _smoke/subspace_small/cache, then the manual _smoke/subspace_<kind>/cache) when they hold
    the kind's states; otherwise this study's cache."""
    if cache_arg: return Path(cache_arg)
    if quick:
        for c in (Path(out).parent / "subspace_small" / "cache", Path(out).parent / f"subspace_{kind}" / "cache"):
            if (c / f"states_{kind}.pt").exists(): return c
    return CACHE


# ------------------------------------------------------------------ through ratios
def through(per_var, per_ref, groups):
    """Per-cell 0/1 switches of a variant and of its reference (same cells, same order), clusters in groups: the
    variant's switch with its interval, the reference's switch, and the through ratio mean(var) / mean(ref) with its
    paired cluster-bootstrap interval; the ratio is None where the reference switches below FLOOR or there are no cells."""
    per_var, per_ref = np.asarray(per_var, dtype=float), np.asarray(per_ref, dtype=float); groups = np.asarray(groups)
    if len(per_var) == 0: return dict(switch=float("nan"), switch_ci=[float("nan"), float("nan")], ref=float("nan"), ratio=None, ratio_ci=None, n=0)
    ref = float(per_ref.mean()); out = dict(switch=float(per_var.mean()), switch_ci=CC.boot_mean(per_var, groups=groups), ref=ref, n=int(len(per_var)))
    if ref < FLOOR: out.update(ratio=None, ratio_ci=None)
    else: out.update(ratio=float(per_var.mean() / ref), ratio_ci=CC.boot_ratio(per_var, per_ref, groups=groups))
    return out


def split_known(per_var, per_ref, groups, known):
    """through() on the known (set size 1) and uncertain cells, and the cluster-bootstrap interval of the difference
    of the two ratios (None where either is undefined)."""
    per_var, per_ref, groups, known = (np.asarray(a) for a in (per_var, per_ref, groups, known)); known = known.astype(bool)
    kn, un = through(per_var[known], per_ref[known], groups[known]), through(per_var[~known], per_ref[~known], groups[~known])
    out = dict(known=kn, uncertain=un, diff_ci=None)
    if kn["ratio"] is not None and un["ratio"] is not None:
        rng = np.random.default_rng(0); vk, rk, gk = per_var[known].astype(float), per_ref[known].astype(float), groups[known]; vu, ru, gu = per_var[~known].astype(float), per_ref[~known].astype(float), groups[~known]
        b = []
        for ik, iu in zip(CC._resamples(len(vk), gk, 1000, rng), CC._resamples(len(vu), gu, 1000, rng)):
            dk, du = rk[ik].mean(), ru[iu].mean()
            if dk > 0 and du > 0: b.append(vk[ik].mean() / dk - vu[iu].mean() / du)
        if b: out["diff_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


# ------------------------------------------------------------------ part 2: the from-scratch four-goal models
def small_tokens(hists, g, H=H):
    """[n, H + 1] tokens as bump_goals/train_goals.step_tokens builds them: the goal token 7 + g, then 1 + 2a + o per
    executed step, 0 beyond; and the decision positions [n] (= the number of steps)."""
    n = len(hists); t = torch.zeros(n, H + 1, dtype=torch.long); ln = torch.tensor([len(a) for a, _ in hists]); t[:, 0] = 7 + g
    for i, (a, o) in enumerate(hists):
        if len(a): t[i, 1:1 + len(a)] = 1 + 2 * torch.tensor(a) + torch.tensor(o)
    return t, ln


class Small:
    """A from-scratch four-goal model (bump_maze/train.py's StepTransformer or StepGRU with the goal as the start
    token) with recording, patching and Jacobians at the decision position's sites: the transformer's residual
    entering block l (sites 1–3) and its final residual (4); the GRU's hidden state at position t + s (sites −2, −1,
    0; a position below 0 is the zero initial state, and a patch there runs every token from it)."""
    def __init__(self, kind, path, H=H, device="cpu", sites=None, batch=2048):
        import torch.nn as nn
        sp = list(sys.path); T = sys.modules.get("maze_train") or _load_by_path("maze_train", MAZE_DIR / "train.py"); sys.path[:] = sp      # train.py puts bump_maze first on sys.path; the other studies' design.py / tables.py must keep winning
        m = T.StepTransformer(H) if kind == "tfm" else T.StepGRU(H); m.tok = nn.Embedding(7 + K, m.tok.embedding_dim)
        if path is not None: m.load_state_dict(torch.load(path, map_location=device))
        self.model, self.kind, self.H, self.dev, self.batch = m.to(device).eval(), kind, H, device, batch
        self.sites = tuple(sites) if sites is not None else (SITES_TFM if kind == "tfm" else SITES_GRU)

    def _tfm(self, tok, ln, patch=None):
        """-> (logits [n, 3], states at every entrance 0..4 at the decision position); patch = (site, new [n, d])."""
        m = self.model; ar = torch.arange(len(tok), device=self.dev); x = m.tok(tok) + m.pos.weight[: tok.shape[1]]; recs = []
        for l in range(m.nl + 1):
            if l > 0:
                b = m.blocks[l - 1]; a = b.attn(x)[0]; mid = x + a; x = mid + b.mlp(b.ln2(mid))
            if patch is not None and patch[0] == l: x = x.clone(); x[ar, ln] = patch[1].to(x.dtype)
            recs.append(x[ar, ln])
        return m.head(m.ln(x))[ar, ln], recs

    def _gru(self, tok, ln, patch=None):
        """-> (logits [n, 3], {site: state}); the natural run over the whole sequence, and after a patch at site s the
        remaining −s tokens run from the installed state."""
        m = self.model; ar = torch.arange(len(tok), device=self.dev); emb = m.tok(tok); h_all, _ = m.gru(emb); d = h_all.shape[-1]
        def nat(s):
            p = ln + s; z = torch.zeros(len(tok), d, device=self.dev, dtype=h_all.dtype); ok = p >= 0
            if ok.any(): z[ok] = h_all[ar[ok], p[ok]]
            return z
        states = {s: nat(s) for s in SITES_GRU}
        if patch is not None:
            s, new = patch; new = new.to(h_all.dtype); states[s] = new
            if s < 0:
                pos = ln[:, None] + torch.arange(s + 1, 1, device=self.dev)[None]                          # the remaining −s positions, ending at the decision
                rest = emb[ar[:, None], pos]; out, _ = m.gru(rest, new[None].contiguous())
                for s2 in SITES_GRU:
                    if s2 > s: states[s2] = out[:, s2 - s - 1]
        return m.head(states[0]), states

    def _run(self, hists, g, patch=None):
        tok, ln = small_tokens(hists, g, self.H); tok, ln = tok.to(self.dev), ln.to(self.dev)
        if self.kind == "tfm": z, recs = self._tfm(tok, ln, patch); X = torch.stack([recs[s] for s in self.sites], 1)
        else: z, states = self._gru(tok, ln, patch); X = torch.stack([states[s] for s in self.sites], 1)
        z = z.double(); return (z - z.mean(1, keepdim=True)).cpu(), X.float().cpu()

    @torch.no_grad()
    def record(self, hists, g):
        """-> (centred logits [n, 3] float64, states at the sites [n, len(sites), d] float32)."""
        out = [self._run(hists[b:b + self.batch], g) for b in range(0, len(hists), self.batch)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])

    @torch.no_grad()
    def patched_record(self, hists, g, site, new):
        new = torch.as_tensor(new).float(); out = [self._run(hists[b:b + self.batch], g, (site, new[b:b + self.batch].to(self.dev))) for b in range(0, len(hists), self.batch)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])

    def patched(self, hists, g, site, new):
        return self.patched_record(hists, g, site, new)[0]

    def jacobian(self, hists, g, site):
        """The Jacobian of the three centred logits with respect to the site's state at the decision position, [n, 3, d]."""
        _, X = self.record(hists, g); si = self.sites.index(site); J = []
        for b in range(0, len(hists), self.batch):
            leaf = X[b:b + self.batch, si].clone().to(self.dev).requires_grad_(True)
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):                 # cuDNN's RNN backward wants training mode; the plain kernels differentiate in eval mode
                z, _ = (self._tfm if self.kind == "tfm" else self._gru)(*[t.to(self.dev) for t in small_tokens(hists[b:b + self.batch], g, self.H)], (site, leaf))
                z = z - z.mean(1, keepdim=True)
                J.append(torch.stack([torch.autograd.grad(z[:, j].sum(), leaf, retain_graph=j < 2)[0] for j in range(3)], 1).detach().double().cpu())
        return torch.cat(J)


def load_small(kind, H=H, device="cpu", run_name=None):
    return Small(kind, GOALS_DIR / "runs" / (run_name or f"{kind}_120000") / "model.pt", H, device)


# ------------------------------------------------------------------ part 3: the fine-tuned model, the pool and the recorded states
class LLM(CC.Model):
    """The coordinates study's hooked fine-tuned model, plus patched recording (the states at the sites after a patch
    at one of them; at the patched site the installed value) and Jacobians of the centred logits."""
    def __init__(self, run_dir=None, worlds_=None, device="cuda", batch=32, sites=SITES_LLM):
        super().__init__(run_dir or CC.RUN / "model", worlds_, device, batch, sites); self.model.requires_grad_(False)

    @torch.no_grad()
    def patched_record(self, hists, g, site, new):
        Z, X = [], []; idx = torch.tensor(self.sites); new = torch.as_tensor(new).float()
        for b in range(0, len(hists), self.batch):
            ids, att = self._ids(hists[b:b + self.batch], g); z, x = self.P.run(ids, att, patch=(site, new[b:b + self.batch].to(self.dev)), record=True); Z.append(z); X.append(x[:, idx])
        X = torch.cat(X)
        if site in self.sites: X[:, self.sites.index(site)] = new                      # the hook records a layer's input before replacing it
        return torch.cat(Z), X

    def jacobian(self, hists, g, site, batch=8):
        """The Jacobian of the three centred logits with respect to the decision token's residual entering `site`, [n, 3, D]."""
        _, X = self.record(hists, g); si = self.sites.index(site); J = []; P = self.P
        for b in range(0, len(hists), batch):
            ids, att = self._ids(hists[b:b + batch], g); leaf = X[b:b + batch, si].clone().to(self.dev).requires_grad_(True)
            with torch.enable_grad():
                P._patch, P._rec = (site, leaf), None
                h = self.model.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state[:, -1]
                z = self.model.lm_head(h).float()[:, self.fmt.act_tok.to(self.dev)]; z = z - z.mean(1, keepdim=True); P._patch = None
                J.append(torch.stack([torch.autograd.grad(z[:, j].sum(), leaf, retain_graph=j < 2)[0] for j in range(3)], 1).detach().double().cpu())
        return torch.cat(J)


def pool(worlds_, sols, quick=False):
    """The coordinates study's decision pool: decisions after the first step along the optimum under every goal,
    deduplicated by history and split by history (fit / test), with each decision's history, step, goal-free set key
    and (test) size."""
    dec = CC.swap_decisions(worlds_, sols, 64 if quick else 1024); fit, test = CC.split_swap_pool(dec, 0.25, 16 if quick else 1024); w = worlds_[0]
    hist = lambda idx: [(dec[i]["acts"], dec[i]["obss"]) for i in idx]; steps = lambda idx: torch.tensor([dec[i]["step"] for i in idx])
    keys = lambda idx: np.array([set_key(w, dec[i]["acts"], dec[i]["obss"]) for i in idx], dtype=object)
    return dict(dec=dec, fit_idx=fit, test_idx=test, hists_fit=hist(fit), hists_test=hist(test), steps_fit=steps(fit), steps_test=steps(test), keys_fit=keys(fit), keys_test=keys(test),
                sizes_test=np.array([dec[i]["size"] for i in test]))


def load_states(kind):
    """The pool's recorded states: for the fine-tuned model the coordinates study's cache (X_fit / X_h1 [n, 4 sites,
    896], L_fit / L_h1, steps, hists, dec indices); for a from-scratch model this study's cache from subspace_small.py."""
    if kind == "llm":
        p = COORDS / "cache" / "states.pt"
        if not p.exists(): raise FileNotFoundError(f"{p} missing: run bump_coords/collect.py first (it records the pool at sites {SITES_LLM})")
        st = torch.load(p, weights_only=False); assert tuple(st["sites"]) == SITES_LLM, st["sites"]; return st["swap"]
    p = CACHE / f"states_{kind}.pt"
    if not p.exists(): raise FileNotFoundError(f"{p} missing: run subspace_small.py --kind {kind} first")
    return torch.load(p, weights_only=False)
