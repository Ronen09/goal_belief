# Geometric Prominence (TASK4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what makes the delayed cue of the round 4 HMM geometrically prominent in a GRU, by sweeping relevance frequency r, strength δ, loss weight λ and delay k, and regressing prominence on decodability, forgetting cost and gradient pressure.

**Architecture:** A parametrised HMM family (`goalgeo/hmm4.py`) with vectorised inference, a weighted sequential trainer with in-memory checkpoints (`train_weighted` in `goalgeo/seqmodels.py`), a measurement module (`goalgeo/prominence.py`) that returns one flat dict per checkpoint, a runner (`scripts/run_task4.py`) that runs conditions in parallel processes and writes tables/JSON, and figures (`goalgeo/plotting4.py`).

**Tech Stack:** Python 3.12, numpy, scipy, torch 2.14 (ROCm, CPU used with `--jobs`), matplotlib, pytest. No sklearn, no pandas.

**Spec:** `docs/superpowers/specs/2026-09-18-geometric-prominence-design.md`

## Global Constraints

- Run everything with `.venv/bin/python` from the project root; tests with `.venv/bin/python -m pytest`.
- Not a git repository: there are no commit steps. Each task ends with the full test suite green.
- GRU: `SeqNet(n_vocab=7, hidden=64, emb=16, out_dim=7)`. Training: T=48, batch 128, Adam 3e-3, 3000 steps, seeds {0, 1, 2}. Checkpoints at steps {0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000}.
- Evaluation: 2000 sequences of length 48, seed 123; RSA on 800 subsampled states; decoding on 3000.
- Base condition: r=1, δ=0.4, k=1, λ=1. Sweeps: r ∈ {0, .05, .1, .25, .5, .75, 1}; δ ∈ {0, .02, .05, .1, .2, .4}; λ ∈ {0, .1, .25, .5, 1, 2, 5}; k ∈ {1, 2, 4, 8, 16} plain and matched; grid r ∈ {.1, .25, .5, 1} × δ ∈ {.05, .1, .2, .4} gated.
- Outputs in `results4/`: `tables4.md`, `results4.json`, `fig1_info_vs_prominence.png`, `fig2_prominence_vs_cost.png`, `fig3_gradient_vs_prominence.png`, `fig4_dynamics.png`, `fig5_delay.png`, `REPORT4.md`.

---

### Task 1: HMM family `goalgeo/hmm4.py`

**Files:**
- Create: `goalgeo/hmm4.py`
- Test: `tests/test_hmm4.py`

**Interfaces:**
- Consumes: `goalgeo.hmm.HMM` dataclass (fields `T, E, pi, name`).
- Produces: `TOK` (dict token→id), `V = 7`, `HMM4(HMM)` with fields `r, delta, k, S (state name→index), mirror (int array), relevant (bool array)`; `make_hmm4(r, delta, k) -> HMM4`; `sample(m, n, T, seed) -> (X [n,T] int, Z [n,T] int)`; `beliefs_seq(m, X) -> [n,T,S]`; `next_token(m, B) -> [..., V]`; `joint_predictive(m, B, k) -> [N, V**k]`; `marginals(m, B, k) -> [N, k*V]`; `forgetting_cost(m, X, Z) -> {"per_step", "per_event", "relevant_frac"}`; `relevant_fraction(m) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hmm4.py
import numpy as np

from goalgeo import hmm4 as H4


def _after(m, tokens):
    return H4.beliefs_seq(m, np.array([tokens]))[0, -1]


def test_base_condition_reduces_to_round4_chain():
    m = H4.make_hmm4(1.0, 0.4, 1)
    S, T = m.S, H4.TOK
    assert len(m.pi) == 11
    bp, bq = _after(m, [T["p"]]), _after(m, [T["q"]])
    assert np.isclose(bp[S["P"]], 1) and np.isclose(bq[S["Q"]], 1)
    assert np.allclose(H4.next_token(m, bp), H4.next_token(m, bq))
    p2p, p2q = H4.joint_predictive(m, bp[None], 2)[0], H4.joint_predictive(m, bq[None], 2)[0]
    assert not np.allclose(p2p, p2q) and np.isclose(p2p.sum(), 1)
    assert np.isclose(m.E[S["C"], T["x"]], 0.9) and np.isclose(m.E[S["D"], T["y"]], 0.9)


def test_one_step_equivalence_holds_at_every_filler_step_for_all_conditions():
    for r, d, k in [(0.0, 0.4, 1), (0.5, 0.1, 4), (1.0, 0.0, 2), (0.25, 0.4, 8)]:
        m = H4.make_hmm4(r, d, k); T = H4.TOK
        for j in range(k + 1):                       # cue, then j filler tokens
            a = _after(m, [T["p"]] + [T["x"]] * j); b = _after(m, [T["q"]] + [T["x"]] * j)
            assert np.allclose(H4.next_token(m, a), H4.next_token(m, b)) or j == k, (r, d, k, j)
        a, b = _after(m, [T["p"]] + [T["x"]] * k), _after(m, [T["q"]] + [T["x"]] * k)
        differs = not np.allclose(H4.next_token(m, a), H4.next_token(m, b))
        assert differs == (r > 0 and d > 0)


def test_control_pair_is_fixed_and_immediate():
    for r, d, k in [(0.0, 0.0, 1), (1.0, 0.4, 16)]:
        m = H4.make_hmm4(r, d, k); T = H4.TOK
        pu, pv = H4.next_token(m, _after(m, [T["u"]])), H4.next_token(m, _after(m, [T["v"]]))
        assert np.isclose(pu[T["x"]] - pv[T["x"]], 0.8)


def test_sample_matches_forward_marginals_and_states():
    m = H4.make_hmm4(0.5, 0.4, 2)
    X, Z = H4.sample(m, n=3000, T=20, seed=0)
    assert X.shape == Z.shape == (3000, 20) and X.max() < H4.V and Z.max() < len(m.pi)
    after_p = Z[:, 1:][X[:, :-1] == H4.TOK["p"]]
    assert np.all(after_p == m.S["FA1"])
    three_after_p = X[:, 3:][X[:, :-3] == H4.TOK["p"]]
    assert abs(np.mean(three_after_p == H4.TOK["n"]) - 0.5) < 0.05
    assert abs(np.mean(three_after_p == H4.TOK["x"]) - 0.45) < 0.05


def test_forgetting_cost_matches_analytic_value():
    for r, d in [(1.0, 0.4), (0.5, 0.2), (0.25, 0.05), (0.0, 0.4), (1.0, 0.0)]:
        m = H4.make_hmm4(r, d, 1)
        X, Z = H4.sample(m, n=500, T=30, seed=1)
        fc = H4.forgetting_cost(m, X, Z)
        h = 0.0 if d == 0.5 else -((0.5 + d) * np.log(0.5 + d) + (0.5 - d) * np.log(0.5 - d))
        expected = r * (np.log(2) - h)
        assert np.isclose(fc["per_event"], expected, atol=1e-6), (r, d, fc, expected)
        assert np.isclose(fc["per_step"], expected * fc["relevant_frac"], rtol=1e-6)
        assert abs(fc["relevant_frac"] - H4.relevant_fraction(m)) < 0.03


def test_mirror_is_an_involution_that_swaps_branches():
    m = H4.make_hmm4(1.0, 0.4, 3)
    assert np.all(m.mirror[m.mirror] == np.arange(len(m.pi)))
    assert m.mirror[m.S["FA2"]] == m.S["FB2"] and m.mirror[m.S["C"]] == m.S["D"] and m.mirror[m.S["U"]] == m.S["U"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hmm4.py -q`
Expected: ImportError / ModuleNotFoundError for `goalgeo.hmm4`.

- [ ] **Step 3: Write the implementation**

```python
# goalgeo/hmm4.py
"""Parametrised HMM family for TASK4 (geometric prominence).

A delayed cue branch: cue p/q -> k filler states (x/y 0.5) -> with prob r the
branch-specific state C/D (x: 0.5 +/- delta), else a neutral state N (emits n).
A fixed control branch: cue u/v -> A'/B' (x 0.9 / 0.1), relevant immediately.
Everything returns uniformly to the four cue states."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import hmm as Hm

TOK = {"p": 0, "q": 1, "u": 2, "v": 3, "x": 4, "y": 5, "n": 6}
V = 7


@dataclass
class HMM4(Hm.HMM):
    r: float = 1.0
    delta: float = 0.4
    k: int = 1
    S: dict | None = None            # state name -> index
    mirror: np.ndarray | None = None  # swaps the A branch with the B branch
    relevant: np.ndarray | None = None  # states that emit the relevant (t*) token: C, D, N


def make_hmm4(r: float = 1.0, delta: float = 0.4, k: int = 1) -> HMM4:
    assert 0 <= r <= 1 and 0 <= delta <= 0.5 and k >= 1
    S = {"P": 0, "Q": 1, "U": 2, "V": 3}
    for i in range(k):
        S[f"FA{i + 1}"] = 4 + i
    for i in range(k):
        S[f"FB{i + 1}"] = 4 + k + i
    base = 4 + 2 * k
    S.update({"C": base, "D": base + 1, "N": base + 2, "A'": base + 3, "B'": base + 4})
    n = base + 5
    T = np.zeros((n, n)); E = np.zeros((n, V))
    ret = [S["P"], S["Q"], S["U"], S["V"]]
    T[S["P"], S["FA1"]] = 1; T[S["Q"], S["FB1"]] = 1
    for i in range(1, k):
        T[S[f"FA{i}"], S[f"FA{i + 1}"]] = 1; T[S[f"FB{i}"], S[f"FB{i + 1}"]] = 1
    T[S[f"FA{k}"], S["C"]] = r; T[S[f"FA{k}"], S["N"]] = 1 - r
    T[S[f"FB{k}"], S["D"]] = r; T[S[f"FB{k}"], S["N"]] = 1 - r
    T[S["U"], S["A'"]] = 1; T[S["V"], S["B'"]] = 1
    for s in ("C", "D", "N", "A'", "B'"):
        T[S[s], ret] = 0.25
    E[S["P"], TOK["p"]] = E[S["Q"], TOK["q"]] = E[S["U"], TOK["u"]] = E[S["V"], TOK["v"]] = 1
    xy = [TOK["x"], TOK["y"]]
    for i in range(k):
        E[S[f"FA{i + 1}"], xy] = 0.5; E[S[f"FB{i + 1}"], xy] = 0.5
    E[S["C"], xy] = [0.5 + delta, 0.5 - delta]; E[S["D"], xy] = [0.5 - delta, 0.5 + delta]
    E[S["N"], TOK["n"]] = 1
    E[S["A'"], xy] = [0.9, 0.1]; E[S["B'"], xy] = [0.1, 0.9]
    w, v = np.linalg.eig(T.T); pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = np.abs(pi) / np.abs(pi).sum()
    mirror = np.arange(n)
    mirror[[S["P"], S["Q"]]] = [S["Q"], S["P"]]; mirror[[S["C"], S["D"]]] = [S["D"], S["C"]]
    for i in range(k):
        a, b = S[f"FA{i + 1}"], S[f"FB{i + 1}"]; mirror[[a, b]] = [b, a]
    relevant = np.zeros(n, bool); relevant[[S["C"], S["D"], S["N"]]] = True
    return HMM4(T, E, pi, f"hmm4(r={r},delta={delta},k={k})", r, delta, k, S, mirror, relevant)


def relevant_fraction(m: HMM4) -> float:
    """Stationary fraction of positions whose token is the relevant one."""
    return float(m.pi[m.relevant].sum())


def sample(m: Hm.HMM, n: int, T: int, seed: int = 0):
    """Vectorised sampling of n sequences of length T from the stationary chain."""
    rng = np.random.default_rng(seed)
    cT, cE = np.cumsum(m.T, 1), np.cumsum(m.E, 1)
    n_s, n_v = m.T.shape[0], m.E.shape[1]
    s = rng.choice(n_s, size=n, p=m.pi)
    X = np.zeros((n, T), int); Z = np.zeros((n, T), int)
    for t in range(T):
        s = np.minimum((rng.random(n)[:, None] > cT[s]).sum(1), n_s - 1)
        Z[:, t] = s
        X[:, t] = np.minimum((rng.random(n)[:, None] > cE[s]).sum(1), n_v - 1)
    return X, Z


def beliefs_seq(m: Hm.HMM, X: np.ndarray) -> np.ndarray:
    """Posterior over the state that emitted X[:, t], for every t: [n, T, S]."""
    n, T_ = X.shape; B = np.zeros((n, T_, len(m.pi)))
    b = np.tile(m.pi, (n, 1))
    for t in range(T_):
        b = (b @ m.T) * m.E[:, X[:, t]].T
        b = b / np.maximum(b.sum(1, keepdims=True), 1e-300)
        B[:, t] = b
    return B


def next_token(m: Hm.HMM, B: np.ndarray) -> np.ndarray:
    """P(next token | belief) for beliefs in the last axis."""
    return (B @ m.T) @ m.E


def joint_predictive(m: Hm.HMM, B: np.ndarray, k: int) -> np.ndarray:
    """Joint distribution over the next k tokens for each belief row: [N, V**k], row-major."""
    B = np.atleast_2d(B); n_v = m.E.shape[1]
    out = np.zeros((len(B), n_v ** k))
    for i, seq in enumerate(itertools.product(range(n_v), repeat=k)):
        bb = B; p = np.ones(len(B))
        for x in seq:
            bb = (bb @ m.T) * m.E[:, x]; s = bb.sum(1); p = p * s
            bb = bb / np.maximum(s[:, None], 1e-300)
        out[:, i] = p
    return out


def marginals(m: Hm.HMM, B: np.ndarray, k: int) -> np.ndarray:
    """P(X_{t+j} | belief) for j = 1..k, concatenated: [N, k*V]."""
    B = np.atleast_2d(B); out = []
    for _ in range(k):
        B = B @ m.T; out.append(B @ m.E)
    return np.concatenate(out, axis=1)


def forgetting_cost(m: HMM4, X: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    """Increase in optimal next-token loss if the filter forgets which branch it is on
    (belief symmetrised across the branch mirror after every update): mean KL per
    position, mean KL per relevant event, and the fraction of relevant positions."""
    B = beliefs_seq(m, X)
    Bf = 0.5 * (B + B[..., m.mirror])
    P, Pf = next_token(m, B)[:, :-1], next_token(m, Bf)[:, :-1]
    kl = (P * (np.log(P + 1e-12) - np.log(Pf + 1e-12))).sum(-1)      # [n, T-1]
    rel = m.relevant[Z[:, 1:]]
    return {"per_step": float(kl.mean()), "per_event": float(kl[rel].mean()) if rel.any() else 0.0,
            "relevant_frac": float(rel.mean())}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hmm4.py -q`
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all previous tests still pass (61 + 6).

---

### Task 2: Weighted sequential trainer with checkpoints

**Files:**
- Modify: `goalgeo/seqmodels.py` (append after `train_objective`)
- Test: `tests/test_prominence.py` (new file, first two tests)

**Interfaces:**
- Consumes: `SeqNet` (`states`, `forward_all`, `out`), `hmm4.sample`, `hmm4.beliefs_seq`, `hmm4.next_token`, `HMM4.relevant`.
- Produces: `CHECKPOINTS` tuple; `weighted_seq_loss(net, X, Y, W) -> scalar tensor`; `train_weighted(net, m, lam=1.0, steps=3000, batch=128, seed=0, T=48, lr=3e-3, pool=4000, checkpoints=CHECKPOINTS) -> (hist: list[float], ckpt: dict[int, state_dict])`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prominence.py
import numpy as np
import torch

from goalgeo import hmm4 as H4, seqmodels as Sm


def test_weighted_loss_zero_weight_removes_relevant_gradient():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    X, Z = H4.sample(m, 8, 12, seed=0)
    Y = H4.next_token(m, H4.beliefs_seq(m, X))[:, :-1]
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y, dtype=torch.float32)
    rel = torch.as_tensor(m.relevant[Z[:, 1:]])
    # weight 1 everywhere except relevant positions -> loss must not depend on the relevant targets
    W = torch.where(rel, 0.0, 1.0)
    l1 = Sm.weighted_seq_loss(net, Xt, Yt, W)
    Y2 = Yt.clone(); Y2[rel] = torch.roll(Y2[rel], 1, dims=-1)
    l2 = Sm.weighted_seq_loss(net, Xt, Y2, W)
    assert torch.isclose(l1, l2)
    W1 = torch.ones_like(W)
    assert not torch.isclose(Sm.weighted_seq_loss(net, Xt, Yt, W1), Sm.weighted_seq_loss(net, Xt, Y2, W1))


def test_train_weighted_learns_and_returns_checkpoints():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    hist, ckpt = Sm.train_weighted(net, m, lam=2.0, steps=80, batch=32, seed=0, T=16, pool=200, checkpoints=(0, 40, 80))
    assert len(hist) == 80 and np.mean(hist[-10:]) < np.mean(hist[:10])
    assert sorted(ckpt) == [0, 40, 80]
    assert not torch.equal(ckpt[0]["out.weight"], ckpt[80]["out.weight"])
    assert torch.equal(ckpt[80]["out.weight"], net.state_dict()["out.weight"].cpu())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prominence.py -q`
Expected: AttributeError: module 'goalgeo.seqmodels' has no attribute 'weighted_seq_loss'.

- [ ] **Step 3: Append the implementation to `goalgeo/seqmodels.py`**

```python
# --- TASK4 (geometric prominence): weighted sequential training with checkpoints ---
from . import hmm4 as H4  # noqa: E402  (placed here to keep the round 4 section untouched)

CHECKPOINTS = (0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000)


def weighted_seq_loss(net, X, Y, W):
    """Per-position soft cross-entropy, weighted per position.
    X [B, T] long tokens; Y [B, T-1, V] exact targets for tokens 1..T-1; W [B, T-1] weights."""
    logits = net.forward_all(X)[:, :-1]
    ce = -(Y * F.log_softmax(logits, -1)).sum(-1)
    return (W * ce).sum() / W.sum()


def train_weighted(net, m: H4.HMM4, lam: float = 1.0, steps: int = 3000, batch: int = 128, seed: int = 0,
                   T: int = 48, lr: float = 3e-3, pool: int = 4000, checkpoints=CHECKPOINTS):
    """Sequential next-token training with exact targets. Positions whose *next* token is
    emitted by a relevant state (C, D, N) get weight ``lam``; all others weight 1.
    Returns (loss history, {step: CPU state_dict copy}) with a copy at every listed step."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed); dev = _dev(net)
    X, Z = H4.sample(m, pool, T, seed=seed)
    Y = H4.next_token(m, H4.beliefs_seq(m, X))[:, :-1]
    W = np.where(m.relevant[Z[:, 1:]], lam, 1.0)
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []; ckpt = {}
    for step in range(steps + 1):
        if step in checkpoints:
            ckpt[step] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if step == steps:
            break
        idx = torch.as_tensor(rng.integers(0, pool, batch), device=dev)
        loss = weighted_seq_loss(net, Xt[idx], Yt[idx], Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
    return hist, ckpt
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prominence.py -q`
Expected: 2 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

---

### Task 3: Measurements `goalgeo/prominence.py`

**Files:**
- Create: `goalgeo/prominence.py`
- Test: `tests/test_prominence.py` (append)

**Interfaces:**
- Consumes: `hmm4` (sample, beliefs_seq, next_token, joint_predictive, marginals, HMM4.S/k/relevant), `geometry` (rdm, upper, rsa, ridge_cv_r2), `SeqNet.states/out/forward_all`.
- Produces: `make_eval(m, n=2000, T=48, seed=123) -> {"X","Z","B","Y"}`; `linear_cv_accuracy(X, y_bool) -> float`; `cross_distance(ha, hb, cap=400, rng=None) -> float`; `positions(m, Z) -> (dict of bool masks [n,T] with keys cueA cueB preA preB ctrlU ctrlV, anchors int array [M,2] of (sequence, cue position))`; `gradients(net, X, Y, anchors, k, dhat) -> {"G_rel","Gpar_rel","G_imm","Gpar_imm"}`; `measure(net, m, ev, n_rsa=800, n_dec=3000, seed=0) -> dict` with keys `D_delay, D_control, D_pre, P_metric, P_metric_pre, D_delay_over_median, var_ratio_delayed_vs_random, belief_r2, branch_acc_pre, rsa_belief, rsa_P1, rsa_P2, rsa_future, kl, kl_relevant, G_rel, Gpar_rel, G_imm, Gpar_imm`.

- [ ] **Step 1: Append the failing tests**

```python
# tests/test_prominence.py (append)
from goalgeo import prominence as Pr


def test_linear_cv_accuracy_on_separable_and_random_labels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 8)); y = X[:, 0] > 0
    assert Pr.linear_cv_accuracy(X, y) > 0.9
    assert abs(Pr.linear_cv_accuracy(X, rng.random(400) > 0.5) - 0.5) < 0.15


def test_positions_and_anchor_geometry():
    m = H4.make_hmm4(1.0, 0.4, 3)
    ev = Pr.make_eval(m, n=50, T=30, seed=0)
    P, anchors = Pr.positions(m, ev["Z"])
    assert P["cueA"].sum() > 0 and P["preB"].sum() > 0 and P["ctrlU"].sum() > 0
    for i, t in anchors:
        assert ev["Z"][i, t] in (m.S["P"], m.S["Q"])
        assert m.relevant[ev["Z"][i, t + m.k + 1]] and t + m.k <= 30 - 2


def test_measure_returns_all_keys_and_sane_values():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    ev = Pr.make_eval(m, n=120, T=24, seed=0)
    res = Pr.measure(net, m, ev, n_rsa=100, n_dec=300)
    for key in ("P_metric", "P_metric_pre", "belief_r2", "branch_acc_pre", "rsa_belief", "rsa_P1", "rsa_P2", "rsa_future",
                "kl", "kl_relevant", "G_rel", "Gpar_rel", "G_imm", "Gpar_imm", "var_ratio_delayed_vs_random"):
        assert key in res and np.isfinite(res[key]), key
    assert res["P_metric"] > 0 and res["kl"] >= 0
    assert res["Gpar_rel"] <= res["G_rel"] + 1e-9 and res["Gpar_imm"] <= res["G_imm"] + 1e-9


def test_cross_distance_of_identical_groups_is_zero():
    h = np.ones((10, 4))
    assert Pr.cross_distance(h, h) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prominence.py -q`
Expected: ImportError for `goalgeo.prominence`.

- [ ] **Step 3: Write the implementation**

```python
# goalgeo/prominence.py
"""TASK4 measurements on a trained GRU: pairwise metric prominence of the delayed
cue, decodability, RSA against belief/predictive geometries, gradient relevance
of the delayed prediction, and KL to the exact targets."""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as G
from . import hmm4 as H4


def make_eval(m: H4.HMM4, n: int = 2000, T: int = 48, seed: int = 123) -> dict:
    X, Z = H4.sample(m, n, T, seed)
    B = H4.beliefs_seq(m, X)
    return {"X": X, "Z": Z, "B": B, "Y": H4.next_token(m, B)}


def linear_cv_accuracy(X: np.ndarray, y: np.ndarray, n_folds: int = 5, alpha: float = 1.0,
                       rng: np.random.Generator | None = None) -> float:
    """Cross-validated accuracy of a ridge classifier (least squares onto +/-1, sign readout)."""
    rng = rng or np.random.default_rng(0)
    n = len(X); folds = np.array_split(rng.permutation(n), n_folds)
    s = np.where(y, 1.0, -1.0); correct = 0
    for held in folds:
        tr = np.ones(n, bool); tr[held] = False
        mx, my = X[tr].mean(0), s[tr].mean(); Xc = X[tr] - mx
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (s[tr] - my))
        correct += (np.sign((X[held] - mx) @ w + my) == s[held]).sum()
    return float(correct / n)


def cross_distance(ha: np.ndarray, hb: np.ndarray, cap: int = 400, rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    ha = ha[rng.choice(len(ha), min(cap, len(ha)), replace=False)]
    hb = hb[rng.choice(len(hb), min(cap, len(hb)), replace=False)]
    return float(np.linalg.norm(ha[:, None] - hb[None], axis=-1).mean())


def positions(m: H4.HMM4, Z: np.ndarray):
    """Boolean masks [n, T] for the cue pair (after p / after q), the pre-relevant pair
    (last filler of each branch) and the control pair (after u / after v), plus gradient
    anchors (sequence, cue position) using the last cue whose relevant loss is inside the sequence."""
    S, k = m.S, m.k
    P = {"cueA": Z == S["P"], "cueB": Z == S["Q"], "preA": Z == S[f"FA{k}"], "preB": Z == S[f"FB{k}"],
         "ctrlU": Z == S["U"], "ctrlV": Z == S["V"]}
    n, T = Z.shape; anchors = []
    for i in range(n):
        ts = np.flatnonzero((Z[i] == S["P"]) | (Z[i] == S["Q"])); ts = ts[ts + k <= T - 2]
        if len(ts):
            anchors.append((i, int(ts[-1])))
    return P, np.array(anchors, dtype=int).reshape(-1, 2)


def gradients(net, X: np.ndarray, Y: np.ndarray, anchors: np.ndarray, k: int, dhat: np.ndarray) -> dict[str, float]:
    """Gradient of the relevant loss l_{t+k} (prediction of the token at t+k+1) and of the
    immediate loss l_t with respect to the cue state h_t, per anchored sequence.
    Returns mean norms and mean absolute projections onto the unit direction dhat."""
    dev = next(net.parameters()).device
    i, t = anchors[:, 0], anchors[:, 1]
    Xt = torch.as_tensor(X[i], device=dev); Yt = torch.as_tensor(Y[i], dtype=torch.float32, device=dev)
    ar = torch.arange(len(i), device=dev); tc = torch.as_tensor(t, device=dev)
    out = {}
    for name, off in (("rel", k), ("imm", 0)):
        net.zero_grad()
        states = net.states(Xt); states.retain_grad()
        logp = torch.log_softmax(net.out(states), -1)
        tt = tc + off
        loss = -(Yt[ar, tt] * logp[ar, tt]).sum()
        loss.backward()
        g = states.grad[ar, tc].cpu().numpy()
        out[f"G_{name}"] = float(np.linalg.norm(g, axis=1).mean())
        out[f"Gpar_{name}"] = float(np.abs(g @ dhat).mean())
    net.zero_grad()
    return out


def measure(net, m: H4.HMM4, ev: dict, n_rsa: int = 800, n_dec: int = 3000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X, Z, B, Y = ev["X"], ev["Z"], ev["B"], ev["Y"]
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(X)).cpu().numpy()          # [n, T, H]
    P, anchors = positions(m, Z)
    h = lambda key: Hs[P[key]]                                       # noqa: E731
    d_delay = cross_distance(h("cueA"), h("cueB"), rng=rng)
    d_ctrl = cross_distance(h("ctrlU"), h("ctrlV"), rng=rng)
    d_pre = cross_distance(h("preA"), h("preB"), rng=rng)
    flat = Hs.reshape(-1, Hs.shape[-1]); Bf = B.reshape(-1, B.shape[-1])
    sub = rng.choice(len(flat), min(n_rsa, len(flat)), replace=False)
    Rh = G.rdm(flat[sub]); med = float(np.median(G.upper(Rh)))
    dhat = h("cueA").mean(0) - h("cueB").mean(0); dhat = dhat / (np.linalg.norm(dhat) + 1e-12)
    dirs = rng.standard_normal((20, Hs.shape[-1])); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    var_d = float((flat @ dhat).var()); var_rand = float((flat @ dirs.T).var(0).mean())
    res = {"D_delay": d_delay, "D_control": d_ctrl, "D_pre": d_pre,
           "P_metric": d_delay / (d_ctrl + 1e-12), "P_metric_pre": d_pre / (d_ctrl + 1e-12),
           "D_delay_over_median": d_delay / (med + 1e-12), "var_ratio_delayed_vs_random": var_d / (var_rand + 1e-12)}
    # decodability
    dec = rng.choice(len(flat), min(n_dec, len(flat)), replace=False)
    res["belief_r2"] = G.ridge_cv_r2(flat[dec], Bf[dec], alpha=1.0)
    pre_mask = (P["preA"] | P["preB"]).ravel(); pre = np.flatnonzero(pre_mask)
    pre = rng.choice(pre, min(n_dec, len(pre)), replace=False)
    res["branch_acc_pre"] = linear_cv_accuracy(flat[pre], P["preA"].ravel()[pre], rng=rng)
    # RSA against belief and predictive geometries
    Bs = Bf[sub]
    refs = {"belief": Bs, "P1": H4.next_token(m, Bs), "P2": H4.joint_predictive(m, Bs, 2), "future": H4.marginals(m, Bs, m.k + 2)}
    for name, ref in refs.items():
        res[f"rsa_{name}"] = G.rsa(Rh, G.rdm(ref))
    # performance: KL to the exact targets
    with torch.no_grad():
        logp = torch.log_softmax(net.forward_all(torch.as_tensor(X)), -1)[:, :-1].cpu().numpy()
    Yn = Y[:, :-1]; kl = (Yn * (np.log(Yn + 1e-12) - logp)).sum(-1)
    rel = m.relevant[Z[:, 1:]]
    res["kl"] = float(kl.mean()); res["kl_relevant"] = float(kl[rel].mean()) if rel.any() else 0.0
    res.update(gradients(net, X, Y, anchors, m.k, dhat))
    return res
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prominence.py -q`
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

---

### Task 4: Runner `scripts/run_task4.py` (conditions, parallel runs, analysis, tables)

**Files:**
- Create: `scripts/run_task4.py`
- Test: smoke run with `--quick`

**Interfaces:**
- Consumes: `hmm4.make_hmm4/relevant_fraction/forgetting_cost`, `seqmodels.SeqNet/train_weighted/CHECKPOINTS`, `prominence.make_eval/measure`, `analysis.to_jsonable`, `plotting4.make_figures(R, out)` (Task 5; the runner imports it lazily and skips figures with a warning if the import fails).
- Produces: `results4/results4.json` with `{"runs": [run...], "conditions": {...}, "regression": {...}, "dynamics": {...}, "grid_ran": bool}`; each run = `{"cond": {r, delta, k, lam}, "sweeps": [...], "seed", "init": measure dict, "final": measure dict, "integrated": {"G_rel_int","Gpar_rel_int","G_imm_int","Gpar_imm_int"}, "forget": {...}, "cost": lam*per_step, "dynamics": {step: measure dict}, "loss_hist": [...]}`; `results4/tables4.md`.

- [ ] **Step 1: Write the script**

```python
# scripts/run_task4.py
"""TASK4: what makes information geometrically prominent? Sweeps relevance frequency r,
strength delta, loss weight lambda and delay k on the parametrised HMM, trains the GRU
with checkpoints, and regresses prominence on decodability, forgetting cost and gradients.

    .venv/bin/python scripts/run_task4.py [--quick] [--jobs 6] [--grid] [--out results4]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr
from goalgeo.analysis import to_jsonable

BASE = dict(r=1.0, delta=0.4, k=1, lam=1.0)
SWEEPS = {"frequency": [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0], "strength": [0, 0.02, 0.05, 0.1, 0.2, 0.4],
          "weight": [0, 0.1, 0.25, 0.5, 1, 2, 5], "delay": [1, 2, 4, 8, 16]}
GRID_R, GRID_D = [0.1, 0.25, 0.5, 1.0], [0.05, 0.1, 0.2, 0.4]
QUICK = {"frequency": [0, 1.0], "strength": [0, 0.4], "weight": [0, 5], "delay": [1, 4]}
PARAM = {"frequency": "r", "strength": "delta", "weight": "lam", "delay": "k"}
INT_KEYS = ("G_rel", "Gpar_rel", "G_imm", "Gpar_imm")


def key_of(c):
    return (float(c["r"]), float(c["delta"]), int(c["k"]), float(c["lam"]))


def matched_lambda(k):
    f1 = H4.relevant_fraction(H4.make_hmm4(1.0, 0.4, 1)); fk = H4.relevant_fraction(H4.make_hmm4(1.0, 0.4, k))
    return float(f1 / fk)


def conditions(quick, grid=False):
    C = {}
    def add(sweep, **kw):
        c = {**BASE, **kw}; c["k"] = int(c["k"])
        C.setdefault(key_of(c), {"cond": c, "sweeps": []})["sweeps"].append(sweep)
    sw = QUICK if quick else SWEEPS
    for name, vals in sw.items():
        for v in vals:
            add(name, **{PARAM[name]: v})
    for k in sw["delay"]:
        if k != 1:
            add("delay_matched", k=k, lam=matched_lambda(k))
    if grid:
        for r in GRID_R:
            for d in GRID_D:
                add("grid", r=r, delta=d)
    return C


def run_one(args):
    cond, seed, steps, quick = args
    torch.set_num_threads(1)
    m = H4.make_hmm4(cond["r"], cond["delta"], cond["k"])
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed)
    ckpts = tuple(s for s in Sm.CHECKPOINTS if s <= steps) if not quick else (0, steps // 2, steps)
    hist, ckpt = Sm.train_weighted(net, m, lam=cond["lam"], steps=steps, seed=seed, checkpoints=ckpts)
    ev = Pr.make_eval(m, n=2000 if not quick else 300, T=48)
    dyn = {}
    for step, sd in ckpt.items():
        net.load_state_dict(sd); dyn[step] = Pr.measure(net, m, ev, n_rsa=800 if not quick else 200, n_dec=3000 if not quick else 500)
    fc = H4.forgetting_cost(m, ev["X"], ev["Z"])
    integ = {f"{k}_int": float(np.mean([dyn[s][k] for s in dyn])) for k in INT_KEYS}
    return {"cond": cond, "seed": seed, "init": dyn[0], "final": dyn[max(dyn)], "integrated": integ, "forget": fc,
            "cost": float(cond["lam"] * fc["per_step"]), "dynamics": dyn, "loss_hist": hist[::10]}


def run_conditions(C, seeds, steps, quick, jobs, done):
    todo = [(C[key]["cond"], s, steps, quick) for key in C for s in seeds if (key, s) not in done]
    print(f"running {len(todo)} runs on {jobs} workers ...", flush=True)
    t0 = time.time(); out = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            r["sweeps"] = C[key_of(r["cond"])]["sweeps"]; out.append(r)
            f = r["final"]; c = r["cond"]
            print(f"[{i + 1}/{len(todo)}] r={c['r']:.2f} d={c['delta']:.2f} k={c['k']:2d} lam={c['lam']:.2f} seed={r['seed']}  "
                  f"P_metric={f['P_metric']:.2f} pre={f['P_metric_pre']:.2f} R2={f['belief_r2']:.2f} acc={f['branch_acc_pre']:.2f} "
                  f"Gpar_int={r['integrated']['Gpar_rel_int']:.3f} kl={f['kl']:.3f}  {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------- analysis ----------------
def rows_of(runs):
    return [{"P_metric": r["final"]["P_metric"], "belief_r2": r["final"]["belief_r2"], "branch_acc_pre": r["final"]["branch_acc_pre"],
             "cost": r["cost"], "G_rel_int": r["integrated"]["G_rel_int"], "Gpar_rel_int": r["integrated"]["Gpar_rel_int"],
             "sweeps": r["sweeps"], "cond": r["cond"]} for r in runs]


def ols(Xm, y):
    Z = np.column_stack([np.ones(len(y)), Xm]); beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    pred = Z @ beta; r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return beta, float(r2)


def regression(rows, names=("belief_r2", "cost", "G_rel_int", "Gpar_rel_int")):
    Xm = np.array([[r[n] for n in names] for r in rows]); y = np.array([r["P_metric"] for r in rows])
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-12; Xs = (Xm - mu) / sd; ys = (y - y.mean()) / (y.std() + 1e-12)
    beta, r2 = ols(Xs, ys)
    out = {"n": len(rows), "r2_full": r2, "beta": {n: float(b) for n, b in zip(names, beta[1:])},
           "spearman": {n: float(spearmanr(Xm[:, j], y).correlation) for j, n in enumerate(names)},
           "r2_single": {n: ols(Xs[:, [j]], ys)[1] for j, n in enumerate(names)}}
    for subset, label in ((("belief_r2",), "decodability_only"), (("cost",), "cost_only"), (("cost", "Gpar_rel_int"), "cost_plus_gradient"),
                          (("belief_r2", "cost"), "decodability_plus_cost")):
        idx = [names.index(n) for n in subset]; out[f"r2_{label}"] = ols(Xs[:, idx], ys)[1]
    # leave-one-sweep-out out-of-sample R^2
    loso = {}
    for sweep in sorted({s for r in rows for s in r["sweeps"]}):
        te = np.array([sweep in r["sweeps"] for r in rows]); tr = ~te
        if te.sum() < 3 or tr.sum() < 5:
            continue
        m_, s_ = Xm[tr].mean(0), Xm[tr].std(0) + 1e-12
        beta, _ = ols((Xm[tr] - m_) / s_, y[tr]); pred = np.column_stack([np.ones(te.sum()), (Xm[te] - m_) / s_]) @ beta
        loso[sweep] = float(1 - ((y[te] - pred) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-12))
    out["loso_r2"] = loso
    return out


def dynamics_summary(run):
    dyn = run["dynamics"]; steps = sorted(dyn)
    def t90(key):
        v0, v1 = dyn[steps[0]][key], dyn[steps[-1]][key]; tot = abs(v1 - v0)
        if tot < 1e-9:
            return None
        return next(s for s in steps if abs(dyn[s][key] - v0) >= 0.9 * tot)
    peak = max(steps, key=lambda s: dyn[s]["G_rel"])
    return {"t90_branch_acc_pre": t90("branch_acc_pre"), "t90_belief_r2": t90("belief_r2"), "t90_P_metric": t90("P_metric"),
            "t90_kl_relevant": t90("kl_relevant"), "t_peak_G_rel": peak,
            "curves": {k: [dyn[s][k] for s in steps] for k in ("branch_acc_pre", "belief_r2", "P_metric", "P_metric_pre", "G_rel", "Gpar_rel", "kl_relevant", "kl")},
            "steps": steps}


def aggregate(runs):
    """Per condition: mean and std over seeds of the final, init and integrated quantities."""
    C = {}
    for r in runs:
        C.setdefault(key_of(r["cond"]), []).append(r)
    agg = {}
    for key, rr in C.items():
        keys_f = rr[0]["final"].keys()
        a = {"cond": rr[0]["cond"], "sweeps": rr[0]["sweeps"], "n_seeds": len(rr), "cost": float(np.mean([r["cost"] for r in rr])),
             "forget_per_event": float(np.mean([r["forget"]["per_event"] for r in rr])), "relevant_frac": rr[0]["forget"]["relevant_frac"]}
        for k in keys_f:
            v = [r["final"][k] for r in rr]; a[k] = float(np.mean(v)); a[f"{k}_std"] = float(np.std(v))
            a[f"init_{k}"] = float(np.mean([r["init"][k] for r in rr]))
        for k in INT_KEYS:
            a[f"{k}_int"] = float(np.mean([r["integrated"][f"{k}_int"] for r in rr]))
        agg[key] = a
    return agg


def sweep_conditions(agg, sweep):
    return sorted([a for a in agg.values() if sweep in a["sweeps"]], key=lambda a: a["cond"][PARAM.get(sweep, "k")] if sweep != "grid" else (a["cond"]["r"], a["cond"]["delta"]))


def sweep_effect(agg, sweep):
    rows = sweep_conditions(agg, sweep)
    if len(rows) < 3:
        return float("nan")
    return float(spearmanr([a["cond"][PARAM[sweep]] for a in rows], [a["P_metric"] for a in rows]).correlation)


def write_tables(R, out):
    agg = R["agg"]; L = ["# TASK4 tables: geometric prominence\n"]
    cols = [("P_metric", "P_metric"), ("P_metric_pre", "P_metric (pre-relevant)"), ("D_delay_over_median", "d/median"), ("var_ratio_delayed_vs_random", "var ratio vs random dir"),
            ("belief_r2", "belief R²"), ("branch_acc_pre", "branch acc (pre)"), ("rsa_belief", "RSA belief"), ("rsa_P1", "RSA 1-step"), ("rsa_P2", "RSA 2-step"), ("rsa_future", "RSA future"),
            ("G_rel_int", "∫G_t"), ("Gpar_rel_int", "∫G_∥"), ("G_rel", "G_t final"), ("kl", "KL"), ("kl_relevant", "KL at t*")]
    for sweep in ("frequency", "strength", "weight", "delay", "delay_matched", "grid"):
        rows = sweep_conditions(agg, sweep)
        if not rows:
            continue
        L += [f"## {sweep} sweep", "", "| r | δ | k | λ | cost λ·ΔL_forget | ΔL per event | " + " | ".join(c[1] for c in cols) + " |", "|---|" * (6 + len(cols))]
        for a in rows:
            c = a["cond"]
            L.append(f"| {c['r']:.2f} | {c['delta']:.2f} | {c['k']} | {c['lam']:.2f} | {a['cost']:.4f} | {a['forget_per_event']:.4f} | " +
                     " | ".join(f"{a[k]:.3f}±{a[k + '_std']:.3f}" if k + "_std" in a else f"{a[k]:.3f}" for k, _ in cols) + " |")
        a = rows[0]
        L.append(f"| init | | | | | | " + " | ".join(f"{a['init_' + k]:.3f}" if "init_" + k in a else "" for k, _ in cols) + " |")
        L.append("")
    reg = R["regression"]
    L += ["## Regression of P_metric (per run, standardised)", "", f"n = {reg['n']}, R² full = {reg['r2_full']:.3f}", "",
          "| predictor | beta | Spearman | R² alone |", "|---|---|---|---|"]
    for n in reg["beta"]:
        L.append(f"| {n} | {reg['beta'][n]:+.3f} | {reg['spearman'][n]:+.3f} | {reg['r2_single'][n]:.3f} |")
    L += ["", "| model | R² |", "|---|---|"] + [f"| {k[3:]} | {v:.3f} |" for k, v in reg.items() if k.startswith("r2_") and k not in ("r2_full", "r2_single")]
    L += ["", "Leave-one-sweep-out out-of-sample R²: " + ", ".join(f"{k} {v:.3f}" for k, v in reg["loso_r2"].items()), ""]
    L += ["## Training dynamics (base condition, per seed)", "", "| seed | t90 branch acc | t90 belief R² | t90 P_metric | t_peak G_t | t90 KL at t* |", "|---|---|---|---|---|---|"]
    for seed, d in R["dynamics"]["base"].items():
        L.append(f"| {seed} | {d['t90_branch_acc_pre']} | {d['t90_belief_r2']} | {d['t90_P_metric']} | {d['t_peak_G_rel']} | {d['t90_kl_relevant']} |")
    L += ["", f"sweep effects (Spearman of P_metric with the swept parameter): " + ", ".join(f"{k} {v:+.2f}" for k, v in R["sweep_effect"].items()), ""]
    (out / "tables4.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results4")
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--grid", action="store_true", help="force the r x delta grid")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(exist_ok=True); t0 = time.time()
    seeds = [0, 1, 2] if not args.quick else [0]; steps = args.steps or (3000 if not args.quick else 200)
    C = conditions(args.quick)
    runs = run_conditions(C, seeds, steps, args.quick, args.jobs, set())
    agg = aggregate(runs)
    effects = {s: sweep_effect(agg, s) for s in ("frequency", "strength", "weight", "delay")}
    grid_ran = False
    if not args.quick and (args.grid or (effects["frequency"] > 0.5 and effects["strength"] > 0.5)):
        Cg = conditions(False, grid=True); done = {(key_of(r["cond"]), r["seed"]) for r in runs}
        Cg = {k: v for k, v in Cg.items() if "grid" in v["sweeps"]}
        for key in Cg:                      # conditions already run just get the grid label
            if key in C:
                for r in runs:
                    if key_of(r["cond"]) == key and "grid" not in r["sweeps"]:
                        r["sweeps"].append("grid")
        runs += run_conditions(Cg, seeds, steps, False, args.jobs, done); grid_ran = True
        agg = aggregate(runs)
    base_key = key_of(BASE)
    R = {"runs": runs, "agg": agg, "regression": regression(rows_of(runs)), "sweep_effect": effects, "grid_ran": grid_ran,
         "dynamics": {"base": {r["seed"]: dynamics_summary(r) for r in runs if key_of(r["cond"]) == base_key},
                      "all": {f"{key_of(r['cond'])}/{r['seed']}": {k: v for k, v in dynamics_summary(r).items() if k != "curves"} for r in runs}}}
    write_tables(R, out)
    J = to_jsonable({**R, "agg": {str(k): v for k, v in agg.items()}})
    (out / "results4.json").write_text(json.dumps(J, indent=1))
    try:
        from goalgeo.plotting4 import make_figures
        make_figures(R, out)
    except ImportError as e:
        print(f"figures skipped: {e}")
    print(f"done in {time.time() - t0:.0f}s -> {out}/  sweep effects {effects}  regression R² {R['regression']['r2_full']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

Run: `.venv/bin/python scripts/run_task4.py --quick --jobs 4 --out /tmp/claude-1000/-home-ronenrr-Documents-goals/134cc8ee-4117-408e-b170-0397caf7ba60/scratchpad/q4`
Expected: prints one line per run (8 conditions + 1 matched = 9 runs), "figures skipped" (Task 5 not yet done), writes `tables4.md` and `results4.json`. Inspect `tables4.md` for finite numbers.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

---

### Task 5: Figures `goalgeo/plotting4.py`

**Files:**
- Create: `goalgeo/plotting4.py`
- Test: re-run the Task 4 smoke run; open the PNGs.

Before writing this file, invoke the `dataviz` skill (chart form and colour rules); keep the project's `SERIES`/`MUTED` palette from `goalgeo/plotting.py`.

**Interfaces:**
- Consumes: `R` dict from the runner (`agg` keyed by (r, delta, k, lam) tuples, `dynamics["base"]`, `regression`).
- Produces: `make_figures(R, out: Path)` writing `fig1_info_vs_prominence.png`, `fig2_prominence_vs_cost.png`, `fig3_gradient_vs_prominence.png`, `fig4_dynamics.png`, `fig5_delay.png`.

- [ ] **Step 1: Write the implementation**

```python
# goalgeo/plotting4.py
"""TASK4 figures: information vs prominence, prominence vs forgetting cost, gradient
pressure vs prominence, training dynamics, delay sweep."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import SERIES, MUTED, _save

PARAM = {"frequency": "r", "strength": "delta", "weight": "lam", "delay": "k", "delay_matched": "k"}
MARK = {"frequency": ("o", SERIES[0]), "strength": ("s", SERIES[1]), "weight": ("^", SERIES[2]), "delay": ("D", SERIES[3]),
        "delay_matched": ("d", SERIES[4]), "grid": ("x", MUTED)}


def _sweep(agg, sweep):
    rows = [a for a in agg.values() if sweep in a["sweeps"]]
    return sorted(rows, key=lambda a: a["cond"][PARAM[sweep]])


def _errbar(ax, x, rows, key, color, label, marker="o"):
    ax.errorbar(x, [a[key] for a in rows], [a[f"{key}_std"] for a in rows], fmt=marker + "-", color=color, ecolor=MUTED, capsize=2, ms=4, label=label)


def fig_info_vs_prominence(R, out):
    rows = _sweep(R["agg"], "frequency")
    if not rows:
        return
    x = [a["cond"]["r"] for a in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    _errbar(axes[0], x, rows, "belief_r2", SERIES[0], "belief R²")
    _errbar(axes[0], x, rows, "branch_acc_pre", SERIES[2], "branch accuracy at t*−1", "s")
    axes[0].set_xlabel("relevance frequency r"); axes[0].set_ylabel("decodability"); axes[0].set_title("Information: decodable?"); axes[0].grid(True); axes[0].legend()
    _errbar(axes[1], x, rows, "P_metric", SERIES[1], "P_metric (cue)")
    _errbar(axes[1], x, rows, "P_metric_pre", SERIES[3], "P_metric (t*−1)", "s")
    axes[1].axhline(rows[0]["init_P_metric"], color=MUTED, ls="--", lw=0.8, label="init")
    axes[1].set_xlabel("relevance frequency r"); axes[1].set_ylabel("delayed / control distance"); axes[1].set_title("Prominence: amplified?"); axes[1].grid(True); axes[1].legend()
    _save(fig, out / "fig1_info_vs_prominence.png")


def _scatter_by_sweep(R, xkey, xlabel, title, path, ykey="P_metric", logx=False):
    agg = R["agg"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
    for sweep, (mk, col) in MARK.items():
        rows = [a for a in agg.values() if sweep in a["sweeps"]]
        if not rows:
            continue
        ax.errorbar([a[xkey] for a in rows], [a[ykey] for a in rows], [a[f"{ykey}_std"] for a in rows], fmt=mk, color=col, ecolor=MUTED, capsize=2, ms=5, ls="none", label=sweep)
    if logx:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(xlabel); ax.set_ylabel("P_metric = d(delayed pair) / d(control pair)"); ax.set_title(title); ax.grid(True); ax.legend(fontsize=7.5)
    _save(fig, path)


def fig_prominence_vs_cost(R, out):
    _scatter_by_sweep(R, "cost", "expected forgetting cost λ·ΔL_forget (nats / position)", "Prominence vs cost of forgetting", out / "fig2_prominence_vs_cost.png", logx=True)


def fig_gradient_vs_prominence(R, out):
    _scatter_by_sweep(R, "Gpar_rel_int", "∫ |∇_h ℓ_t* · d̂| over training", "Prominence vs gradient pressure", out / "fig3_gradient_vs_prominence.png", logx=True)


def fig_dynamics(R, out):
    D = R["dynamics"]["base"]
    if not D:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), layout="constrained", sharex=True)
    panels = [("branch_acc_pre", "branch accuracy at t*−1", SERIES[2]), ("P_metric", "P_metric", SERIES[1]), ("G_rel", "‖∇_h ℓ_t*‖ at the cue", SERIES[3]), ("kl_relevant", "KL to target at t*", SERIES[0])]
    for ax, (key, label, col) in zip(axes.ravel(), panels):
        for seed, d in D.items():
            ax.plot(d["steps"], d["curves"][key], "o-", color=col, ms=3, alpha=0.4 + 0.6 * (seed == 0), label=f"seed {seed}")
        ax.set_title(label); ax.grid(True); ax.set_xscale("symlog", linthresh=50)
    axes[1, 0].set_xlabel("training step"); axes[1, 1].set_xlabel("training step"); axes[0, 0].legend(fontsize=7.5)
    _save(fig, out / "fig4_dynamics.png")


def fig_delay(R, out):
    plain, matched = _sweep(R["agg"], "delay"), _sweep(R["agg"], "delay_matched")
    if not plain:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for rows, col, lab, mk in ((plain, SERIES[1], "λ = 1", "o"), (matched, SERIES[4], "λ matched", "d")):
        if rows:
            _errbar(axes[0], [a["cond"]["k"] for a in rows], rows, "P_metric", col, lab, mk)
            _errbar(axes[1], [a["cond"]["k"] for a in rows], rows, "branch_acc_pre", col, lab, mk)
    axes[0].set_ylabel("P_metric"); axes[1].set_ylabel("branch accuracy at t*−1")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("delay k (filler steps)"); ax.grid(True); ax.legend()
    axes[0].set_title("Prominence vs delay"); axes[1].set_title("Decodability vs delay")
    _save(fig, out / "fig5_delay.png")


def make_figures(R, out):
    fig_info_vs_prominence(R, out); fig_prominence_vs_cost(R, out); fig_gradient_vs_prominence(R, out); fig_dynamics(R, out); fig_delay(R, out)
```

- [ ] **Step 2: Re-run the smoke run and check the five PNGs exist and render**

Run: `.venv/bin/python scripts/run_task4.py --quick --jobs 4 --out <scratchpad>/q4 && ls <scratchpad>/q4/*.png`
Expected: five PNGs, no "figures skipped" line. View each PNG with the Read tool.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

---

### Task 6: Full run, report, docs

**Files:**
- Create: `results4/REPORT4.md`
- Modify: `README.md` (rounds table, setup commands, layout rows, test count)
- Modify: memory file `goal-occupancy-project.md` (round 5 headline results)

- [ ] **Step 1: Full run in the background**

Run: `.venv/bin/python scripts/run_task4.py --jobs 8 2>&1 | tee results4/run_log.txt` (background; expect 40–90 min).
Expected: ~140 runs (more if the grid is triggered), `results4/tables4.md`, `results4/results4.json`, five figures.

- [ ] **Step 2: Sanity checks on the results before writing**

- Base condition P_metric at init ≈ 1.3 (round 4 found 1.34) and final near 0.85 (round 4 found 0.84); belief R² ≈ 1.
- KL of every run with λ ≥ 0.5 below 0.02 nats (Control 5); list runs that failed to learn.
- r = 0 and δ = 0: branch accuracy at t*−1 and P_metric versus init (Control 2).
- Matched-λ delay runs: cost column roughly constant across k.

- [ ] **Step 3: Write `results4/REPORT4.md`**

Sections: setup (HMM family, model, run date, wall time), the five sweeps with the table columns that matter (P_metric, P_metric_pre, belief R², branch accuracy, ∫G_∥, cost, KL), the regression and leave-one-sweep-out numbers, the dynamics ordering, controls 1–5, the reading against the eight success criteria and outcomes A–D of TASK4.md, and files.

- [ ] **Step 4: Update README.md and memory**

Add a round 5 row (TASK4.md → spec, `scripts/run_task4.py --jobs 8`, `results4/REPORT4.md`), the setup command, layout rows for `hmm4.py`, `prominence.py`, `plotting4.py`, and the new test count. Update the memory file with the headline results and how to rerun.

- [ ] **Step 5: Final verification**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. Confirm every file listed in Global Constraints exists in `results4/`.
