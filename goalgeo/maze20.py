"""A hand-drawn 20 x 20 aliased maze designed so that the goal and the location must be combined non-additively.

Layout as text: `#` wall, `.` open, letters A.. goals (in alphabetical order), `*` landmarks. Everything else is as in
bigmaze: deterministic moves N, S, W, E (a move into a wall leaves the agent in place), a goal shown with the first
symbol, random spawn on any cell that is no goal, gamma^(t - 1) for entering the goal at the t-th move.

Observations are what the local walls look like, not random labels: two sibling pairs of noisy symbols,
    0 straight N-S corridor      1 straight E-W corridor        (siblings: confused with probability eps)
    2 corner or dead end         3 junction (three or more exits) or open floor
so every N-S corridor is aliased with every other, and landmark cells (`*`) emit a unique symbol without noise. The
agent therefore knows its local shape and must carry the landmarks it passed to know which corridor it is in.

The additive ceiling (no network): with the cell known, the best rule argmax_a H(cell, a) + G(goal, a) against the
shortest-path moves, fitted directly (several restarts, batched over goal sets on a device); the 2 x 2 state-goal
reversals (s1, g1) -> a, (s1, g2) -> b, (s2, g1) -> b, (s2, g2) -> a that no additive rule can satisfy; and the
hardest goal placement by local search over goal sets (the design criterion of this maze).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from goalgeo import bigmaze as BM, mazebelief as MB

GOAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# PLACEHOLDER: the layout under search
ROWS = [
        ".E.........#.....#.#",
        "######.#.###.#.#.#.#",
        "...D.#.#...#.#.#...#",
        ".###.#.#...#.#.###.#",
        "A.C#.#.#.#B#.#...#.#",
        ".###.#.###.#.###.#.#",
        ".#...#.#...#.#...#.#",
        ".#.###.#.###.#G#.#.#",
        ".#...#.#.F.#.#H#...#",
        "####.#.#.#.#.#.#####",
        ".....#.#.#...#.#...#",
        ".#####.###.###.#.#.#",
        ".....#...#.#...#.#.#",
        ".###.###.###.##..#.#",
        ".#.....#...#.....#.#",
        "##.#######.#.#####.#",
        "...#.......#...#...#",
        ".###.###########.#.#",
        ".................#..",
        ".###############.###"
    ]


def from_rows(rows, eps=0.2, gamma=0.97, H=60, structural=True, pairs=2, rng=None):
    """Build the maze from its text rows. structural: symbols by local shape (module docstring); otherwise random
    symbols from `pairs` sibling pairs as in bigmaze (rng required). Landmarks are the `*` cells."""
    assert len({len(r) for r in rows}) == 1, "ragged rows"
    cells, goals, land = [], {}, []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                continue
            cells.append((r, c))
            if ch in GOAL_LETTERS:
                goals[ch] = len(cells) - 1
            elif ch == "*":
                land.append(len(cells) - 1)
            else:
                assert ch == ".", f"unknown cell character {ch!r} at {(r, c)}"
    n = len(cells)
    m0 = MB.Maze(cells, np.zeros(n, int), (0,), eps, gamma, H)
    if structural:
        sym = np.array([shape_symbol(m0.nxt[s], s) for s in range(n)])
        pairs = 2
    else:
        sym = rng.integers(0, 2 * pairs, n)
    for j, i in enumerate(land):
        sym[i] = 2 * pairs + j
    g = tuple(goals[k] for k in sorted(goals))
    m = MB.Maze(cells, sym, g, eps, gamma, H)
    m.n_sym = 2 * pairs + len(land)
    m.E = np.zeros((n, m.n_sym))
    for s in range(n):
        if sym[s] < 2 * pairs:
            m.E[s, sym[s]] = 1 - eps; m.E[s, sym[s] ^ 1] = eps
        else:
            m.E[s, sym[s]] = 1
    m.landmarks = tuple(land)
    m.rows = list(rows)
    D = np.stack([BM.distances(m.nxt, s) for s in range(n)])
    assert np.isfinite(D).all(), "the maze is not connected"
    return m


def the_maze(**kw):
    """The maze of the maze20 experiment: ROWS with its goals and landmarks (from_rows keyword arguments)."""
    return from_rows(ROWS, **kw)


def shape_symbol(nxt_s, s):
    """0 N-S corridor, 1 E-W corridor, 2 corner or dead end, 3 junction or open floor (MOVES order N, S, W, E)."""
    o = [int(nxt_s[a]) != s for a in range(4)]
    k = sum(o)
    if k >= 3:
        return 3
    if k == 2 and o[0] and o[1]:
        return 0
    if k == 2 and o[2] and o[3]:
        return 1
    return 2


def draw(maze, overlay=None):
    """The layout; goals as letters, landmarks `*`, other cells their symbol (or `overlay[cell]` as a digit)."""
    R, C = max(r for r, _ in maze.cells) + 1, max(c for _, c in maze.cells) + 1
    ix = {c: i for i, c in enumerate(maze.cells)}
    out = []
    for r in range(R):
        s = ""
        for c in range(C):
            if (r, c) not in ix:
                s += "#"
            elif ix[(r, c)] in maze.goals:
                s += GOAL_LETTERS[maze.goals.index(ix[(r, c)])]
            elif ix[(r, c)] in maze.landmarks and overlay is None:
                s += "*"
            elif overlay is not None:
                v = int(overlay[ix[(r, c)]]); s += "." if v == 0 else str(min(v, 9))
            else:
                s += str(int(maze.sym[ix[(r, c)]]))
        out.append(s)
    return "\n".join(out)


# ------------------------------------------------------------------ the additive ceiling

def all_distances(maze):
    return np.stack([BM.distances(maze.nxt, s) for s in range(maze.n)])                   # [n, n] moves from s to t


def optimal_moves(maze, goals, D=None):
    """[K, n, 4] shortest-path moves (ties kept), and [K, n] the decisions to score (not on the goal itself)."""
    D = all_distances(maze) if D is None else D
    dn = D[np.asarray(goals)][:, maze.nxt]
    opt = dn <= dn.min(-1, keepdims=True) + 1e-9
    keep = np.ones((len(goals), maze.n), bool)
    for k, g in enumerate(goals):
        keep[k, g] = False
    return opt, keep


def ceiling(maze, goal_sets, D=None, restarts=4, steps=2000, additive=True, device="cuda", lr=0.05, refine_rounds=0):
    """Best additive rule for each goal set [B, K] (additive=False: the goal-blind rule argmax H(cell)). Returns the
    share of decisions with a shortest-path move [B] and, per set, the errors per cell summed over goals [B, n].
    The fit is a lower bound on the true maximum; restarts and steps are what the additive-ceiling experiment used."""
    D = all_distances(maze) if D is None else D
    gs = np.asarray(goal_sets); B, K = gs.shape; n = maze.n
    opt = np.stack([optimal_moves(maze, g, D)[0] for g in gs]); keep = np.stack([optimal_moves(maze, g, D)[1] for g in gs])
    opt, keep = torch.tensor(opt, device=device), torch.tensor(keep, device=device)
    best = torch.zeros(B, device=device); best_err = torch.zeros(B, n, device=device)
    bestH, bestG = torch.zeros(B, n, 4, device=device), torch.zeros(B, K, 4, device=device)
    for r in range(restarts):
        torch.manual_seed(r)
        Hc = torch.zeros(B, n, 4, device=device, requires_grad=True)
        G = (0.1 * torch.randn(B, K, 4, device=device)).requires_grad_(additive)
        o = torch.optim.Adam([Hc, G] if additive else [Hc], lr=lr)
        for i in range(steps):
            l = Hc[:, None] + (G[:, :, None] if additive else 0)
            lp = torch.log_softmax(l * (1 + i / 300), -1)                                   # sharpen: the target is the argmax
            loss = -(torch.logsumexp(lp.masked_fill(~opt, -1e9), -1))[keep].mean() * B
            o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            a = (Hc[:, None] + (G[:, :, None] if additive else 0)).argmax(-1)
            ok = opt.gather(3, a[..., None]).squeeze(-1)
            acc = (ok & keep).sum((1, 2)).float() / keep.sum((1, 2))
            err = (~ok & keep).sum(1).float()
            imp = acc > best
            best = torch.where(imp, acc, best); best_err[imp] = err[imp]
            bestH[imp], bestG[imp] = Hc.detach()[imp], (G.detach()[imp] if additive else 0)
    if refine_rounds and additive:
        for b in range(B):
            a, Hb, Gb = refine(opt[b], keep[b], bestH[b], bestG[b], rounds=refine_rounds)
            if a > best[b]:
                best[b] = a
                ab = (Hb[None] + Gb[:, None]).argmax(-1)
                best_err[b] = (~opt[b].gather(2, ab[..., None]).squeeze(-1) & keep[b]).sum(0).float()
    return best.cpu().numpy(), best_err.cpu().numpy()


def refine(opt, keep, Hc, G, rounds=6, cand=512, scales=(1.0, 0.3, 0.1), seed=0):
    """Coordinate ascent on the exact count after the gradient fit: for every cell, `cand` random perturbations of its
    H row at each scale (G fixed), keep the best; then the same for every goal's G row (H fixed). opt [K, n, 4] bool,
    keep [K, n] bool, Hc [n, 4], G [K, 4] tensors on one device. Returns (accuracy, Hc, G)."""
    gen = torch.Generator(device=Hc.device); gen.manual_seed(seed)
    K, n, _ = opt.shape
    def acc_of(Hc, G):
        a = (Hc[None] + G[:, None]).argmax(-1)
        return (opt.gather(2, a[..., None]).squeeze(-1) & keep).sum().item() / keep.sum().item()
    for _ in range(rounds):
        for sc in scales:                                                            # cells
            C = Hc[:, None] + sc * torch.randn(n, cand, 4, device=Hc.device, generator=gen)      # [n, cand, 4]
            C = torch.cat([Hc[:, None], C], 1)
            a = (C[None] + G[:, None, None]).argmax(-1)                                          # [K, n, cand+1]
            ok = (opt.gather(2, a.reshape(K, n, -1)).reshape(K, n, -1) & keep[..., None]).sum(0)  # [n, cand+1]
            Hc = C[torch.arange(n), ok.argmax(1)]
        for sc in scales:                                                            # goals
            C = G[:, None] + sc * torch.randn(K, cand, 4, device=G.device, generator=gen)
            C = torch.cat([G[:, None], C], 1)                                                    # [K, cand+1, 4]
            a = (Hc[None, None] + C[:, :, None]).argmax(-1)                                      # [K, cand+1, n]
            ok = (opt[:, None].expand(K, cand + 1, n, 4).gather(3, a[..., None]).squeeze(-1) & keep[:, None]).sum(2)
            G = C[torch.arange(K), ok.argmax(1)]
    return acc_of(Hc, G), Hc, G


def ceiling_exact(maze, goals, D=None, time_limit=600, big=40.0, margin=1.0, verbose=False):
    """The additive ceiling by mixed-integer programming (HiGHS): maximise the number of (cell, goal) decisions whose
    argmax of H(cell, a) + G(goal, a) is a shortest-path move. One binary per (cell, goal, shortest-path move) selects
    which optimal move wins by `margin` (the scale is free: H and G are bounded by big / 4). Returns the exact value
    when the solver finishes (incumbent == upper), otherwise the incumbent (a lower bound) and the dual bound (an upper
    bound) at the time limit, with the per-cell errors of the incumbent and the fitted H and G."""
    D = all_distances(maze) if D is None else D
    opt, keep = optimal_moves(maze, goals, D)
    n, K = maze.n, len(goals)
    hi = lambda s, a: s * 4 + a
    gi = lambda g, a: n * 4 + g * 4 + a
    ys = [(s, g, a) for g in range(K) for s in range(n) if keep[g, s] for a in range(4) if opt[g, s, a]]
    y0 = n * 4 + K * 4
    yi = {t: y0 + i for i, t in enumerate(ys)}
    rows, cols, vals, lo, up = [], [], [], [], []
    r = 0
    by_sg = {}
    for (s, g, a) in ys:
        by_sg.setdefault((s, g), []).append(a)
    for (s, g), A in by_sg.items():                                                  # at most one winning move per decision
        if len(A) > 1:
            for a in A:
                rows.append(r); cols.append(yi[(s, g, a)]); vals.append(1.0)
            lo.append(-np.inf); up.append(1.0); r += 1
    for (s, g, a) in ys:                                                             # y = 1 => a beats every b by margin
        for b in range(4):
            if b != a:
                rows += [r] * 5; cols += [hi(s, a), hi(s, b), gi(g, a), gi(g, b), yi[(s, g, a)]]
                vals += [1.0, -1.0, 1.0, -1.0, -big]; lo.append(margin - big); up.append(np.inf); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, y0 + len(ys)))
    c = np.zeros(y0 + len(ys)); c[y0:] = -1.0
    integrality = np.zeros(y0 + len(ys)); integrality[y0:] = 1
    bounds = Bounds(np.r_[np.full(y0, -big / 4), np.zeros(len(ys))], np.r_[np.full(y0, big / 4), np.ones(len(ys))])
    t = time.time()
    res = milp(c, constraints=LinearConstraint(A, lo, up), integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, disp=verbose, mip_rel_gap=1e-6))
    total = int(keep.sum())
    out = dict(seconds=time.time() - t, status=res.message, decisions=total, incumbent=None, upper=None)
    if res.x is not None:
        Hc, G = res.x[: n * 4].reshape(n, 4), res.x[n * 4: y0].reshape(K, 4)
        a = (Hc[None] + G[:, None]).argmax(-1)
        ok = np.take_along_axis(opt, a[..., None], 2).squeeze(-1)
        out.update(incumbent=float((ok & keep).sum() / total), errors=(~ok & keep).sum(0), H=Hc, G=G)
    if getattr(res, "mip_dual_bound", None) is not None:
        out["upper"] = float(min(1.0, -res.mip_dual_bound / total))
    out["exact"] = out["incumbent"] is not None and out["upper"] is not None and out["upper"] - out["incumbent"] < 0.5 / total
    return out


def ceiling_exact_blind(maze, goals, D=None):
    """The goal-blind ceiling, exactly: per cell, the move that is a shortest-path move for the most goals."""
    opt, keep = optimal_moves(maze, goals, D)
    return float((opt & keep[..., None]).sum(0).max(-1).sum() / keep.sum())


def reversal_examples(maze, goals, D=None):
    """The 2 x 2 reversal patterns as a list of dicts, largest first: goals g1 < g2, moves a < b (as letters N S W E),
    the cells needing (a at g1, b at g2) and those needing (b, a), with one example cell (row, col) of each."""
    opt, _ = optimal_moves(maze, goals, D)
    uniq, act = opt.sum(-1) == 1, opt.argmax(-1)
    K, out = len(goals), []
    for i in range(K):
        for j in range(i + 1, K):
            S = np.nonzero(uniq[i] & uniq[j] & (act[i] != act[j]))[0]
            by = {}
            for s in S:
                by.setdefault((int(act[i, s]), int(act[j, s])), []).append(int(s))
            for (a, b), L in by.items():
                if a < b and (b, a) in by:
                    out.append(dict(g1=i, g2=j, a="NSWE"[a], b="NSWE"[b], n_ab=len(L), n_ba=len(by[(b, a)]),
                                    cells_ab=L, cells_ba=by[(b, a)], cell_ab=maze.cells[L[0]], cell_ba=maze.cells[by[(b, a)][0]]))
    return sorted(out, key=lambda e: -e["n_ab"] * e["n_ba"])


def reversals(maze, goals, D=None):
    """2 x 2 state-goal reversals among decisions with a unique shortest-path move: the number of (s1, s2, g1, g2)
    patterns and the number of cells in at least one."""
    opt, _ = optimal_moves(maze, goals, D)
    uniq, act = opt.sum(-1) == 1, opt.argmax(-1)
    K = len(goals); cnt, cells = 0, set()
    for i in range(K):
        for j in range(i + 1, K):
            S = np.nonzero(uniq[i] & uniq[j] & (act[i] != act[j]))[0]
            by = {}
            for s in S:
                by.setdefault((act[i, s], act[j, s]), []).append(int(s))
            for (a, b), L in by.items():
                if a < b and (b, a) in by:
                    cnt += len(L) * len(by[(b, a)]); cells |= set(L) | set(by[(b, a)])
    return cnt, len(cells)


def hardest_goals(maze, K, rounds=8, pop=256, keep_top=16, seed=0, D=None, candidates=None, verbose=False, **fit):
    """The goal set with the lowest additive ceiling: random sets, then rounds of local moves (one or two goals
    replaced) around the best sets so far. Returns (goals, ceiling, every set evaluated)."""
    rng = np.random.default_rng(seed)
    D = all_distances(maze) if D is None else D
    cand = np.arange(maze.n) if candidates is None else np.asarray(candidates)
    gs = np.stack([rng.choice(cand, K, replace=False) for _ in range(pop)])
    acc, _ = ceiling(maze, gs, D, **fit)
    seen = {tuple(sorted(int(x) for x in g)): float(a) for g, a in zip(gs, acc)}
    for rd in range(rounds):
        top = sorted(seen.items(), key=lambda kv: kv[1])[:keep_top]
        new = []
        while len(new) < pop:
            g = list(top[rng.integers(len(top))][0])
            for _ in range(rng.integers(1, 3)):
                g[rng.integers(K)] = int(rng.choice(cand))
            if len(set(g)) == K and tuple(sorted(g)) not in seen:
                new.append(g)
        acc, _ = ceiling(maze, np.array(new), D, **fit)
        seen.update({tuple(sorted(g)): float(a) for g, a in zip(new, acc)})
        if verbose:
            print(f"  round {rd}: hardest {min(seen.values()):.3f} over {len(seen)} goal sets", flush=True)
    g, a = min(seen.items(), key=lambda kv: kv[1])
    return list(g), a, seen
