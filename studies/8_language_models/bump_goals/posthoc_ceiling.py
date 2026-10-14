"""Post hoc (after the registered ADD-ON result was seen): can any additive rule argmax h[goal-free set] + g[goal, step]
do well online in the four-goal bump maze? The exact additive ceiling rule and the goal-blind rule are fitted on the
optimum's decisions (as in design.py, with the fitted tables kept), scored offline by goal-free set size, and run in
the environment under every goal on the test starts; then refitted on their own online decisions as well (three
rounds, so sets the optimum never meets are covered), and run again. An unseen set falls back to argmax g[goal, step].

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_goals/posthoc_ceiling.py
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze")); sys.path.insert(0, str(HERE))
from design import worlds_and_solutions, decisions_all_goals, K, N_EP                 # noqa: E402
from train_goals import test_starts                                                   # noqa: E402

N_ONLINE, ROUNDS, N_ROLL = 1024, 3, 1024
SIZES = (("1", 1, 1), ("2-4", 2, 4), ("5-16", 5, 16), (">16", 17, 10 ** 9))


def fit(flat, K, H, use_h, use_g, device, steps=3000, lr=0.05, seed=0):
    """flat: list of (key, goal, step, optimal-set bool [3]). Returns the key index, h [sets, 3], g [K, H, 3] and the
    share of the fitted decisions whose argmax is optimal (as goalgeo.bumpmaze.additive_ceiling, tables kept)."""
    keys = {}; ks = torch.tensor([keys.setdefault(k, len(keys)) for k, _, _, _ in flat], device=device)
    gs = torch.tensor([g for _, g, _, _ in flat], device=device); st = torch.tensor([min(t, H - 1) for _, _, t, _ in flat], device=device)
    opt = torch.tensor(np.array([o for _, _, _, o in flat]), device=device)
    torch.manual_seed(seed)
    h = torch.zeros(len(keys), 3, device=device, requires_grad=True); g = torch.zeros(K, H, 3, device=device, requires_grad=True)
    o_ = torch.optim.Adam([h, g], lr=lr)
    for _ in range(steps):
        z = (h[ks] if use_h else 0) + (g[gs, st] if use_g else 0)
        loss = -(torch.logsumexp(torch.log_softmax(z, -1).masked_fill(~opt, -1e9), -1)).mean()
        o_.zero_grad(); loss.backward(); o_.step()
    with torch.no_grad():
        z = (h[ks] if use_h else 0) + (g[gs, st] if use_g else 0); ok = opt.gather(1, z.argmax(1, keepdim=True)).squeeze(1)
    return dict(keys=keys, h=h.detach().cpu() if use_h else None, g=g.detach().cpu() if use_g else None, optimal=float(ok.double().mean()), ok=ok.cpu(), ks=ks.cpu())


def rule_actions(rule, keys_now, goal, step, H):
    """Actions of a fitted rule for a batch of goal-free set keys under one goal at one step; the unseen flag."""
    idx = torch.tensor([rule["keys"].get(k, -1) for k in keys_now]); z = torch.zeros(len(keys_now), 3, dtype=torch.float64)
    if rule["h"] is not None: z[idx >= 0] += rule["h"][idx[idx >= 0]].double()
    if rule["g"] is not None: z += rule["g"][goal, min(step, H - 1)].double()
    return z.argmax(1), (idx < 0)


def run_online(rule, worlds, sols, starts_fn, record=False, seed=0):
    """The rule in the environment under every goal; return / success per goal, the share of decisions at unseen
    sets; optionally the decisions (key, goal, step, optimal set under the goal) for refitting."""
    H = worlds[0].H; key = lambda m: np.packbits(m).tobytes(); rows, dec, unseen, n_dec = [], [], 0, 0
    for gi, (w, sol) in enumerate(zip(worlds, sols)):
        starts = starts_fn(w, gi); n = len(starts); e = BMZ.Env(w, n, torch.Generator().manual_seed(seed), start=starts); ret = torch.zeros(n)
        common = BMZ.Filter(w, n, remove_goal=False)
        for t in range(H):
            alive = torch.nonzero(~e.done).squeeze(1)
            if len(alive) == 0: break
            cm = common.mask.numpy(); ks = [key(cm[i]) for i in alive.tolist()]
            a_al, un = rule_actions(rule, ks, gi, t, H); unseen += int(un.sum()); n_dec += len(alive)
            if record:
                masks = e.filter.mask.numpy()
                for j, i in enumerate(alive.tolist()):
                    q = sol["Q"][sol["sets"][key(masks[i])]]; dec.append((ks[j], gi, t, q >= q.max() - 1e-9))
            act = torch.zeros(n, dtype=torch.long); act[alive] = a_al; r, obs = e.step(act); ret += r; common.update(act, obs)
        rows.append(dict(ret=float(ret.mean()), success=float((ret > 0).float().mean())))
    return dict(per_goal=rows, ret=float(np.mean([r["ret"] for r in rows])), success=float(np.mean([r["success"] for r in rows])), unseen=unseen / max(n_dec, 1)), dec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=ROUNDS); a = ap.parse_args(); rounds = a.rounds
    t0 = time.time(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    free, goals, worlds, sols = worlds_and_solutions(); H = worlds[0].H; design = json.loads((HERE / "design.json").read_text())
    opt_ret = float(np.mean([r["optimal"]["ret"] for r in design["per_goal"]]))
    dec0 = decisions_all_goals(worlds, sols, N_EP, 100); flat0 = [(d["key"], g2, d["step"], o) for d in dec0 for g2, o in d["opts"].items()]
    size0 = np.array([int(np.unpackbits(np.frombuffer(k, dtype=np.uint8)).sum()) for k, _, _, _ in flat0])
    print(f"{len(flat0)} decision-goal pairs from the optimum {time.time() - t0:.0f}s", flush=True)
    res = dict(n_pairs_optimum=len(flat0), optimum_ret=opt_ret, rounds=[])
    test_fn = lambda w, gi: test_starts(w, gi)[:N_ONLINE]
    flat = {"additive": list(flat0), "goal_blind": list(flat0)}
    for rnd in range(rounds + 1):
        row = dict(round=rnd, n_pairs={k: len(v) for k, v in flat.items()})
        for name, use_g in (("additive", True), ("goal_blind", False)):
            rule = fit(flat[name], K, H, True, use_g, dev)
            # offline on the optimum's decisions (the ceiling's own measure), by goal-free set size
            ok0 = rule["ok"][: len(flat0)].numpy()
            row[name] = dict(offline_optimum=float(ok0.mean()), offline_by_size={b: float(ok0[(size0 >= lo) & (size0 <= hi)].mean()) for b, lo, hi in SIZES}, offline_all_fitted=rule["optimal"], sets=len(rule["keys"]))
            online, _ = run_online(rule, worlds, sols, test_fn); row[name]["online"] = online
            if rnd < rounds:
                _, own = run_online(rule, worlds, sols, lambda w, gi: torch.as_tensor(w.start_states)[torch.randint(len(w.start_states), (N_ROLL,), generator=torch.Generator().manual_seed(1000 + 10 * rnd + gi))], record=True, seed=rnd + 1)
                flat[name] = flat[name] + own
        hb = fit(flat0, K, H, False, True, dev) if rnd == 0 else None
        if hb is not None: row["history_blind"] = dict(offline_optimum=hb["optimal"], online=run_online(hb, worlds, sols, test_fn)[0])
        row["recovery"] = (row["additive"]["online"]["ret"] - row["goal_blind"]["online"]["ret"]) / max(opt_ret - row["goal_blind"]["online"]["ret"], 1e-9)
        res["rounds"].append(row)
        print(f"round {rnd}: additive offline {row['additive']['offline_optimum']:.3f} (by size " + " / ".join(f"{row['additive']['offline_by_size'][b]:.2f}" for b, _, _ in SIZES) + f"; {row['additive']['sets']} sets) online return {row['additive']['online']['ret']:.3f} success {row['additive']['online']['success']:.3f} unseen {row['additive']['online']['unseen']:.2f}; "
              f"goal-blind offline {row['goal_blind']['offline_optimum']:.3f} online {row['goal_blind']['online']['ret']:.3f} / {row['goal_blind']['online']['success']:.3f}; recovery {row['recovery']:.3f} {time.time() - t0:.0f}s", flush=True)
    (HERE / "posthoc_ceiling.json").write_text(json.dumps(res, indent=1))
    L = ["# Post hoc: the exact additive ceiling rule online", "", f"Generated by `posthoc_ceiling.py` (after the registered ADD-ON result was seen). The best additive rule argmax h[goal-free set] + g[goal, step] and the best goal-blind rule argmax h[set], fitted by gradient on the optimum's decisions under every goal ({len(flat0)} decision-goal pairs, as in `design.md`), run in the environment on the test starts ({N_ONLINE} per goal) under every goal; then refitted on their own online decisions as well ({N_ROLL} episodes per goal per round, {rounds} rounds) and run again. An unseen set falls back to argmax g[goal, step]. Recovery: (additive − goal-blind) / (optimum − goal-blind) online; the optimum returns {opt_ret:.3f}.", "",
         "| fit | rule | sets | offline optimal on the optimum's decisions | by set size 1 / 2–4 / 5–16 / > 16 | online return | success | decisions at unseen sets |", "|---|---|---|---|---|---|---|---|"]
    for row in res["rounds"]:
        for name in ("additive", "goal_blind"):
            r = row[name]; L.append(f"| {'optimum\'s decisions' if row['round'] == 0 else f'+ own decisions, round {row['round']}'} | {name.replace('_', '-')} | {r['sets']} | {r['offline_optimum']:.3f} | " + " / ".join(f"{r['offline_by_size'][b]:.2f}" for b, _, _ in SIZES) + f" | {r['online']['ret']:.3f} | {r['online']['success']:.3f} | {r['online']['unseen']:.2f} |")
        if "history_blind" in row: L.append(f"| optimum's decisions | history-blind g[goal, step] | | {row['history_blind']['offline_optimum']:.3f} | | {row['history_blind']['online']['ret']:.3f} | {row['history_blind']['online']['success']:.3f} | |")
    L += ["", "Recovery of the goal-directed return by the exact additive rule, by round: " + ", ".join(f"{row['recovery']:.3f}" for row in res["rounds"]) + "."]
    (HERE / "posthoc_ceiling.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
