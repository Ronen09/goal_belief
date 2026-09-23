"""TASK3: the supervision bottleneck. Writes rounds/r03_supervision/.

    .venv/bin/python rounds/r03_supervision/run.py            # full (~30 min CPU)
    .venv/bin/python rounds/r03_supervision/run.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from goalgeo import analysis as A, geometry as G, quotient as Qt, supervision as Sv, targets as T, models2 as M
import plots as P3  # noqa: E402
from goalgeo.envs import make_env, make_corridor_env, make_ring_env
from goalgeo.model import PolicyNet
from goalgeo.occupancy import goal_occupancy
from goalgeo.planning import solve_all_goals

GAMMA = 0.9
L3 = ["h1", "h2", "h3"]
CKPTS = [0, 10, 30, 100, 300, 1000, 3000]


def dump(obj, path):
    Path(path).write_text(json.dumps(A.to_jsonable(obj), indent=1))


def cname(c):
    return "hard" if c[0] == "hard" else (f"{c[0]}_tau{c[1]:g}" if c[1] is not None else c[0])


def rows_of(tg, S, K):
    return tg.s * K + tg.g


def fit(env, sol, occ, kind, tau, seed, steps, arch="mlp3", loss=None, ckpts=None, rows=None, encoding="onehot"):
    tg = T.build_targets(env, sol, occ, kind, tau)
    if arch == "mlp3" and encoding == "coord":
        net = PolicyNet(env, encoding="coord", seed=seed, out_dim=tg.out_dim); net.hidden_layers = L3
    else:
        net = M.make_model(arch, env, out_dim=tg.out_dim, seed=seed)
    hist = T.train_targets(net, tg, steps=steps, loss=loss, checkpoint_steps=ckpts, seed=seed, rows=rows)
    return net, tg, hist


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # NaN means "not defined" for that condition (e.g. accuracy of a regression head)
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="rounds/r03_supervision")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    TAUS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5] if not args.quick else [0.02, 0.1, 0.5, 2]
    conds = [("hard", None)] + [("boltzmann", t) for t in TAUS]
    seeds = [0, 1, 2, 3, 4] if not args.quick else [0, 1]
    seeds3 = seeds[:3]
    steps = 3000 if not args.quick else 300
    ckpts = [c for c in CKPTS if c <= steps]
    R = {"conds": [cname(c) for c in conds]}

    env = make_env("base"); sol = solve_all_goals(env, GAMMA); occ = goal_occupancy(env, sol.pi, GAMMA)
    S, K = env.n_states, env.n_goals
    hard_tg = T.build_targets(env, sol, occ, "hard")
    rng = np.random.default_rng(0)
    pair_rows = np.sort(rng.choice(len(hard_tg.s), min(1000, len(hard_tg.s)), replace=False))
    classes = Sv.hard_classes(hard_tg)

    # ------------------------------------------------------------- Task 1: temperature sweep (+ checkpoints for Task 7)
    nets, hists, tgs, acts = {}, {}, {}, {}
    rep1 = {}
    for c in conds:
        rep1[c] = []
        for seed in seeds:
            net, tg, hist = fit(env, sol, occ, c[0], c[1], seed, steps, ckpts=ckpts)
            nets[(c, seed)], hists[(c, seed)], tgs[c] = net, hist, tg
            acts[(c, seed)] = net.all_activations()
            rep1[c].append(Sv.geometry_report(acts[(c, seed)], L3, env, occ, sol, tg, GAMMA, pair_rows))
            rep1[c][-1]["accuracy"] = T.argmax_accuracy(net, tg)
        r = rep1[c]
        print(f"[task1] {cname(c):18s} acc={np.mean([x['accuracy'] for x in r]):.3f} h3 RSA target={np.mean([x['h3']['rsa_target'] for x in r]):.2f} policy={np.mean([x['h3']['rsa_policy'] for x in r]):.2f} "
              f"occ={np.mean([x['h3']['rsa_occupancy'] for x in r]):.2f} adv={np.mean([x['h3']['rsa_advantage'] for x in r]):.2f} PR={np.mean([x['h3']['participation_ratio'] for x in r]):.2f}  {time.time()-t0:.0f}s")
    rand_acts = {seed: PolicyNet(env, seed=100 + seed).all_activations() for seed in seeds3}
    P3.plot_temperature_sweep(conds, rep1, out / "task1_temperature_sweep.png")
    P3.plot_target_geometry(conds, rep1, out / "task2_target_geometry.png")
    R["task1_2"] = {cname(c): rep1[c] for c in conds}

    # ------------------------------------------------------------- Task 3: fixed-argmax counterfactual
    natural = {c: {l: [] for l in L3} for c in conds}
    sa_pairs = Sv.same_argmax_pairs(hard_tg, sol)
    for c in conds:
        for seed in seeds:
            for l in L3:
                Hf = acts[(c, seed)][l].reshape(S * K, -1)[rows_of(tgs[c], S, K)]
                natural[c][l].append(Sv.counterfactual_report(Hf, sa_pairs, tgs[c]))
    margins = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    task = M.SyntheticTask(n_background=200, margins=margins, seed=0)
    synth = {c: {l: [] for l in L3} for c in conds}

    def synth_run(arch, c, seed):
        Y = T.targets_from_Q(task.Q, c[0], c[1])
        s_idx, g_idx = task.rows()
        tg = T.Targets(c[0], c[1], s_idx, g_idx, Y, Y.shape[1], "ce", task.Q)
        net = M.make_model(arch, task, out_dim=4, seed=seed)
        T.train_targets(net, tg, steps=steps, seed=seed)
        H = net.all_activations()
        res = {}
        for l in net.hidden_layers:
            X = H[l][:, 0, :]
            bg = X[task.background]; a, b = rng.integers(0, len(bg), 3000), rng.integers(0, len(bg), 3000); keep = a != b
            med = np.median(np.linalg.norm(bg[a[keep]] - bg[b[keep]], axis=1))
            res[l] = np.linalg.norm(X[task.probes] - X[task.reference][None], axis=1) / med
        return res, net.hidden_layers
    for c in conds:
        for seed in seeds3:
            res, _ = synth_run("mlp3", c, seed)
            for l in L3:
                synth[c][l].append(res[l])
        print(f"[task3] {cname(c):18s} synthetic d_h(probe)/median at h3 for |Δm| = {np.round(np.abs(np.array(margins)-0.5),2).tolist()}: {np.round(np.mean(synth[c]['h3'],0),2).tolist()} | natural low/high |Δmargin|: {np.mean([r['low_dmargin_dist'] for r in natural[c]['h3']]):.2f}/{np.mean([r['high_dmargin_dist'] for r in natural[c]['h3']]):.2f}")
    P3.plot_counterfactual(synth, natural, conds, margins, out / "task3_counterfactual.png")
    R["task3"] = {"margins": margins, "synthetic": {cname(c): {l: np.mean(v, 0).tolist() for l, v in synth[c].items()} for c in conds},
                  "natural": {cname(c): natural[c] for c in conds}}

    # ------------------------------------------------------------- Task 4: fixed environment, different T(Q)
    conds4 = [("hard", None), ("boltzmann", 0.05), ("boltzmann", 0.5), ("advantage", None), ("q", None), ("occupancy", None)]
    rep4, acts4 = {}, {}
    for c in conds4:
        n4 = cname(c); rep4[n4] = []
        for seed in seeds3:
            if c in conds and (c, seed) in acts:
                H, tg = acts[(c, seed)], tgs[c]; net = nets[(c, seed)]
            else:
                net, tg, _ = fit(env, sol, occ, c[0], c[1], seed, steps); H = net.all_activations()
            acts4[(n4, seed)] = H
            r = Sv.geometry_report(H, L3, env, occ, sol, tg, GAMMA, pair_rows); r["accuracy"] = T.argmax_accuracy(net, tg)
            rep4[n4].append(r)
        print(f"[task4] {n4:18s} acc={np.nanmean([x['accuracy'] for x in rep4[n4]]):.3f} h3 RSA target={np.mean([x['h3']['rsa_target'] for x in rep4[n4]]):.2f} policy={np.mean([x['h3']['rsa_policy'] for x in rep4[n4]]):.2f} occ={np.mean([x['h3']['rsa_occupancy'] for x in rep4[n4]]):.2f} adv={np.mean([x['h3']['rsa_advantage'] for x in rep4[n4]]):.2f}  {time.time()-t0:.0f}s")
    names4 = [cname(c) for c in conds4]
    cross4 = {}
    for a in names4:
        for b in names4:
            vals = []
            for sa in seeds3:
                for sb in seeds3:
                    if a == b and sa == sb:
                        continue
                    vals.append(G.rsa(G.rdm(acts4[(a, sa)]["h3"].mean(1)), G.rdm(acts4[(b, sb)]["h3"].mean(1))))
            cross4[(a, b)] = float(np.mean(vals))
    P3.plot_fixed_env(names4, cross4, rep4, out / "task4_fixed_env.png")
    R["task4"] = {"reports": rep4, "cross_rsa": {f"{a}|{b}": v for (a, b), v in cross4.items()}}

    # ------------------------------------------------------------- Task 5: information bottleneck
    stats5, hidden_pr = {}, {}
    for c in conds4:
        n5 = cname(c); tg = T.build_targets(env, sol, occ, c[0], c[1])
        stats5[n5] = Sv.bottleneck_stats(tg, sol, occ)
        hidden_pr[n5] = float(np.mean([G.participation_ratio(acts4[(n5, s)]["h3"].reshape(S * K, -1)[rows_of(tg, S, K)]) for s in seeds3]))
    for c in conds:
        if c[0] == "boltzmann" and cname(c) not in stats5:
            stats5[cname(c)] = Sv.bottleneck_stats(tgs[c], sol, occ)
            hidden_pr[cname(c)] = float(np.mean([G.participation_ratio(acts[(c, s)]["h3"].reshape(S * K, -1)[rows_of(tgs[c], S, K)]) for s in seeds3]))
    names5 = list(stats5)
    P3.plot_bottleneck(names5, stats5, hidden_pr, out / "task5_bottleneck.png")
    R["task5"] = {"stats": stats5, "hidden_pr_h3": hidden_pr}
    for n5 in names5:
        s5 = stats5[n5]
        print(f"[task5] {n5:18s} classes={s5['n_classes']:4d} rank={s5['rank']:3d} PR={s5['participation_ratio']:.2f} rsa(Q)={s5['rsa_with_Q']:.2f} recover Q/adv/occ R²={s5['recover_Q_r2']:.2f}/{s5['recover_adv_r2']:.2f}/{s5['recover_occ_r2']:.2f} hidden PR={hidden_pr[n5]:.2f}")

    # ------------------------------------------------------------- Task 6/7: sufficient statistic, compression during training
    conds7 = [("hard", None), ("boltzmann", 0.05), ("boltzmann", 0.5)]
    conds7 = [c for c in conds7 if c in conds] or conds[:3]
    rep6 = {}
    for c in conds:
        pairs = Sv.sufficient_statistic_pairs(tgs[c], occ)
        rep6[cname(c)] = {l: [Sv.pair_distances(acts[(c, s)][l].reshape(S * K, -1)[rows_of(tgs[c], S, K)], pairs) for s in seeds] for l in L3}
        rep6[cname(c)]["within_between_h3"] = [Sv.within_between(acts[(c, s)]["h3"].reshape(S * K, -1)[rows_of(tgs[c], S, K)], classes) for s in seeds]
    traj = {}
    for c in conds7:
        t = {k: [] for k in ("ratio", "within", "between", "rsa_target", "rsa_policy", "rsa_advantage", "rsa_occupancy", "participation_ratio")}
        Rs = Sv.state_level_rdms(env, occ, sol, tgs[c], GAMMA)
        for step in ckpts:
            vals = {k: [] for k in t}
            for seed in seeds3:
                probe = PolicyNet(env, seed=seed, out_dim=tgs[c].out_dim); probe.load_state_dict(hists[(c, seed)].checkpoints[step])
                H = probe.all_activations()["h3"]; Hf = H.reshape(S * K, -1)[rows_of(tgs[c], S, K)]
                wb = Sv.within_between(Hf, classes)
                vals["ratio"].append(wb["ratio"]); vals["within"].append(wb["within"]); vals["between"].append(wb["between"])
                Rh = G.rdm(H.mean(1))
                for k in ("target", "policy", "advantage", "occupancy"):
                    vals[f"rsa_{k}"].append(G.rsa(Rh, Rs[k]))
                vals["participation_ratio"].append(G.participation_ratio(H.mean(1)))
            for k in t:
                t[k].append(float(np.mean(vals[k])))
        traj[c] = t
        print(f"[task7] {cname(c):18s} within/between ratio over steps {ckpts}: {np.round(t['ratio'],2).tolist()}; between rel. to init: {np.round(np.array(t['between'])/t['between'][0],2).tolist()}")
    P3.plot_training(ckpts, traj, out / "task7_training.png")
    R["task6"] = rep6; R["task7"] = {"steps": ckpts, "traj": {cname(c): v for c, v in traj.items()}}

    # ------------------------------------------------------------- Task 8: generalisation
    rep8 = {}
    for ename in ("base", "twins"):
        e8 = make_env(ename); s8 = solve_all_goals(e8, GAMMA); o8 = goal_occupancy(e8, s8.pi, GAMMA)
        h8 = T.build_targets(e8, s8, o8, "hard"); cls8 = Sv.hard_classes(h8)
        n = len(h8.s); perm = np.random.default_rng(1).permutation(n); heldout = perm[: int(0.3 * n)]; train_rows = perm[int(0.3 * n):]
        for c in (("hard", None), ("boltzmann", 0.05)):
            key = f"{ename}/{cname(c)}"; rep8[key] = []
            for seed in seeds3:
                net, tg, _ = fit(e8, s8, o8, c[0], c[1], seed, steps, rows=train_rows)
                Hf = net.all_activations()["h3"].reshape(e8.n_states * e8.n_goals, -1)[rows_of(tg, e8.n_states, e8.n_goals)]
                cen = np.stack([Hf[train_rows][cls8[train_rows] == k].mean(0) for k in np.unique(cls8)])
                pred = np.unique(cls8)[np.argmin(np.linalg.norm(Hf[heldout][:, None] - cen[None], axis=-1), 1)]
                rep8[key].append({"acc_train": T.argmax_accuracy(net, tg, train_rows), "acc_heldout": T.argmax_accuracy(net, tg, heldout),
                                  "ratio_train": Sv.within_between(Hf[train_rows], cls8[train_rows])["ratio"], "ratio_heldout": Sv.within_between(Hf[heldout], cls8[heldout])["ratio"],
                                  "centroid_acc_heldout": float((pred == cls8[heldout]).mean()), "centroid_acc_chance": float(np.max(np.bincount(cls8[train_rows])) / len(train_rows))})
            r = rep8[key]
            print(f"[task8] {key:22s} acc train/heldout={np.mean([x['acc_train'] for x in r]):.3f}/{np.mean([x['acc_heldout'] for x in r]):.3f} ratio train/heldout={np.mean([x['ratio_train'] for x in r]):.2f}/{np.mean([x['ratio_heldout'] for x in r]):.2f} centroid acc={np.mean([x['centroid_acc_heldout'] for x in r]):.2f} (chance {r[0]['centroid_acc_chance']:.2f})")
    # corridor with coordinate input and unseen states
    ec = make_corridor_env(); sc = solve_all_goals(ec, GAMMA); oc = goal_occupancy(ec, sc.pi, GAMMA)
    hc = T.build_targets(ec, sc, oc, "hard"); tabc = Qt.support_table(sc)
    unseen_states = np.array([3, 7, 13, 17]); train_rows = np.where(~np.isin(hc.s, unseen_states))[0]; test_rows = np.where(np.isin(hc.s, unseen_states))[0]
    for c in (("hard", None), ("boltzmann", 0.05)):
        key = f"corridor-coord/{cname(c)}"; rep8[key] = []
        for seed in seeds3:
            net, tg, _ = fit(ec, sc, oc, c[0], c[1], seed, steps, rows=train_rows, encoding="coord")
            Hs = net.all_activations()["h3"].mean(1)
            same_run = np.array([[np.array_equal(tabc[i], tabc[j]) for j in range(ec.n_states)] for i in range(ec.n_states)])
            seen = np.setdiff1d(np.arange(ec.n_states), unseen_states)
            nn_ok = []
            for u in unseen_states:
                d = np.linalg.norm(Hs[seen] - Hs[u], axis=1); nn_ok.append(bool(same_run[u, seen[np.argmin(d)]]))
            cls_c = Sv.hard_classes(hc)
            rep8[key].append({"acc_train": T.argmax_accuracy(net, tg, train_rows), "acc_heldout": T.argmax_accuracy(net, tg, test_rows),
                              "ratio_train": Sv.within_between(net.all_activations()["h3"].reshape(-1, 128)[rows_of(tg, ec.n_states, ec.n_goals)][train_rows], cls_c[train_rows])["ratio"],
                              "ratio_heldout": Sv.within_between(net.all_activations()["h3"].reshape(-1, 128)[rows_of(tg, ec.n_states, ec.n_goals)][test_rows], cls_c[test_rows])["ratio"],
                              "centroid_acc_heldout": float(np.mean(nn_ok)), "centroid_acc_chance": float(same_run[np.ix_(unseen_states, seen)].mean())})
        r = rep8[key]
        print(f"[task8] {key:22s} acc train/unseen-states={np.mean([x['acc_train'] for x in r]):.3f}/{np.mean([x['acc_heldout'] for x in r]):.3f}; unseen cell's nearest seen cell in same run: {np.mean([x['centroid_acc_heldout'] for x in r]):.2f} (chance {r[0]['centroid_acc_chance']:.2f})")
    P3.plot_generalisation(rep8, out / "task8_generalisation.png")
    R["task8"] = rep8

    # ------------------------------------------------------------- Task 9: architectures
    archs = ["mlp3", "mlp6", "resmlp", "transformer"] if not args.quick else ["mlp3", "resmlp", "transformer"]
    conds9 = [("hard", None), ("boltzmann", 0.1), ("boltzmann", 0.5)]
    rep9, synth9 = {}, {}
    for a in archs:
        for c in conds9:
            rep9[(a, c)] = []
            for seed in seeds3:
                net, tg, _ = fit(env, sol, occ, c[0], c[1], seed, steps, arch=a); H = net.all_activations()
                r = Sv.geometry_report(H, net.hidden_layers, env, occ, sol, tg, GAMMA, pair_rows)
                rep9[(a, c)].append({"last": r[net.hidden_layers[-1]], "mid": r[net.hidden_layers[len(net.hidden_layers) // 2]], "accuracy": T.argmax_accuracy(net, tg), "layers": net.hidden_layers})
            if c != ("boltzmann", 0.5):
                synth9[(a, c)] = []
                for seed in seeds3:
                    res, layers = synth_run(a, c, seed); synth9[(a, c)].append(res[layers[-1]])
            r = rep9[(a, c)]
            print(f"[task9] {a:12s} {cname(c):18s} acc={np.mean([x['accuracy'] for x in r]):.3f} last-layer RSA target={np.mean([x['last']['rsa_target'] for x in r]):.2f} policy={np.mean([x['last']['rsa_policy'] for x in r]):.2f} occ={np.mean([x['last']['rsa_occupancy'] for x in r]):.2f}  {time.time()-t0:.0f}s")
    P3.plot_architectures(archs, conds9, rep9, synth9, margins, out / "task9_architectures.png")
    R["task9"] = {"reports": {f"{a}|{cname(c)}": v for (a, c), v in rep9.items()}, "synthetic_last_layer": {f"{a}|{cname(c)}": np.mean(v, 0).tolist() for (a, c), v in synth9.items()}}

    # ------------------------------------------------------------- Task 10: unique interaction variance and ablation
    rep10 = []
    for seed in seeds3:
        net = nets[(("hard", None), seed)]; H_all = acts[(("hard", None), seed)]; r = {}
        for l in L3:
            H = H_all[l]; dec = Sv.additive_decomposition(H); rng10 = np.random.default_rng(seed)
            base_m = Qt._behaviour_metrics(net, sol, l, H, rng10)
            rem_u = Qt._behaviour_metrics(net, sol, l, H - dec["r_unique"], rng10)
            rem_all = Qt._behaviour_metrics(net, sol, l, H - dec["r"], rng10)
            Bm = dec["basis_additive"]; noise = rng10.normal(size=H.shape); nf = noise.reshape(-1, H.shape[-1]); nf = nf - (nf @ Bm.T) @ Bm
            noise = nf.reshape(H.shape); noise *= np.linalg.norm(dec["r_unique"], axis=-1, keepdims=True) / (np.linalg.norm(noise, axis=-1, keepdims=True) + 1e-12)
            rand_c = Qt._behaviour_metrics(net, sol, l, H - noise, rng10)
            r[l] = {k: dec[k] for k in ("state", "goal", "interaction", "unique_fraction", "unique_of_total", "additive_span_dim")}
            r[l].update({"baseline": base_m, "remove_unique": rem_u, "random_in_complement": rand_c, "remove_all_interaction": rem_all})
        rep10.append(r)
    for l in L3:
        print(f"[task10] {l}: interaction={np.mean([r[l]['interaction'] for r in rep10]):.3f} unique fraction of it={np.mean([r[l]['unique_fraction'] for r in rep10]):.2f} (of total {np.mean([r[l]['unique_of_total'] for r in rep10]):.3f}); acc base/−unique/−random/−all = "
              + "/".join(f"{np.mean([r[l][k]['accuracy'] for r in rep10]):.3f}" for k in ("baseline", "remove_unique", "random_in_complement", "remove_all_interaction")))
    P3.plot_unique_interaction(rep10, out / "task10_unique_interaction.png")
    R["task10"] = rep10

    # ------------------------------------------------------------- Task 11: steering across temperature
    rep11 = {}
    for c in conds:
        rep11[c] = []
        for seed in seeds3:
            net = nets[(c, seed)]; tg = tgs[c]; r = {}
            for l in ("h2", "h3"):
                tab = Qt.steering_predictors(net, sol, occ, l, n_goal_pairs=8 if not args.quick else 4, seed=seed)
                # target-probability margin for g2 at the state: p(a*_g2) - p(a1)
                Yrow = np.zeros((S, K, tg.out_dim)); Yrow[tg.s, tg.g] = tg.Y
                tm = []
                for s_, g1, g2 in zip(tab["state"].astype(int), tab["g1"].astype(int), tab["g2"].astype(int)):
                    a1 = int(np.argmax(sol.pi[g1, s_])); a2 = int(np.argmax(sol.pi[g2, s_]))
                    tm.append(Yrow[s_, g2, a2] - Yrow[s_, g2, a1])
                tab["target_margin"] = np.array(tm)
                summ = Qt.steering_predictor_summary(tab)
                from scipy.stats import spearmanr
                fin = np.isfinite(tab["alpha_star"]); x = tab["target_margin"]
                summ["spearman_target_margin"] = float(spearmanr(tab["alpha_star"][fin], x[fin]).correlation) if fin.sum() > 5 and np.std(x[fin]) > 0 else float("nan")
                st = A.steering_report(net, sol, occ, layers=(l,), n_goal_pairs=8, alphas=np.linspace(0, 2, 21), seed=seed)["layers"][l]
                summ["steer_success_alpha1"] = st["agree_g2"][10]
                r[l] = summ
            rep11[c].append(r)
        print(f"[task11] {cname(c):18s} h3: steer@1={np.mean([r['h3']['steer_success_alpha1'] for r in rep11[c]]):.2f} ρ(α*,·): lin={np.nanmean([r['h3']['spearman_alpha_lin'] for r in rep11[c]]):+.2f} logit={np.nanmean([r['h3']['spearman_logit_margin'] for r in rep11[c]]):+.2f} target={np.nanmean([r['h3']['spearman_target_margin'] for r in rep11[c]]):+.2f} m1={np.nanmean([r['h3']['spearman_m1'] for r in rep11[c]]):+.2f} m2={np.nanmean([r['h3']['spearman_m2'] for r in rep11[c]]):+.2f}")
    P3.plot_steering_tau(conds, rep11, out / "task11_steering_tau.png")
    R["task11"] = {cname(c): v for c, v in rep11.items()}

    # ------------------------------------------------------------- Task 13: same targets, different losses
    rep13, acts13 = {}, {}
    for tau in (0.05, 0.5):
        for loss in ("ce", "kl", "mse_prob", "mse_logit"):
            key = f"tau{tau:g}/{loss}"; rep13[key] = []
            for seed in seeds3:
                net, tg, _ = fit(env, sol, occ, "boltzmann", tau, seed, steps, loss=loss); H = net.all_activations(); acts13[(key, seed)] = H
                r = Sv.geometry_report(H, L3, env, occ, sol, tg, GAMMA, pair_rows); r["accuracy"] = T.argmax_accuracy(net, tg); rep13[key].append(r)
            print(f"[task13] {key:16s} acc={np.mean([x['accuracy'] for x in rep13[key]]):.3f} h3 RSA target={np.mean([x['h3']['rsa_target'] for x in rep13[key]]):.2f} policy={np.mean([x['h3']['rsa_policy'] for x in rep13[key]]):.2f} occ={np.mean([x['h3']['rsa_occupancy'] for x in rep13[key]]):.2f}  {time.time()-t0:.0f}s")
    names13 = list(rep13); cross13 = {}
    for a in names13:
        for b in names13:
            vals = [G.rsa(G.rdm(acts13[(a, sa)]["h3"].mean(1)), G.rdm(acts13[(b, sb)]["h3"].mean(1))) for sa in seeds3 for sb in seeds3 if not (a == b and sa == sb)]
            cross13[(a, b)] = float(np.mean(vals))
    P3.plot_losses(names13, cross13, rep13, out / "task13_losses.png")
    R["task13"] = {"reports": rep13, "cross_rsa": {f"{a}|{b}": v for (a, b), v in cross13.items()}}

    # ------------------------------------------------------------- Task 14: ring A/B/C
    er = make_ring_env(); sr = solve_all_goals(er, GAMMA); orr = goal_occupancy(er, sr.pi, GAMMA)
    hr = T.build_targets(er, sr, orr, "hard"); pairs = Sv.abc_pairs(hr, tau=0.02)
    rep14 = {}
    for c in (("hard", None), ("boltzmann", 0.02), ("random-init", None)):
        key = cname(c) if c[0] != "random-init" else "random init"; rep14[key] = []
        for seed in seeds:
            if c[0] == "random-init":
                net = PolicyNet(er, seed=100 + seed); tg = hr
            else:
                net, tg, _ = fit(er, sr, orr, c[0], c[1], seed, steps)
            H = net.all_activations()
            rep14[key].append({l: Sv.abc_report(H[l].reshape(er.n_states * er.n_goals, -1)[rows_of(tg, er.n_states, er.n_goals)], pairs) for l in L3})
        print(f"[task14] {key:16s} h3 d_h/median A/B/C = " + "/".join(f"{np.mean([r['h3'][p] for r in rep14[key]]):.2f}" for p in "ABC"))
    P3.plot_abc(rep14, out / "task14_abc.png")
    R["task14"] = {"pairs": pairs["stats"], "reports": rep14}

    dump(R, out / "results3.json")
    write_tables(R, out)
    print(f"done in {time.time()-t0:.0f}s -> {out}/")


def write_tables(R, out: Path):
    L = ["h1", "h2", "h3"]
    lines = ["# TASK3 auto-generated tables\n"]

    def mo(reps, l, k):
        return float(np.nanmean([r[l][k] for r in reps]))

    lines += ["## Tasks 1-2: temperature sweep (base grid, mean over seeds)", "",
              "| condition | acc | layer | RSA target | RSA policy | RSA occupancy | RSA advantage | RSA SR | RSA spatial | partial target|env | partial occ|target | β target | R² | R² no target | pair RSA target | pair RSA Q | pair RSA policy row | PR |", "|---|" + "---|" * 17]
    for n, reps in R["task1_2"].items():
        for l in L:
            lines.append(f"| {n} | {np.mean([r['accuracy'] for r in reps]):.3f} | {l} | {mo(reps,l,'rsa_target'):+.2f} | {mo(reps,l,'rsa_policy'):+.2f} | {mo(reps,l,'rsa_occupancy'):+.2f} | {mo(reps,l,'rsa_advantage'):+.2f} | {mo(reps,l,'rsa_SR'):+.2f} | {mo(reps,l,'rsa_spatial'):+.2f} | "
                         f"{mo(reps,l,'partial_target_given_env'):+.2f} | {mo(reps,l,'partial_occupancy_given_target'):+.2f} | {np.mean([r[l]['regression']['beta_target'] for r in reps]):+.2f} | {np.mean([r[l]['regression']['r2'] for r in reps]):.2f} | {np.mean([r[l]['regression_no_target']['r2'] for r in reps]):.2f} | "
                         f"{mo(reps,l,'pair_rsa_target'):+.2f} | {mo(reps,l,'pair_rsa_Q'):+.2f} | {mo(reps,l,'pair_rsa_policy_row'):+.2f} | {mo(reps,l,'participation_ratio'):.2f} |")
    t3 = R["task3"]
    lines += ["", "## Task 3: fixed-argmax counterfactual", "", f"Synthetic probe margins {t3['margins']} (reference margin 0.5). Hidden distance to the reference / median background distance, h3:", "",
              "| condition | " + " | ".join(f"m={m}" for m in t3["margins"]) + " |", "|---|" + "---|" * len(t3["margins"])]
    for n, v in t3["synthetic"].items():
        lines.append(f"| {n} | " + " | ".join(f"{x:.2f}" for x in v["h3"]) + " |")
    lines += ["", "Base grid, pairs of (s,g) rows with the same argmax set:", "", "| condition | layer | median d_h same-argmax | small |Δmargin| | large |Δmargin| | ρ(d_h, |Δmargin|) | ρ(d_h, ‖Δtarget‖) |", "|---|" + "---|" * 6]
    for n, v in t3["natural"].items():
        for l in L:
            lines.append(f"| {n} | {l} | {np.mean([r['median_same_argmax_dist'] for r in v[l]]):.2f} | {np.mean([r['low_dmargin_dist'] for r in v[l]]):.2f} | {np.mean([r['high_dmargin_dist'] for r in v[l]]):.2f} | {np.nanmean([r['spearman_dist_dmargin'] for r in v[l]]):+.2f} | {np.nanmean([r['spearman_dist_dtarget'] for r in v[l]]):+.2f} |")
    t4 = R["task4"]
    lines += ["", "## Task 4: fixed environment, different T(Q) (h3)", "", "| condition | acc | RSA target | RSA policy | RSA occupancy | RSA advantage | RSA SR | RSA spatial | PR |", "|---|" + "---|" * 8]
    for n, reps in t4["reports"].items():
        lines.append(f"| {n} | {np.nanmean([r['accuracy'] for r in reps]):.3f} | {mo(reps,'h3','rsa_target'):+.2f} | {mo(reps,'h3','rsa_policy'):+.2f} | {mo(reps,'h3','rsa_occupancy'):+.2f} | {mo(reps,'h3','rsa_advantage'):+.2f} | {mo(reps,'h3','rsa_SR'):+.2f} | {mo(reps,'h3','rsa_spatial'):+.2f} | {mo(reps,'h3','participation_ratio'):.2f} |")
    names = list(t4["reports"])
    lines += ["", "RSA between hidden geometries (h3; diagonal = between seeds):", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        lines.append(f"| {a} | " + " | ".join(f"{t4['cross_rsa'][f'{a}|{b}']:.2f}" for b in names) + " |")
    t5 = R["task5"]
    lines += ["", "## Task 5: information destroyed by each target map", "", "| target | rows | classes | rank | PR | RSA with Q | recover Q R² | recover advantage R² | recover Z_sa R² (held-out states) | hidden PR h3 (rows) |", "|---|" + "---|" * 9]
    for n, s in t5["stats"].items():
        lines.append(f"| {n} | {s['n_rows']} | {s['n_classes']} | {s['rank']} | {s['participation_ratio']:.2f} | {s['rsa_with_Q']:.2f} | {s['recover_Q_r2']:.2f} | {s['recover_adv_r2']:.2f} | {s['recover_occ_r2']:.2f} | {t5['hidden_pr_h3'][n]:.2f} |")
    lines += ["", "## Task 6: sufficient-statistic pairs (row level; hidden distance / median)", "", "| condition | layer | same target, far Z_sa | n | different target, near Z_sa | n | within/between hard-class ratio (h3) |", "|---|" + "---|" * 6]
    for n, v in R["task6"].items():
        for l in L:
            lines.append(f"| {n} | {l} | {np.nanmean([r['same_target_far_env'] for r in v[l]]):.2f} | {v[l][0]['n_same_target_far_env']} | {np.nanmean([r['diff_target_near_env'] for r in v[l]]):.2f} | {v[l][0]['n_diff_target_near_env']} | {np.mean([r['ratio'] for r in v['within_between_h3']]):.2f} |")
    t7 = R["task7"]
    lines += ["", f"## Task 7: training trajectory (h3), steps {t7['steps']}", ""]
    for n, t in t7["traj"].items():
        lines += [f"**{n}**", "- within/between ratio: " + ", ".join(f"{x:.2f}" for x in t["ratio"]),
                  "- within (rel. to init): " + ", ".join(f"{x/t['within'][0]:.2f}" for x in t["within"]), "- between (rel. to init): " + ", ".join(f"{x/t['between'][0]:.2f}" for x in t["between"]),
                  "- RSA target: " + ", ".join(f"{x:.2f}" for x in t["rsa_target"]), "- RSA policy: " + ", ".join(f"{x:.2f}" for x in t["rsa_policy"]),
                  "- RSA advantage: " + ", ".join(f"{x:.2f}" for x in t["rsa_advantage"]), "- RSA occupancy: " + ", ".join(f"{x:.2f}" for x in t["rsa_occupancy"]),
                  "- PR: " + ", ".join(f"{x:.2f}" for x in t["participation_ratio"]), ""]
    lines += ["## Task 8: generalisation", "", "| setting | acc train | acc held-out | within/between train | within/between held-out | centroid/run accuracy held-out | chance |", "|---|" + "---|" * 6]
    for n, reps in R["task8"].items():
        lines.append(f"| {n} | {np.mean([r['acc_train'] for r in reps]):.3f} | {np.mean([r['acc_heldout'] for r in reps]):.3f} | {np.mean([r['ratio_train'] for r in reps]):.2f} | {np.mean([r['ratio_heldout'] for r in reps]):.2f} | {np.mean([r['centroid_acc_heldout'] for r in reps]):.2f} | {reps[0]['centroid_acc_chance']:.2f} |")
    lines += ["", "## Task 9: architectures (last hidden layer)", "", "| arch | condition | acc | RSA target | RSA policy | RSA occupancy | RSA advantage | PR | mid-layer RSA target | mid-layer RSA policy |", "|---|" + "---|" * 9]
    for n, reps in R["task9"]["reports"].items():
        a, c = n.split("|")
        lines.append(f"| {a} | {c} | {np.mean([r['accuracy'] for r in reps]):.3f} | {np.mean([r['last']['rsa_target'] for r in reps]):+.2f} | {np.mean([r['last']['rsa_policy'] for r in reps]):+.2f} | {np.mean([r['last']['rsa_occupancy'] for r in reps]):+.2f} | {np.mean([r['last']['rsa_advantage'] for r in reps]):+.2f} | {np.mean([r['last']['participation_ratio'] for r in reps]):.2f} | {np.mean([r['mid']['rsa_target'] for r in reps]):+.2f} | {np.mean([r['mid']['rsa_policy'] for r in reps]):+.2f} |")
    lines += ["", "Synthetic fixed-argmax sweep, last layer, d_h(probe)/median for margins " + str(R["task3"]["margins"]) + ":", ""]
    for n, v in R["task9"]["synthetic_last_layer"].items():
        lines.append(f"- {n}: " + ", ".join(f"{x:.2f}" for x in v))
    lines += ["", "## Task 10: unique interaction variance (hard BC, base)", "", "| layer | state | goal | interaction | unique fraction of interaction | unique of total | additive span dim | acc baseline | acc −unique | acc −random (complement) | acc −all interaction | steer baseline | steer −unique | steer −all |", "|---|" + "---|" * 13]
    for l in L:
        r = R["task10"]
        lines.append(f"| {l} | {np.mean([x[l]['state'] for x in r]):.2f} | {np.mean([x[l]['goal'] for x in r]):.2f} | {np.mean([x[l]['interaction'] for x in r]):.3f} | {np.mean([x[l]['unique_fraction'] for x in r]):.2f} | {np.mean([x[l]['unique_of_total'] for x in r]):.3f} | {r[0][l]['additive_span_dim']} | "
                     + " | ".join(f"{np.mean([x[l][k]['accuracy'] for x in r]):.3f}" for k in ("baseline", "remove_unique", "random_in_complement", "remove_all_interaction")) + " | "
                     + " | ".join(f"{np.nanmean([x[l][k]['steer_success'] for x in r]):.2f}" for k in ("baseline", "remove_unique", "remove_all_interaction")) + " |")
    lines += ["", "## Task 11: steering across temperature", "", "| condition | layer | steer success α=1 | ρ α_lin | ρ logit margin | ρ target margin | ρ m1 | ρ m2 | ρ ‖Δρ‖ | never flipped |", "|---|" + "---|" * 9]
    for n, reps in R["task11"].items():
        for l in ("h2", "h3"):
            lines.append(f"| {n} | {l} | {mo(reps,l,'steer_success_alpha1'):.2f} | {mo(reps,l,'spearman_alpha_lin'):+.2f} | {mo(reps,l,'spearman_logit_margin'):+.2f} | {mo(reps,l,'spearman_target_margin'):+.2f} | {mo(reps,l,'spearman_m1'):+.2f} | {mo(reps,l,'spearman_m2'):+.2f} | {mo(reps,l,'spearman_d_occ'):+.2f} | {mo(reps,l,'n_never_flipped'):.0f} |")
    t13 = R["task13"]
    lines += ["", "## Task 13: same targets, different losses (h3)", "", "| setting | acc | RSA target | RSA policy | RSA occupancy | RSA advantage | PR |", "|---|" + "---|" * 6]
    for n, reps in t13["reports"].items():
        lines.append(f"| {n} | {np.mean([r['accuracy'] for r in reps]):.3f} | {mo(reps,'h3','rsa_target'):+.2f} | {mo(reps,'h3','rsa_policy'):+.2f} | {mo(reps,'h3','rsa_occupancy'):+.2f} | {mo(reps,'h3','rsa_advantage'):+.2f} | {mo(reps,'h3','participation_ratio'):.2f} |")
    names = list(t13["reports"])
    lines += ["", "RSA between hidden geometries across losses (h3):", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        lines.append(f"| {a} | " + " | ".join(f"{t13['cross_rsa'][f'{a}|{b}']:.2f}" for b in names) + " |")
    t14 = R["task14"]
    lines += ["", "## Task 14: ring A/B/C pairs (row level; hidden distance / median)", "", "- pair statistics: " + "; ".join(f"{k}: n={v['n']}, median ‖ΔQ‖={v['dQ']:.2f}, median ‖Δsoft(τ=0.02)‖={v['dS']:.2f}" for k, v in t14["pairs"].items()), "",
              "| condition | layer | A | B | C |", "|---|---|---|---|---|"]
    for n, reps in t14["reports"].items():
        for l in L:
            lines.append(f"| {n} | {l} | " + " | ".join(f"{np.mean([r[l][p] for r in reps]):.2f}" for p in "ABC") + " |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
