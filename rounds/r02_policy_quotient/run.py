"""TASK2: policy-quotient versus occupancy geometry. Writes rounds/r02_policy_quotient/.

    .venv/bin/python rounds/r02_policy_quotient/run.py            # full (~25 min CPU)
    .venv/bin/python rounds/r02_policy_quotient/run.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from goalgeo import analysis as A, geometry as G, quotient as Qt, plotting as P
import plots as P2  # noqa: E402
from goalgeo.envs import make_env, make_switch_env, make_corridor_env
from goalgeo.model import PolicyNet
from goalgeo.occupancy import goal_occupancy, successor_representation
from goalgeo.planning import solve_all_goals, optimal_dataset
from goalgeo.train import train_bc, evaluate_accuracy

GAMMA = 0.9
L3 = ["h1", "h2", "h3"]


def dump(obj, path):
    Path(path).write_text(json.dumps(A.to_jsonable(obj), indent=1))


TARGETS = "argmax"


def train(env, sol, seed, steps, init=None, lr=1e-3):
    s, g, y = optimal_dataset(env, sol, targets=TARGETS)
    net = PolicyNet(env, seed=seed)
    if init is not None:
        net.load_state_dict(init)
    train_bc(net, s, g, y, steps=steps, lr=lr, seed=seed)
    return net, evaluate_accuracy(net, s, g, sol_targets(env, sol))


def sol_targets(env, sol):
    return optimal_dataset(env, sol)[2]


def mean_over(reps, layer, key):
    return float(np.mean([r[layer][key] for r in reps]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="rounds/r02_policy_quotient")
    ap.add_argument("--targets", default="argmax", choices=["argmax", "boltzmann"]); ap.add_argument("--sweep-only", action="store_true")
    args = ap.parse_args()
    global TARGETS; TARGETS = args.targets
    out = Path(args.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    lams = [float(round(x, 2)) for x in np.arange(0, 0.96, 0.05)] if not args.quick else [0.0, 0.2, 0.4, 0.6, 0.8]
    seeds = [0, 1, 2, 3, 4] if not args.quick else [0, 1]
    chain_seeds = [0, 1, 2] if not args.quick else [0]
    steps = 3000 if not args.quick else 300
    chain_steps = 1000 if not args.quick else 100
    R = {}

    # ---------------------------------------------------------------- Tasks 1-2
    gt = Qt.sweep_ground_truth(lams, GAMMA)
    P2.plot_sweep_ground_truth(gt, out / "task1_ground_truth.png")
    P.plot_env(gt["envs"][0], gt["sols"][0], gt["envs"][0].goal_coords.index((4, 12)), out / "env_switch_lam0.png")
    P.plot_env(gt["envs"][-1], gt["sols"][-1], gt["envs"][0].goal_coords.index((4, 12)), out / f"env_switch_lam{lams[-1]}.png")
    n_boundary_pairs = sum(1 for v in gt["boundaries"].values() if v)
    R["task1"] = {"lams": lams, "n_states": gt["envs"][0].n_states, "n_goals": gt["envs"][0].n_goals, "n_flips_per_step": gt["n_flips"],
                  "n_sg_pairs_with_boundary": n_boundary_pairs, "n_sg_pairs": int(np.prod(gt["margin"].shape[1:])),
                  "occ_change_per_step": np.linalg.norm(gt["Z_sa"][1:] - gt["Z_sa"][:-1], axis=(1, 2)).tolist()}
    print(f"[task1-2] switch env S={gt['envs'][0].n_states} K={gt['envs'][0].n_goals}; (s,g) pairs with a boundary: {n_boundary_pairs}; flips/step {gt['n_flips']}  {time.time()-t0:.0f}s")

    # ---------------------------------------------------------------- Task 3: train across the sweep
    nets = {}      # (lam_idx, seed) -> net
    acts = {}      # (lam_idx, seed) -> layer -> [S,K,d]
    accs = {}
    for i, lam in enumerate(lams):
        for seed in seeds:
            net, acc = train(gt["envs"][i], gt["sols"][i], seed, steps)
            nets[(i, seed)] = net; acts[(i, seed)] = net.all_activations(); accs[(i, seed)] = acc
        print(f"[task3] λ={lam:.2f} acc={np.mean([accs[(i, s)] for s in seeds]):.3f}  {time.time()-t0:.0f}s")
    rand_acts = {seed: PolicyNet(gt["envs"][0], seed=100 + seed).all_activations() for seed in seeds}
    # warm-start chains + same-λ continuation controls
    chain, chain_ctrl, chain_states = {}, {}, {}
    for seed in chain_seeds:
        chain[(0, seed)] = acts[(0, seed)]
        chain_states[(0, seed)] = copy.deepcopy(nets[(0, seed)].state_dict())
        for i in range(1, len(lams)):
            net, _ = train(gt["envs"][i], gt["sols"][i], seed, chain_steps, init=chain_states[(i - 1, seed)])
            chain[(i, seed)] = net.all_activations(); chain_states[(i, seed)] = copy.deepcopy(net.state_dict())
        # noise floor: keep training the chain network at the SAME λ for chain_steps and measure the drift
        for i in range(len(lams) - 1):
            netc, _ = train(gt["envs"][i], gt["sols"][i], seed + 50, chain_steps, init=chain_states[(i, seed)])
            chain_ctrl[(i, seed)] = netc.all_activations()
    print(f"[task3] chains done  {time.time()-t0:.0f}s")
    R["task3"] = {"accuracy_by_lambda": [float(np.mean([accs[(i, s)] for s in seeds])) for i in range(len(lams))]}

    # ---------------------------------------------------------------- Task 4: change across λ
    S, K = gt["envs"][0].n_states, gt["envs"][0].n_goals
    supp = np.stack([s.pi > 0 for s in gt["sols"]])                    # [L, K, S, A]
    Qs = np.stack([o.Q for o in gt["occs"]])                            # [L, K, S, A]
    marg = gt["margin"]                                                 # [L, K, S]
    agg = {"mid": [(lams[i] + lams[i + 1]) / 2 for i in range(len(lams) - 1)], "occ_change": R["task1"]["occ_change_per_step"], "n_flips": gt["n_flips"]}
    per_sg_tables = {l: {k: [] for k in ("d_h", "d_h_proc", "d_occ", "d_sr", "d_margin", "flip", "state_flips")} for l in L3}
    for layer in L3:
        agg[layer] = {}
        chain_m, chain_f, proc_m, proc_f, rdm_m, rdm_f = [], [], [], [], [], []
        cat = {k: {"chain": [], "proc": []} for k in ("flipped", "same_state", "other")}
        for i in range(len(lams) - 1):
            flip = (supp[i] != supp[i + 1]).any(-1).T                   # [S, K]
            state_flips = flip.sum(1)                                   # [S]
            d_occ = np.linalg.norm(Qs[i + 1] - Qs[i], axis=-1).T        # [S, K]
            d_sr = np.repeat(np.linalg.norm(gt["SR"][i + 1] - gt["SR"][i], axis=-1)[:, None], K, 1)
            d_marg = np.abs(marg[i + 1] - marg[i]).T
            # chain (raw, seed-matched) with same-λ continuation as the noise floor
            for seed in chain_seeds:
                Hn, Hp = chain[(i + 1, seed)][layer].reshape(S * K, -1), chain[(i, seed)][layer].reshape(S * K, -1)
                Hc = chain_ctrl[(i, seed)][layer].reshape(S * K, -1)
                d = np.linalg.norm(Hn - Hp, axis=1); f = np.linalg.norm(Hc - Hp, axis=1)
                floor = f.mean() + 1e-12
                chain_m.append(d.mean()); chain_f.append(f.mean())
                dn = (d / floor).reshape(S, K)
                cat["flipped"]["chain"].append(dn[flip].mean() if flip.any() else np.nan)
                ss = (state_flips[:, None] > 0) & ~flip
                cat["same_state"]["chain"].append(dn[ss].mean() if ss.any() else np.nan)
                oth = np.repeat(state_flips[:, None] == 0, K, 1)
                cat["other"]["chain"].append(dn[oth].mean() if oth.any() else np.nan)
                T = per_sg_tables[layer]
                T["d_h"].append(dn.ravel()); T["d_occ"].append(d_occ.ravel()); T["d_sr"].append(d_sr.ravel()); T["d_margin"].append(d_marg.ravel())
                T["flip"].append(flip.ravel().astype(float)); T["state_flips"].append(np.repeat(state_flips, K).astype(float))
            # Procrustes between independently trained networks; floor = between seeds at the same λ
            for seed in seeds:
                Hi, Hj = acts[(i, seed)][layer].reshape(S * K, -1), acts[(i + 1, seed)][layer].reshape(S * K, -1)
                d = G.procrustes_rowwise_distance(Hi, Hj)
                s2 = seeds[(seeds.index(seed) + 1) % len(seeds)]
                f = G.procrustes_rowwise_distance(Hi, acts[(i, s2)][layer].reshape(S * K, -1))
                proc_m.append(d.mean()); proc_f.append(f.mean())
                dn = (d / (f.mean() + 1e-12)).reshape(S, K)
                cat["flipped"]["proc"].append(dn[flip].mean() if flip.any() else np.nan)
                ss = (state_flips[:, None] > 0) & ~flip
                cat["same_state"]["proc"].append(dn[ss].mean() if ss.any() else np.nan)
                cat["other"]["proc"].append(dn[np.repeat(state_flips[:, None] == 0, K, 1)].mean())
                per_sg_tables[layer]["d_h_proc"].append(dn.ravel())
                # RDM change (state view); floor = between seeds at same λ
                Ri = G.rdm(A.state_views(acts[(i, seed)][layer])["mean"]); Rj = G.rdm(A.state_views(acts[(i + 1, seed)][layer])["mean"])
                Rs = G.rdm(A.state_views(acts[(i, s2)][layer])["mean"])
                rdm_m.append(1 - G.rsa(Ri, Rj)); rdm_f.append(1 - G.rsa(Ri, Rs))
        n_int = len(lams) - 1
        agg[layer]["chain"] = {"mean": np.array(chain_m).reshape(n_int, -1).mean(1).tolist(), "floor": np.array(chain_f).reshape(n_int, -1).mean(1).tolist(),
                               "flipped": float(np.nanmean(cat["flipped"]["chain"])), "same_state": float(np.nanmean(cat["same_state"]["chain"])), "other": float(np.nanmean(cat["other"]["chain"]))}
        agg[layer]["procrustes"] = {"mean": np.array(proc_m).reshape(n_int, -1).mean(1).tolist(), "floor": np.array(proc_f).reshape(n_int, -1).mean(1).tolist(),
                                    "flipped": float(np.nanmean(cat["flipped"]["proc"])), "same_state": float(np.nanmean(cat["same_state"]["proc"])), "other": float(np.nanmean(cat["other"]["proc"]))}
        agg[layer]["rdm"] = {"mean": np.array(rdm_m).reshape(n_int, -1).mean(1).tolist(), "floor": np.array(rdm_f).reshape(n_int, -1).mean(1).tolist()}
        # pooled per-(s,g) regression of hidden change on ground-truth changes
        T = {k: np.concatenate(v) for k, v in per_sg_tables[layer].items() if v}
        from scipy.stats import spearmanr, rankdata
        for target in ("d_h", "d_h_proc"):
            if target not in T:
                continue
            y = T[target]
            Xcols = {}
            reps_needed = len(y) // (S * K * (len(lams) - 1))
            for k in ("d_occ", "d_sr", "d_margin", "flip", "state_flips"):
                per_interval = np.array(per_sg_tables[layer][k]).reshape(len(lams) - 1, len(chain_seeds), S * K)[:, 0, :]
                Xcols[k] = np.repeat(per_interval[:, None, :], reps_needed, axis=1).ravel()
            ok = np.isfinite(y)
            res = {f"spearman_{k}": float(spearmanr(y[ok], v[ok]).correlation) for k, v in Xcols.items() if np.std(v[ok]) > 0}
            nf = ok & (Xcols["flip"] == 0)
            res["spearman_d_occ_unflipped_only"] = float(spearmanr(y[nf], Xcols["d_occ"][nf]).correlation) if np.std(Xcols["d_occ"][nf]) > 0 else float("nan")
            res["mean_change_flipped"] = float(y[ok & (Xcols["flip"] == 1)].mean()) if (Xcols["flip"] == 1).any() else float("nan")
            res["mean_change_unflipped"] = float(y[nf].mean())
            Xm = np.column_stack([rankdata(Xcols[k][ok]) for k in ("d_occ", "d_sr", "d_margin", "flip")]); Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-12)
            yr = rankdata(y[ok]); yr = (yr - yr.mean()) / yr.std()
            beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yr)), Xm]), yr, rcond=None)
            res["regression_beta"] = dict(zip(("d_occ", "d_sr", "d_margin", "flip"), map(float, beta[1:])))
            res["regression_r2"] = float(1 - np.var(yr - np.column_stack([np.ones(len(yr)), Xm]) @ beta) / np.var(yr))
            agg[layer][f"per_sg_{target}"] = res
        if layer == "h2":
            P2.plot_change_scatter({"d_h": T["d_h"], "d_occ": np.array(per_sg_tables[layer]["d_occ"]).ravel(), "d_margin": np.array(per_sg_tables[layer]["d_margin"]).ravel(), "flip": np.array(per_sg_tables[layer]["flip"]).ravel()}, out / "task4_scatter_h2.png")
    for layer in L3:
        P2.plot_change_across_lambda(agg, out / f"task4_change_{layer}.png", layer)
    R["task4"] = agg
    for layer in L3:
        r = agg[layer]["per_sg_d_h"]
        print(f"[task4] {layer}: chain change flipped/same-state/other = {agg[layer]['chain']['flipped']:.2f}/{agg[layer]['chain']['same_state']:.2f}/{agg[layer]['chain']['other']:.2f} (x noise floor); "
              f"β(occ,sr,margin,flip)={tuple(round(v,2) for v in r['regression_beta'].values())} ρ(occ|unflipped)={r['spearman_d_occ_unflipped_only']:+.2f}")

    # ---------------------------------------------------------------- shared: extra environments
    extra = {}
    for name in ([] if args.sweep_only else ["base", "twins", "portal", "barrier", "corridor"]):
        e = make_corridor_env() if name == "corridor" else make_env(name)
        sol = solve_all_goals(e, GAMMA); occ = goal_occupancy(e, sol.pi, GAMMA)
        ns = [train(e, sol, sd, steps)[0] for sd in seeds[:3]]
        extra[name] = {"env": e, "sol": sol, "occ": occ, "nets": ns, "acts": [n.all_activations() for n in ns],
                       "rand": [PolicyNet(e, seed=100 + k).all_activations() for k in range(len(ns))]}
        print(f"[extra] {name}: acc={np.mean([evaluate_accuracy(n, *optimal_dataset(e, sol)) for n in ns]):.3f}  {time.time()-t0:.0f}s")
    sw_idx = {lam: lams.index(lam) for lam in ([0.0, 0.2, 0.5, 0.9] if not args.quick else [0.0, 0.4, 0.8])}
    for lam, i in sw_idx.items():
        extra[f"switch{lam}"] = {"env": gt["envs"][i], "sol": gt["sols"][i], "occ": gt["occs"][i], "nets": [nets[(i, s)] for s in seeds[:3]],
                                 "acts": [acts[(i, s)] for s in seeds[:3]], "rand": [rand_acts[s] for s in seeds[:3]]}

    # ---------------------------------------------------------------- Task 5: quotient pairs
    q5 = {}
    for name in ([] if args.sweep_only else ["corridor", "base", "twins"]) + [f"switch{l}" for l in sw_idx]:
        E = extra[name]; pairs = Qt.quotient_pairs(E["env"], E["sol"], E["occ"])
        if not pairs["case_A"] or not pairs["case_B"]:
            print(f"[task5] {name}: no pairs (A={len(pairs['case_A'])}, B={len(pairs['case_B'])})"); continue
        q5[name] = [Qt.quotient_pair_report(h, hr, pairs) for h, hr in zip(E["acts"], E["rand"])]
        q5[name + "_meta"] = {"n_A": len(pairs["case_A"]), "n_B": len(pairs["case_B"]), "n_same_table_pairs": pairs["n_same"], "n_pairs": pairs["n_pairs"],
                              "d_occ_A": q5[name][0]["h2"]["d_occ_A"], "d_occ_B": q5[name][0]["h2"]["d_occ_B"]}
        print(f"[task5] {name}: A={len(pairs['case_A'])} B={len(pairs['case_B'])} d_occ A/B={q5[name][0]['h2']['d_occ_A']:.2f}/{q5[name][0]['h2']['d_occ_B']:.2f}; "
              f"h3 d_h A/B = {mean_over(q5[name],'h3','d_h_A'):.2f}/{mean_over(q5[name],'h3','d_h_B'):.2f} (random {mean_over(q5[name],'h3','d_h_A_random'):.2f}/{mean_over(q5[name],'h3','d_h_B_random'):.2f})")
    if any(not k.endswith("_meta") for k in q5):
        P2.plot_quotient_pairs({k: v for k, v in q5.items() if not k.endswith("_meta")}, out / "task5_quotient_pairs.png")
    R["task5"] = q5

    # ---------------------------------------------------------------- Task 6: extended regression
    q6 = {}
    for name, E in extra.items():
        rd = Qt.extended_rdms(E["env"], E["occ"], E["sol"], GAMMA)
        q6[name] = [Qt.extended_regression(h, rd) for h in E["acts"]]
        q6[name + "_hyp_rsa"] = {f"{a}~{b}": G.rsa(rd[a], rd[b]) for a in rd for b in rd if a < b}
        r = q6[name][0]["h2"]["regression"]
        print(f"[task6] {name:10s} h2 β: " + " ".join(f"{k[5:]}={np.mean([x['h2']['regression'][k] for x in q6[name]]):+.2f}" for k in r if k.startswith("beta")) +
              f" R²={np.mean([x['h2']['regression']['r2'] for x in q6[name]]):.2f} | partial policy|rest={mean_over(q6[name],'h2','partial_policy_given_rest'):+.2f}")
    P2.plot_extended_regression({k: v for k, v in q6.items() if not k.endswith("_hyp_rsa")}, out / "task6_regression.png")
    R["task6"] = q6

    # ---------------------------------------------------------------- Task 7: interaction ablation
    q7 = {}
    for name in ([] if args.sweep_only else ["base", "switch0.5" if not args.quick else "switch0.4"]):
        E = extra[name]
        q7[name] = [{l: Qt.interaction_ablation(n, E["sol"], E["occ"], l, n_random=3, seed=k) for l in L3} for k, n in enumerate(E["nets"])]
        for l in L3:
            r = q7[name]
            print(f"[task7] {name} {l}: acc base/−h_sg/−rand/−sub/−randsub = " + "/".join(f"{np.mean([x[l][c]['accuracy'] for x in r]):.2f}" for c in ("baseline", "remove_component", "random_component", "remove_subspace", "random_subspace")) +
                  f"; goal-sens = " + "/".join(f"{np.mean([x[l][c]['goal_sensitivity'] for x in r]):.2f}" for c in ("baseline", "remove_component", "random_component")) +
                  f"; steer = " + "/".join(f"{np.mean([x[l][c]['steer_success'] for x in r]):.2f}" for c in ("baseline", "remove_component", "random_component")) + f"; k={r[0][l]['k_subspace']} additive-var-in-subspace={np.mean([x[l]['additive_var_in_interaction_subspace'] for x in r]):.2f}")
    if q7:
        P2.plot_ablation(q7, out / "task7_ablation.png")
    R["task7"] = q7

    # ---------------------------------------------------------------- Task 8: steering predictors
    q8 = {}
    for name in ([] if args.sweep_only else ["base", "switch0.5" if not args.quick else "switch0.4"]):
        E = extra[name]
        q8[name] = [{l: Qt.steering_predictor_summary(Qt.steering_predictors(n, E["sol"], E["occ"], l, n_goal_pairs=12 if not args.quick else 4, seed=k)) for l in L3} for k, n in enumerate(E["nets"])]
        for l in L3:
            r = q8[name]
            print(f"[task8] {name} {l}: Spearman(α*,·) " + " ".join(f"{p}={np.nanmean([x[l][f'spearman_{p}'] for x in r]):+.2f}" for p in ("alpha_lin", "logit_margin", "m1", "m2", "d_occ", "occ_model")) +
                  f" | MAE lin={np.nanmean([x[l]['mae_alpha_lin'] for x in r]):.2f} occ={np.nanmean([x[l]['mae_occ_model'] for x in r]):.2f}")
    if q8:
        P2.plot_steering_predictors(q8, out / "task8_steering_predictors.png")
    R["task8"] = q8

    # ---------------------------------------------------------------- Task 9: stochastic vs deterministic policy tables across λ
    track = {"lams": lams}
    R_pol0 = G.rdm(gt["policy_tables"][0]); R_occ0 = G.rdm(gt["Z_sa"][0])
    for layer in L3:
        t = {k: [] for k in ("rsa_policy_lam", "rsa_policy_det", "rsa_occ_lam", "rsa_occ_det", "partial_policy_lam_given_det", "partial_policy_det_given_lam")}
        for i in range(len(lams)):
            Rp, Ro = G.rdm(gt["policy_tables"][i]), G.rdm(gt["Z_sa"][i])
            vals = {k: [] for k in t}
            for seed in seeds:
                Rh = G.rdm(A.state_views(acts[(i, seed)][layer])["mean"])
                vals["rsa_policy_lam"].append(G.rsa(Rh, Rp)); vals["rsa_policy_det"].append(G.rsa(Rh, R_pol0))
                vals["rsa_occ_lam"].append(G.rsa(Rh, Ro)); vals["rsa_occ_det"].append(G.rsa(Rh, R_occ0))
                vals["partial_policy_lam_given_det"].append(G.partial_rsa(Rh, Rp, [R_pol0])); vals["partial_policy_det_given_lam"].append(G.partial_rsa(Rh, R_pol0, [Rp]))
            for k in t:
                t[k].append(float(np.mean(vals[k])))
        track[layer] = t
    track["rsa_policy_lam_vs_det"] = [G.rsa(G.rdm(gt["policy_tables"][i]), R_pol0) for i in range(len(lams))]
    track["rsa_occ_lam_vs_det"] = [G.rsa(G.rdm(gt["Z_sa"][i]), R_occ0) for i in range(len(lams))]
    P2.plot_stochastic_tracking(track, out / "task9_stochastic_tracking.png")
    R["task9"] = track
    print(f"[task9] h2 at λ={lams[-1]}: RSA policy(λ)={track['h2']['rsa_policy_lam'][-1]:.2f} policy(0)={track['h2']['rsa_policy_det'][-1]:.2f}; partial λ|0={track['h2']['partial_policy_lam_given_det'][-1]:+.2f} 0|λ={track['h2']['partial_policy_det_given_lam'][-1]:+.2f}  {time.time()-t0:.0f}s")

    # ---------------------------------------------------------------- Task 10: dimensionality
    q10 = {}
    for name, E in extra.items():
        row = {"occupancy": Qt.dimensionality(E["occ"].Z_sa), "policy": Qt.dimensionality(Qt.policy_table(E["sol"])),
               "SR": Qt.dimensionality(successor_representation(E["env"], GAMMA)), "n_states": E["env"].n_states}
        for l in L3:
            row[l] = {k: float(np.mean([Qt.dimensionality(A.state_views(h[l])["mean"])[k] for h in E["acts"]])) for k in ("participation_ratio", "entropy_rank", "n_pcs_90", "two_nn")}
            row[f"{l}_random"] = {k: float(np.mean([Qt.dimensionality(A.state_views(h[l])["mean"])[k] for h in E["rand"]])) for k in ("participation_ratio", "entropy_rank", "n_pcs_90", "two_nn")}
        q10[name] = row
    P2.plot_dimensionality(q10, out / "task10_dimensionality.png")
    from scipy.stats import spearmanr
    envs10 = list(q10)
    if len(envs10) < 4:
        R["task10"] = {"table": q10}; dump(R, out / "results2.json"); write_tables(R, out); print(f"done in {time.time()-t0:.0f}s -> {out}/"); return
    R["task10"] = {"table": q10,
                   "spearman_PR_h3_vs_occupancy": float(spearmanr([q10[e]["occupancy"]["participation_ratio"] for e in envs10], [q10[e]["h3"]["participation_ratio"] for e in envs10]).correlation),
                   "spearman_PR_h3_vs_policy": float(spearmanr([q10[e]["policy"]["participation_ratio"] for e in envs10], [q10[e]["h3"]["participation_ratio"] for e in envs10]).correlation),
                   "spearman_PR_h3_vs_SR": float(spearmanr([q10[e]["SR"]["participation_ratio"] for e in envs10], [q10[e]["h3"]["participation_ratio"] for e in envs10]).correlation)}
    print(f"[task10] PR(h3) vs PR(occupancy) ρ={R['task10']['spearman_PR_h3_vs_occupancy']:+.2f}, vs PR(policy) ρ={R['task10']['spearman_PR_h3_vs_policy']:+.2f}, vs PR(SR) ρ={R['task10']['spearman_PR_h3_vs_SR']:+.2f}")

    dump(R, out / "results2.json")
    write_tables(R, out)
    print(f"done in {time.time()-t0:.0f}s -> {out}/")


def write_tables(R, out: Path):
    lines = ["# TASK2 auto-generated tables\n"]
    t1 = R["task1"]
    lines += ["## Task 1-2", f"- switch env: {t1['n_states']} states, {t1['n_goals']} goals, λ ∈ {t1['lams']}",
              f"- (s,g) pairs with at least one policy boundary: {t1['n_sg_pairs_with_boundary']} of {t1['n_sg_pairs']}",
              "- action flips per λ step: " + ", ".join(map(str, t1["n_flips_per_step"])),
              "- ‖ΔZ_sa‖ per λ step: " + ", ".join(f"{v:.2f}" for v in t1["occ_change_per_step"]), "",
              "## Task 3", "- BC accuracy by λ: " + ", ".join(f"{v:.3f}" for v in R["task3"]["accuracy_by_lambda"]), "",
              "## Task 4: representational change across λ (pooled over λ steps and seeds)", "",
              "| layer | measure | flipped (s,g) | same state, other goal | other (s,g) | β d_occ | β d_SR | β d_margin | β flip | R² | ρ(d_h, d_occ) unflipped only | mean change flipped / unflipped |", "|---|" + "---|" * 11]
    for l in L3:
        a = R["task4"][l]
        for meas, key in (("chain", "per_sg_d_h"), ("procrustes", "per_sg_d_h_proc")):
            r = a[key]; b = r["regression_beta"]
            lines.append(f"| {l} | {meas} | {a[meas]['flipped']:.2f} | {a[meas]['same_state']:.2f} | {a[meas]['other']:.2f} | {b['d_occ']:+.2f} | {b['d_sr']:+.2f} | {b['d_margin']:+.2f} | {b['flip']:+.2f} | {r['regression_r2']:.2f} | {r['spearman_d_occ_unflipped_only']:+.2f} | {r['mean_change_flipped']:.2f} / {r['mean_change_unflipped']:.2f} |")
    lines += ["", "Per-λ-step aggregate (h2): λ midpoint, chain change/floor, Procrustes change/floor, RDM change/floor, ‖ΔZ_sa‖, flips", ""]
    a = R["task4"]
    for j, mid in enumerate(a["mid"]):
        lines.append(f"- λ={mid:.3f}: chain {a['h2']['chain']['mean'][j]/max(a['h2']['chain']['floor'][j],1e-9):.2f}, proc {a['h2']['procrustes']['mean'][j]/max(a['h2']['procrustes']['floor'][j],1e-9):.2f}, rdm {a['h2']['rdm']['mean'][j]/max(a['h2']['rdm']['floor'][j],1e-9):.2f}, ΔZ {a['occ_change'][j]:.2f}, flips {a['n_flips'][j]}")
    lines += ["", "## Task 5: quotient pairs (hidden distance / median pairwise distance)", "",
              "| env | n_A | n_B | d_occ A | d_occ B | layer | d_h A | d_h B | d_h A random | d_h B random | frac(A<B) |", "|---|" + "---|" * 10]
    for name, reps in R["task5"].items():
        if name.endswith("_meta"):
            continue
        m = R["task5"][name + "_meta"]
        for l in L3:
            lines.append(f"| {name} | {m['n_A']} | {m['n_B']} | {m['d_occ_A']:.2f} | {m['d_occ_B']:.2f} | {l} | {mean_over(reps,l,'d_h_A'):.2f} | {mean_over(reps,l,'d_h_B'):.2f} | {mean_over(reps,l,'d_h_A_random'):.2f} | {mean_over(reps,l,'d_h_B_random'):.2f} | {mean_over(reps,l,'frac_pairs_A_below_B'):.2f} |")
    lines += ["", "## Task 6: six-way RDM regression (standardised β) and partials", "",
              "| env | layer | β occ | β SR | β policy | β advantage | β margin | β spatial | R² | R² without policy | partial policy|rest | partial occ|rest | partial margin|rest | partial adv|rest | RSA policy | RSA occ | RSA margin | RSA advantage |", "|---|" + "---|" * 17]
    for name, reps in R["task6"].items():
        if name.endswith("_hyp_rsa"):
            continue
        for l in L3:
            g = lambda k: float(np.mean([r[l]["regression"][k] for r in reps]))
            lines.append(f"| {name} | {l} | {g('beta_occupancy'):+.2f} | {g('beta_SR'):+.2f} | {g('beta_policy'):+.2f} | {g('beta_advantage'):+.2f} | {g('beta_margin'):+.2f} | {g('beta_spatial'):+.2f} | {g('r2'):.2f} | "
                         f"{float(np.mean([r[l]['regression_no_policy']['r2'] for r in reps])):.2f} | {mean_over(reps,l,'partial_policy_given_rest'):+.2f} | {mean_over(reps,l,'partial_occupancy_given_rest'):+.2f} | {mean_over(reps,l,'partial_margin_given_rest'):+.2f} | {mean_over(reps,l,'partial_advantage_given_rest'):+.2f} | "
                         f"{mean_over(reps,l,'rsa_policy'):+.2f} | {mean_over(reps,l,'rsa_occupancy'):+.2f} | {mean_over(reps,l,'rsa_margin'):+.2f} | {mean_over(reps,l,'rsa_advantage'):+.2f} |")
    lines += ["", "Hypothesis inter-correlations (RSA) per env:"]
    for name, v in R["task6"].items():
        if name.endswith("_hyp_rsa"):
            lines.append(f"- {name[:-8]}: " + ", ".join(f"{k} {x:.2f}" for k, x in v.items()))
    lines += ["", "## Task 7: interaction ablation", "", "| env | layer | interaction var | k | additive var inside k-subspace | condition | accuracy | goal sensitivity | steering success |", "|---|" + "---|" * 8]
    for name, reps in R["task7"].items():
        for l in L3:
            for c in ("baseline", "remove_component", "random_component", "remove_subspace", "random_subspace", "only_interaction_plus_state"):
                lines.append(f"| {name} | {l} | {np.mean([r[l]['interaction_fraction'] for r in reps]):.2f} | {reps[0][l]['k_subspace']} | {np.mean([r[l]['additive_var_in_interaction_subspace'] for r in reps]):.2f} | {c} | {np.mean([r[l][c]['accuracy'] for r in reps]):.3f} | {np.mean([r[l][c]['goal_sensitivity'] for r in reps]):.3f} | {np.nanmean([r[l][c]['steer_success'] for r in reps]):.3f} |")
    lines += ["", "## Task 8: what predicts the steering flip threshold α*?", "", "| env | layer | n | never flipped | ρ α_lin | ρ logit margin | ρ m1 | ρ m2 | ρ ‖Δρ‖ | ρ occ model | MAE α_lin | MAE occ model | regression R² | β (α_lin, logit, m1, m2, dρ, occ) |", "|---|" + "---|" * 13]
    for name, reps in R["task8"].items():
        for l in L3:
            g = lambda k: float(np.nanmean([r[l][k] for r in reps]))
            b = {k: float(np.mean([r[l]["regression_beta"][k] for r in reps])) for k in reps[0][l]["regression_beta"]}
            lines.append(f"| {name} | {l} | {int(np.mean([r[l]['n'] for r in reps]))} | {int(np.mean([r[l]['n_never_flipped'] for r in reps]))} | {g('spearman_alpha_lin'):+.2f} | {g('spearman_logit_margin'):+.2f} | {g('spearman_m1'):+.2f} | {g('spearman_m2'):+.2f} | {g('spearman_d_occ'):+.2f} | {g('spearman_occ_model'):+.2f} | {g('mae_alpha_lin'):.2f} | {g('mae_occ_model'):.2f} | {g('regression_r2'):.2f} | " + ", ".join(f"{v:+.2f}" for v in b.values()) + " |")
    t9 = R["task9"]
    lines += ["", "## Task 9: tracking π*_λ versus π*_0 across the sweep (h2)", "", "| λ | RSA(policy λ, policy 0) | RSA(occ λ, occ 0) | h2 RSA policy λ | h2 RSA policy 0 | partial λ|0 | partial 0|λ | h2 RSA occ λ | h2 RSA occ 0 |", "|---|" + "---|" * 8]
    for i, lam in enumerate(t9["lams"]):
        h = t9["h2"]
        lines.append(f"| {lam:.2f} | {t9['rsa_policy_lam_vs_det'][i]:.2f} | {t9['rsa_occ_lam_vs_det'][i]:.2f} | {h['rsa_policy_lam'][i]:.2f} | {h['rsa_policy_det'][i]:.2f} | {h['partial_policy_lam_given_det'][i]:+.2f} | {h['partial_policy_det_given_lam'][i]:+.2f} | {h['rsa_occ_lam'][i]:.2f} | {h['rsa_occ_det'][i]:.2f} |")
    t10 = R["task10"]
    lines += ["", "## Task 10: dimensionality", ""]
    if "spearman_PR_h3_vs_occupancy" in t10:
        lines += [f"- Spearman across environments of PR(h3) with PR(Z_sa) {t10['spearman_PR_h3_vs_occupancy']:+.2f}, with PR(policy) {t10['spearman_PR_h3_vs_policy']:+.2f}, with PR(SR) {t10['spearman_PR_h3_vs_SR']:+.2f}", ""]
    lines += [
              "| env | states | PR Z_sa | PR policy | PR SR | PR h1 | PR h2 | PR h3 | PR h3 random | 2NN Z_sa | 2NN policy | 2NN h3 | 2NN h3 random | PCs90 Z_sa | PCs90 h3 |", "|---|" + "---|" * 14]
    for name, r in t10["table"].items():
        lines.append(f"| {name} | {r['n_states']} | {r['occupancy']['participation_ratio']:.2f} | {r['policy']['participation_ratio']:.2f} | {r['SR']['participation_ratio']:.2f} | {r['h1']['participation_ratio']:.2f} | {r['h2']['participation_ratio']:.2f} | {r['h3']['participation_ratio']:.2f} | {r['h3_random']['participation_ratio']:.2f} | "
                     f"{r['occupancy']['two_nn']:.2f} | {r['policy']['two_nn']:.2f} | {r['h3']['two_nn']:.2f} | {r['h3_random']['two_nn']:.2f} | {r['occupancy']['n_pcs_90']} | {r['h3']['n_pcs_90']:.0f} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
