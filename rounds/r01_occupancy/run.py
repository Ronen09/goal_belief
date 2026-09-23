"""Run every phase of rounds/r01_occupancy/BRIEF.md and write rounds/r01_occupancy/ (figures, JSON, report tables).

    .venv/bin/python rounds/r01_occupancy/run.py            # full run
    .venv/bin/python rounds/r01_occupancy/run.py --quick    # smoke run (fewer seeds/steps)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from goalgeo import analysis as A, geometry as G, plotting as P
from goalgeo.envs import make_env, ENV_NAMES
from goalgeo.model import PolicyNet
from goalgeo.occupancy import goal_occupancy, shortest_path_occupancy
from goalgeo.planning import solve_all_goals, optimal_dataset, rollout
from goalgeo.train import train_bc, evaluate_accuracy

GAMMA = 0.9
CKPT_STEPS = [0, 10, 30, 100, 300, 1000, 3000]


def dump(obj, path):
    Path(path).write_text(json.dumps(A.to_jsonable(obj), indent=1))


def train_models(env, sol, encoding, seeds, steps, lr=1e-3, ckpts=None):
    s, g, y = optimal_dataset(env, sol)
    nets, hists = [], []
    for seed in seeds:
        net = PolicyNet(env, encoding=encoding, seed=seed)
        h = train_bc(net, s, g, y, steps=steps, lr=lr, checkpoint_steps=ckpts, seed=seed)
        nets.append(net); hists.append(h)
    return nets, hists, (s, g, y)


def rollout_success(env, net, sol, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ok, opt = [], []
    D = env.geodesic()
    with torch.no_grad():
        logits = net(torch.arange(env.n_states).repeat_interleave(env.n_goals),
                     torch.arange(env.n_goals).repeat(env.n_states)).reshape(env.n_states, env.n_goals, -1)
        pi_net = torch.softmax(logits * 50, -1).numpy()  # near-greedy
    for _ in range(n):
        gi = int(rng.integers(env.n_goals)); s0 = int(rng.integers(env.n_states))
        tr = rollout(env, pi_net[:, gi], s0, env.goal_states[gi], rng, max_steps=100)
        ok.append(tr.reached)
        if tr.reached:
            opt.append(len(tr.actions) == D[s0, env.goal_states[gi]])
    return float(np.mean(ok)), float(np.mean(opt)) if opt else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="rounds/r01_occupancy")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    seeds6 = [0] if args.quick else [0, 1, 2]
    steps = 300 if args.quick else 3000
    ckpts = [c for c in CKPT_STEPS if c <= steps]
    t0 = time.time()
    report = {}

    # ------------------------------------------------------------- Phase 1-3
    env = make_env("base"); sol = solve_all_goals(env, GAMMA); occ = goal_occupancy(env, sol.pi, GAMMA)
    d = out / "base"; d.mkdir(exist_ok=True)
    P.plot_env(env, sol, 5, d / "env.png")
    summ = A.occupancy_geometry_summary(occ)
    P.plot_occupancy_geometry(env, occ, summ, d / "occupancy_geometry.png")
    hyps = A.hypothesis_rdms(env, occ, GAMMA, sol=sol)
    summ["rsa_between_hypotheses"] = {f"{a}~{b}": G.rsa(hyps[a], hyps[b]) for a in hyps for b in hyps if a < b}
    summ["Z_state_participation_ratio"] = G.participation_ratio(occ.Z_state)
    dump(summ, d / "occupancy_summary.json")
    report["phase3"] = summ
    print(f"[phase1-3] base: S={env.n_states} K={env.n_goals} PR(Z_sa)={summ['participation_ratio']:.2f}  {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- Phase 4
    trained = {}
    for enc in ("onehot", "coord"):
        nets, hists, data = train_models(env, sol, enc, seeds, steps, ckpts=ckpts)
        accs = [evaluate_accuracy(n, *data) for n in nets]
        succ = [rollout_success(env, n, sol) for n in nets]
        trained[enc] = (nets, hists)
        report[f"phase4_{enc}"] = {"train_accuracy": accs, "rollout_success": [s for s, _ in succ],
                                   "rollout_optimal_fraction": [o for _, o in succ], "final_loss": [h.loss[-1] for h in hists]}
        if enc == "onehot":
            P.plot_training(hists, d / "training.png")
        print(f"[phase4] {enc}: acc={np.mean(accs):.3f} rollout success={np.mean([s for s,_ in succ]):.3f}  {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- Phase 5
    phase5 = {}
    for enc in ("onehot", "coord"):
        nets, hists = trained[enc]
        rep_tr = [A.layer_report(n.all_activations(), hyps, occ, env) for n in nets]
        rep_rand = [A.layer_report(PolicyNet(env, encoding=enc, seed=100 + i).all_activations(), hyps, occ, env) for i in range(len(nets))]
        rep_tr_concat = [A.layer_report(n.all_activations(), hyps, occ, env, view="concat") for n in nets]
        pair = [A.pair_level_report(n.all_activations(), occ, env, sol) for n in nets]
        phase5[enc] = {"trained_mean_view": rep_tr, "random_mean_view": rep_rand, "trained_concat_view": rep_tr_concat, "pair_level": pair}
        P.plot_layer_rsa(rep_tr, rep_rand, d / f"phase5_rsa_{enc}.png")
        P.plot_dimensionality(rep_tr, rep_rand, summ["participation_ratio"], d / f"phase5_dimensionality_{enc}.png")
        P.plot_pair_level(pair, d / f"phase5_pairlevel_{enc}.png")
        gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
        occ_at = lambda gm: goal_occupancy(env, solve_all_goals(env, gm).pi, gm)
        sweeps = [A.gamma_sweep(n.all_activations(), env, occ_at, gammas) for n in nets]
        sweep = {l: [float(np.mean([s[l][i] for s in sweeps])) for i in range(len(gammas))] for l in A.HIDDEN_LAYERS}
        phase5[enc]["gamma_sweep"] = {"gammas": gammas, "rsa": sweep}
        P.plot_gamma_sweep(gammas, sweep, d / f"phase5_gamma_sweep_{enc}.png", GAMMA)
        # emergence over checkpoints
        ck_reports, ck_acc = [], []
        s_, g_, y_ = optimal_dataset(env, sol)
        for step in ckpts:
            reps, ac = [], []
            for n, h in zip(nets, hists):
                probe = PolicyNet(env, encoding=enc); probe.load_state_dict(h.checkpoints[step])
                reps.append(A.layer_report(probe.all_activations(), hyps, occ, env))
                ac.append(evaluate_accuracy(probe, s_, g_, y_))
            ck_reports.append(reps); ck_acc.append(float(np.mean(ac)))
        for layer in A.HIDDEN_LAYERS:
            curves = {"rsa": {h: [float(np.mean([r[layer][f"rsa_{h}"] for r in reps])) for reps in ck_reports] for h in ("occupancy", "spatial", "geodesic", "SR")},
                      "decode": {k: [float(np.mean([r[layer][f"decode_r2_{k}"] for r in reps])) for reps in ck_reports] for k in ("occupancy", "spatial")}}
            phase5[enc][f"emergence_{layer}"] = {"steps": ckpts, "curves": curves, "acc": ck_acc}
            if layer == "h2":
                P.plot_emergence(ckpts, curves, ck_acc, d / f"phase5_emergence_{enc}.png", layer)
        l2 = rep_tr[0]["h2"]
        print(f"[phase5] {enc}: h2 RSA occ={np.mean([r['h2']['rsa_occupancy'] for r in rep_tr]):.2f} spatial={np.mean([r['h2']['rsa_spatial'] for r in rep_tr]):.2f} "
              f"decodeR2 occ={np.mean([r['h2']['decode_r2_occupancy'] for r in rep_tr]):.2f} (random {np.mean([r['h2']['decode_r2_occupancy'] for r in rep_rand]):.2f})  {time.time()-t0:.0f}s")
    dump(phase5, d / "phase5.json"); report["phase5"] = phase5

    # ------------------------------------------------------------- Phase 6
    d6 = out / "phase6"; d6.mkdir(exist_ok=True)
    phase6 = {}
    for ename in ENV_NAMES:
        e = make_env(ename); so = solve_all_goals(e, GAMMA); oc = goal_occupancy(e, so.pi, GAMMA)
        hy = A.hypothesis_rdms(e, oc, GAMMA, sol=so)
        P.plot_env(e, so, min(5, e.n_goals - 1), d6 / f"env_{ename}.png")
        phase6[f"{ename}_hypothesis_rsa"] = {"occupancy~spatial": G.rsa(hy["occupancy"], hy["spatial"]),
                                             "occupancy~geodesic": G.rsa(hy["occupancy"], hy["geodesic"]),
                                             "spatial~geodesic": G.rsa(hy["spatial"], hy["geodesic"]),
                                             "occupancy~policy": G.rsa(hy["occupancy"], hy["policy"]),
                                             "spatial~policy": G.rsa(hy["spatial"], hy["policy"])}
        for enc in ("onehot", "coord"):
            if ename == "base" and not args.quick:
                nets = trained[enc][0][:len(seeds6)]
            else:
                nets, _, _ = train_models(e, so, enc, seeds6, steps)
            accs = [evaluate_accuracy(n, *optimal_dataset(e, so)) for n in nets]
            reps = [A.discriminating_report(n.all_activations(), hy, e) for n in nets]
            reps_rand = [A.discriminating_report(PolicyNet(e, encoding=enc, seed=100 + i).all_activations(), hy, e) for i in range(len(nets))]
            phase6[(ename, enc)] = reps
            phase6[(ename, enc, "random")] = reps_rand
            phase6[(ename, enc, "acc")] = accs
            print(f"[phase6] {ename:8s} {enc:6s} acc={np.mean(accs):.3f} h2: β_occ={np.mean([r['h2']['beta_occupancy'] for r in reps]):+.2f} "
                  f"β_sp={np.mean([r['h2']['beta_spatial'] for r in reps]):+.2f} disagree ρ occ={np.mean([r['h2']['disagree_rsa_occupancy'] for r in reps]):+.2f} "
                  f"sp={np.mean([r['h2']['disagree_rsa_spatial'] for r in reps]):+.2f}"
                  + (f" special={np.mean([r['h2']['special_pairs_median'] for r in reps]):.2f} ctrl_sp={np.mean([r['h2']['special_pairs_spatial_matched_control'] for r in reps]):.2f}" if e.special_pairs else "")
                  + f"  {time.time()-t0:.0f}s")
    for layer in ("h1", "h2", "h3"):
        P.plot_discriminating({k: v for k, v in phase6.items() if isinstance(k, tuple) and len(k) == 2}, d6 / f"phase6_{layer}.png", ENV_NAMES, ["onehot", "coord"], layer)
    dump({("|".join(k) if isinstance(k, tuple) else k): v for k, v in phase6.items()}, d6 / "phase6.json")
    report["phase6"] = phase6

    # ------------------------------------------------------------- Phase 7
    nets, _ = trained["onehot"]
    rep7 = [A.goal_dependence_report(n.all_activations(), hyps, occ, env, sol) for n in nets]
    rep7_rand = [A.goal_dependence_report(PolicyNet(env, seed=100 + i).all_activations(), hyps, occ, env, sol) for i in range(len(nets))]
    P.plot_goal_dependence(rep7, d / "phase7.png")
    dump({"trained": rep7, "random": rep7_rand}, d / "phase7.json"); report["phase7"] = {"trained": rep7, "random": rep7_rand}
    print("[phase7] variance fractions (state/goal/interaction) per layer:",
          {l: tuple(round(np.mean([r[l][k] for r in rep7]), 2) for k in ("state", "goal", "interaction")) for l in A.HIDDEN_LAYERS}, f"{time.time()-t0:.0f}s")

    # ------------------------------------------------------------- Phase 8
    rep8 = [A.steering_report(n, sol, occ, n_goal_pairs=6 if args.quick else 15, seed=i) for i, n in enumerate(nets)]
    P.plot_steering(rep8, d / "phase8_steering.png")
    P.plot_flip_vs_margin(rep8, d / "phase8_flip_vs_margin.png")
    dump(rep8, d / "phase8.json"); report["phase8"] = rep8
    for layer in A.HIDDEN_LAYERS:
        a2 = np.mean([r["layers"][layer]["agree_g2"] for r in rep8], 0)
        print(f"[phase8] {layer}: agree g2 at α=1: {a2[10]:.2f}, α=2: {a2[-1]:.2f}; flipped by α=1: {np.mean([r['layers'][layer]['frac_flipped_by_alpha1'] for r in rep8]):.2f}; "
              f"corr(flip α, margin)={np.nanmean([r['layers'][layer]['flip_alpha_vs_margin_spearman'] for r in rep8]):+.2f}")

    # ------------------------------------------------------------- Phase 9 (stochastic)
    d9 = out / "stochastic"; d9.mkdir(exist_ok=True)
    report["phase9"] = {}
    for variant, env_s in (("slip0.2", make_env("base", slip=0.2)), ("ice", make_env("ice"))):
        sol_s = solve_all_goals(env_s, GAMMA)
        occ_s = goal_occupancy(env_s, sol_s.pi, GAMMA); occ_sp = shortest_path_occupancy(env_s, GAMMA)
        K_, S_, A_ = sol_s.pi.shape
        pol_stoch = np.transpose(sol_s.pi, (1, 0, 2)).reshape(S_, K_ * A_)
        pol_det = np.transpose(sol.pi, (1, 0, 2)).reshape(S_, K_ * A_)   # deterministic-shortest-path policy table
        hy_s = A.hypothesis_rdms(env_s, occ_s, GAMMA, extra={"shortest_path": occ_sp.Z_sa, "policy_stochastic": pol_stoch, "policy_shortest": pol_det})
        policy_change = float(np.mean([(sol_s.pi[g] > 0) != (sol.pi[g] > 0) for g in range(env.n_goals)]))
        nets_s, _, data_s = train_models(env_s, sol_s, "onehot", seeds6, steps)
        rep9 = []
        for n in nets_s:
            H = n.all_activations(); r = {}
            for layer in A.HIDDEN_LAYERS:
                X = A.state_views(H[layer])["mean"]; R = G.rdm(X)
                r[layer] = {
                    "rsa_occupancy": G.rsa(R, hy_s["occupancy"]), "rsa_shortest_path": G.rsa(R, hy_s["shortest_path"]),
                    "rsa_spatial": G.rsa(R, hy_s["spatial"]),
                    "partial_stoch_given_sp": G.partial_rsa(R, hy_s["occupancy"], [hy_s["shortest_path"]]),
                    "partial_sp_given_stoch": G.partial_rsa(R, hy_s["shortest_path"], [hy_s["occupancy"]]),
                    "decode_r2_stochastic": G.ridge_cv_r2(X, occ_s.Z_sa, alpha=1.0),
                    "decode_r2_shortest_path": G.ridge_cv_r2(X, occ_sp.Z_sa, alpha=1.0),
                    "rsa_policy_stochastic": G.rsa(R, hy_s["policy_stochastic"]),
                    "rsa_policy_shortest": G.rsa(R, hy_s["policy_shortest"]),
                    "partial_polstoch_given_poldet": G.partial_rsa(R, hy_s["policy_stochastic"], [hy_s["policy_shortest"]]),
                    "partial_poldet_given_polstoch": G.partial_rsa(R, hy_s["policy_shortest"], [hy_s["policy_stochastic"]]),
                }
            rep9.append(r)
        P.plot_stochastic(rep9, d9 / f"phase9_{variant}.png", variant)
        P.plot_env(env_s, sol_s, 5, d9 / f"env_{variant}.png")
        meta9 = {"rsa_stochastic_vs_shortest_path_hypotheses": G.rsa(hy_s["occupancy"], hy_s["shortest_path"]),
                 "rsa_policy_stochastic_vs_policy_shortest": G.rsa(hy_s["policy_stochastic"], hy_s["policy_shortest"]),
                 "rsa_stochastic_vs_spatial": G.rsa(hy_s["occupancy"], hy_s["spatial"]),
                 "fraction_state_action_support_changed": policy_change,
                 "train_accuracy": [evaluate_accuracy(n, *data_s) for n in nets_s]}
        report["phase9"][variant] = {"per_seed": rep9, "meta": meta9}
        print(f"[phase9] {variant}: hypotheses RSA(stoch, shortest)={meta9['rsa_stochastic_vs_shortest_path_hypotheses']:.3f}; acc={np.mean(meta9['train_accuracy']):.3f}; "
              f"h2 partial stoch|sp={np.mean([r['h2']['partial_stoch_given_sp'] for r in rep9]):+.2f}, sp|stoch={np.mean([r['h2']['partial_sp_given_stoch'] for r in rep9]):+.2f}  {time.time()-t0:.0f}s")
    dump(report["phase9"], d9 / "phase9.json")

    write_report_tables(report, out)
    print(f"done in {time.time()-t0:.0f}s -> {out}/")


def write_report_tables(report, out: Path):
    """Auto-generated numeric tables (rounds/r01_occupancy/tables.md); the narrative report is hand-written."""
    L = A.HIDDEN_LAYERS
    lines = ["# Auto-generated result tables\n"]

    def row(name, vals):
        return f"| {name} | " + " | ".join(f"{v:+.2f}" if isinstance(v, float) else str(v) for v in vals) + " |"

    def mean_over(reps, layer, key):
        return float(np.mean([r[layer][key] for r in reps]))

    s = report["phase3"]
    lines += ["## Phase 3: occupancy geometry (base)\n",
              f"- states {s['n_states']}, Z_sa dim {s['dim']}, participation ratio {s['participation_ratio']:.2f}, entropy rank {s['entropy_rank']:.2f}, PCs for 90% var {s['n_pcs_90pct']}",
              f"- mean pairwise cosine {s['mean_cosine']:.3f}",
              "- RSA between hypothesis RDMs: " + ", ".join(f"{k} {v:.2f}" for k, v in s["rsa_between_hypotheses"].items()), ""]
    for enc in ("onehot", "coord"):
        p4 = report[f"phase4_{enc}"]
        lines += [f"## Phase 4 ({enc}): accuracy {np.mean(p4['train_accuracy']):.4f}, rollout success {np.mean(p4['rollout_success']):.3f}, optimal-length fraction {np.nanmean(p4['rollout_optimal_fraction']):.3f}\n"]
    for enc in ("onehot", "coord"):
        p5 = report["phase5"][enc]
        tr, rd = p5["trained_mean_view"], p5["random_mean_view"]
        lines += [f"## Phase 5 ({enc} input), state view = mean over goals; mean over {len(tr)} seeds\n",
                  "| metric | " + " | ".join(L) + " |", "|---|" + "---|" * len(L)]
        for key in ["rsa_occupancy", "rsa_occupancy_state", "rsa_occupancy_log", "rsa_policy", "rsa_spatial", "rsa_geodesic", "rsa_SR", "rsa_identity",
                    "partial_occ_given_spatial", "partial_spatial_given_occ", "partial_occ_given_geodesic", "partial_geodesic_given_occ",
                    "cka_occupancy", "cka_spatial", "cka_SR", "decode_r2_occupancy", "decode_r2_spatial", "decode_r2_occupancy_pairs",
                    "participation_ratio", "subspace_overlap_k3", "subspace_overlap_k5"]:
            lines.append(row(key, [mean_over(tr, l, key) for l in L]))
        for key in ["rsa_occupancy", "rsa_spatial", "rsa_policy", "decode_r2_occupancy", "cka_occupancy", "participation_ratio"]:
            lines.append(row(f"{key} (random init)", [mean_over(rd, l, key) for l in L]))
        cc = p5["trained_concat_view"]
        for key in ["rsa_occupancy", "rsa_spatial", "rsa_SR", "rsa_policy", "decode_r2_occupancy"]:
            lines.append(row(f"{key} (concat-over-goals view)", [mean_over(cc, l, key) for l in L]))
        for l in L:
            rr = [r[l]["rdm_regression"] for r in tr]
            lines.append(row(f"RDM regression β @{l} (occ, spatial, geo, SR; R²)", [float(np.mean([x[k] for x in rr])) for k in ("beta_occupancy", "beta_spatial", "beta_geodesic", "beta_SR", "r2")]))
        pl = p5["pair_level"]
        lines += ["", f"Pair-level RSA ({enc}):", "| model | " + " | ".join(L) + " |", "|---|" + "---|" * len(L)]
        for key in ["rsa_pair_occupancy", "rsa_state_occupancy_only", "rsa_pair_Q", "rsa_pair_spatial", "rsa_pair_action", "rsa_goal_identity_only", "decode_r2_Q"]:
            lines.append(row(key, [mean_over(pl, l, key) for l in L]))
        gs = p5["gamma_sweep"]
        lines += ["", f"γ sweep ({enc}): RSA of h̄(s) with the occupancy RDM built at γ = {gs['gammas']}:"]
        for l in L:
            lines.append(row(f"  {l}", gs["rsa"][l]))
        em = p5["emergence_h2"]
        lines += ["", f"Emergence at h2 ({enc}), steps {em['steps']}:",
                  row("accuracy", em["acc"]), row("RSA occupancy", em["curves"]["rsa"]["occupancy"]), row("RSA spatial", em["curves"]["rsa"]["spatial"]),
                  row("decode R² occupancy", em["curves"]["decode"]["occupancy"]), ""]
    p6 = report["phase6"]
    lines += ["## Phase 6: discriminating environments (mean over seeds)\n"]
    for ename in ENV_NAMES:
        h = p6[f"{ename}_hypothesis_rsa"]
        lines.append(f"**{ename}**: RSA occupancy~spatial {h['occupancy~spatial']:.2f}, occupancy~geodesic {h['occupancy~geodesic']:.2f}, occupancy~policy {h['occupancy~policy']:.2f}, spatial~policy {h['spatial~policy']:.2f}")
        lines += ["| encoding | layer | acc | β_occ | β_spatial | β_geo | β_SR | R² | ρ_policy | β_occ (3-way: occ+sp+policy) | β_sp (3-way) | β_policy (3-way) | disagree ρ_occ | disagree ρ_sp | designed pairs | same-spatial ctrl | same-occ ctrl | random-init designed | random-init same-spatial |", "|---|" + "---|" * 19]
        for enc in ("onehot", "coord"):
            reps, rr = p6[(ename, enc)], p6[(ename, enc, "random")]
            for l in L:
                vals = [float(np.mean(p6[(ename, enc, "acc")]))] + [mean_over(reps, l, k) for k in ("beta_occupancy", "beta_spatial", "beta_geodesic", "beta_SR", "r2", "rsa_policy")]
                vals += [float(np.mean([r[l]["policy_regression"][k] for r in reps])) for k in ("beta_occupancy", "beta_spatial", "beta_policy")]
                vals += [mean_over(reps, l, k) for k in ("disagree_rsa_occupancy", "disagree_rsa_spatial")]
                if "special_pairs_median" in reps[0][l]:
                    vals += [mean_over(reps, l, k) for k in ("special_pairs_median", "special_pairs_spatial_matched_control", "special_pairs_occ_matched_control")]
                    vals += [mean_over(rr, l, k) for k in ("special_pairs_median", "special_pairs_spatial_matched_control")]
                else:
                    vals += [float("nan")] * 5
                lines.append(f"| {enc} | {l} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
        lines.append("")
    p7 = report["phase7"]["trained"]; p7r = report["phase7"]["random"]
    lines += ["## Phase 7: goal dependence (onehot, base)\n", "| metric | " + " | ".join(L) + " |", "|---|" + "---|" * len(L)]
    for key in ["state", "goal", "interaction", "state_component_rsa_occupancy", "state_component_rsa_spatial", "goal_component_rsa_goal_occupancy",
                "goal_component_rsa_goal_spatial", "interaction_rsa_Q", "interaction_rsa_action", "interaction_decode_r2_Q", "mean_cos_across_goals"]:
        lines.append(row(key, [mean_over(p7, l, key) for l in L]))
    for key in ["state", "goal", "interaction", "interaction_rsa_Q"]:
        lines.append(row(f"{key} (random init)", [mean_over(p7r, l, key) for l in L]))
    p8 = report["phase8"]
    lines += ["", "## Phase 8: steering (onehot, base; mean over seeds and goal pairs)\n",
              "| layer | agree g2 α=0 | α=0.5 | α=1 | α=2 | agree g1 α=1 | random dir α=1 | other-goal dir α=1 | flipped by α=1 | flipped by α=2 | corr(flip α, g2 margin) | corr(flip α, predicted α*) | MAE(flip α, predicted α*) |", "|---|" + "---|" * 12]
    for l in L:
        def m(k, i=None):
            v = [r["layers"][l][k] for r in p8]
            return float(np.nanmean([x[i] for x in v])) if i is not None else float(np.nanmean(v))
        lines.append(f"| {l} | {m('agree_g2',0):.2f} | {m('agree_g2',5):.2f} | {m('agree_g2',10):.2f} | {m('agree_g2',20):.2f} | {m('agree_g1',10):.2f} | {m('agree_g2_random_dir',10):.2f} | {m('agree_g2_other_goal_dir',10):.2f} | {m('frac_flipped_by_alpha1'):.2f} | {m('frac_flipped_by_alpha2'):.2f} | {m('flip_alpha_vs_margin_spearman'):+.2f} | {m('flip_alpha_vs_predicted_spearman'):+.2f} | {m('flip_alpha_vs_predicted_mae'):.2f} |")
    for variant, p9 in report["phase9"].items():
        lines += ["", f"## Phase 9: stochastic transitions ({variant}, onehot)\n",
                  f"- RSA between the two hypotheses themselves: {p9['meta']['rsa_stochastic_vs_shortest_path_hypotheses']:.3f}; between the two policy tables {p9['meta']['rsa_policy_stochastic_vs_policy_shortest']:.3f}; RSA exact~spatial {p9['meta']['rsa_stochastic_vs_spatial']:.3f}; fraction of (goal,state,action) optimal-support entries that changed vs deterministic: {p9['meta']['fraction_state_action_support_changed']:.3f}; train accuracy {np.mean(p9['meta']['train_accuracy']):.3f}",
                  "| metric | " + " | ".join(L) + " |", "|---|" + "---|" * len(L)]
        for key in ["rsa_occupancy", "rsa_shortest_path", "rsa_spatial", "rsa_policy_stochastic", "rsa_policy_shortest", "partial_stoch_given_sp", "partial_sp_given_stoch", "partial_polstoch_given_poldet", "partial_poldet_given_polstoch", "decode_r2_stochastic", "decode_r2_shortest_path"]:
            lines.append(row(key, [mean_over(p9["per_seed"], l, key) for l in L]))
    (out / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
