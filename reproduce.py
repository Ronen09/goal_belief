"""Reproduce any round of the project with one command.

    python reproduce.py list                 # every round: question, cost, dependencies
    python reproduce.py r13                  # rerun round 13 end to end (training + measurement + tables)
    python reproduce.py r13 --tables         # rebuild round 13's tables and figures from its saved data
    python reproduce.py r13 --quick          # a minutes-long smoke run into rounds/r13_*/_smoke/
    python reproduce.py all [--quick|--tables]
    python reproduce.py check                # tests + rebuild every table that has saved data + smoke runs;
                                             # reports any table that differs from the committed one

Every step is a script inside the round's own directory, run from the repository root. Rounds that
reuse another round's models or pairs name it in `needs`; those inputs are committed, so a round can
be rerun on its own.
"""

from __future__ import annotations

import argparse, os, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


@dataclass
class Step:
    script: str                     # file inside the round directory
    args: tuple = ()
    quick: bool = False             # accepts --quick --out DIR
    tables: bool = False            # rebuilds tables/figures from saved data (no training)


@dataclass
class Round:
    key: str
    dir: str
    question: str
    cost: str
    steps: list = field(default_factory=list)
    needs: tuple = ()


ROUNDS = [
    Round("r01", "r01_occupancy", "Does a goal-conditioned policy learn goal-occupancy geometry?", "6 min CPU",
          [Step("run.py", quick=True)]),
    Round("r02", "r02_policy_quotient", "Policy quotient vs occupancy geometry; soft targets", "15 min CPU",
          [Step("run.py", quick=True), Step("run.py", ("--targets", "boltzmann", "--sweep-only", "--out", "rounds/r02_policy_quotient/boltzmann"))]),
    Round("r03", "r03_supervision", "The supervision bottleneck: what the target keeps", "18 min CPU",
          [Step("run.py", quick=True)]),
    Round("r04", "r04_hmm_objectives", "One-step vs k-step vs sequential objectives on an HMM", "5 min",
          [Step("run.py", ("--device", "cuda"), quick=True)]),
    Round("r05", "r05_prominence", "What makes information geometrically prominent?", "20 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8", "--grid", "--dynamics"), quick=True)]),
    Round("r06", "r06_readout_scale", "Readout gain vs hidden separation", "8 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8"), quick=True)]),
    Round("r07", "r07_allocation", "What sets the gain/separation split", "6 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8"), quick=True)]),
    Round("r08", "r08_invariants", "Which representation measures are invariant", "13 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8"), quick=True), Step("followup.py")]),
    Round("r09", "r09_interventions", "Intervention equivalence and local metrics", "5 min",
          [Step("run.py", ("--jobs", "6"), quick=True)], needs=("r08",)),
    Round("r10", "r10_transformer", "Transformer reproduction of rounds 5-9", "10 min GPU",
          [Step("run.py", ("--jobs", "4"), quick=True), Step("followup.py")]),
    Round("r11", "r11_implementation_freedom", "Cut identifiability and factorisation freedom", "1.5 h GPU",
          [Step("run.py", ("--exp", "all"), quick=True), Step("followup.py")]),
    Round("r12", "r12_hidden_goal", "A hidden goal: is the posterior affinely recoverable, and is it the causal state?",
          "40 min GPU + CPU, then 6 min", [Step("run.py", quick=True), Step("followup.py"), Step("tables.py", tables=True),
                                          Step("causal_run.py", quick=True), Step("causal_tables.py", tables=True)]),
    Round("r13", "r13_filter_state", "Is the recurrent state the environment's minimal predictive state?", "7 min, 96 CPU workers",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r12",)),
    Round("r14", "r14_window_carry", "Does a narrow attention window force a steerable belief state?", "31 min CPU workers",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r12", "r13")),
    Round("r15", "r15_prior_vs_recompute", "Does a next-token transformer use its previous belief as a prior?", "11 min GPU",
          [Step("run.py", quick=True), Step("recompute.py", tables=True), Step("tables.py", tables=True)]),
    Round("r16", "r16_kv_dropout", "Can K/V dropout induce recurrence continuously?", "1 h GPU + CPU workers",
          [Step("run.py", quick=True),
           Step("run.py", ("--out", "rounds/r16_kv_dropout/fine", "--ps", "0.05", "0.1", "0.2", "0.3", "--plain-ps", "0.1", "0.25")),
           Step("tables.py", tables=True)], needs=("r15",)),
    Round("r17", "r17_navigate_commit", "How does a full-attention transformer come to build and use a belief while its policy improves?",
          "4 h GPU",
          [Step("tune.py", quick=True),
           Step("train.py", ("--cond", "ppo", "--q", "fixed", "--seeds", "0", "1", "2", "3", "4", "--updates", "3000",
                             "--dense", "600", "1800", "100", "--out", "rounds/r17_navigate_commit/runs/ppo_fixed"), quick=True),
           Step("train.py", ("--cond", "frozen", "--q", "fixed", "--seeds", "0", "1", "--updates", "2000",
                             "--out", "rounds/r17_navigate_commit/runs/frozen_fixed")),
           Step("train.py", ("--cond", "ppo", "--q", "grid", "--seeds", "0", "1", "2", "--updates", "3500",
                             "--out", "rounds/r17_navigate_commit/runs/ppo_grid")),
           Step("train.py", ("--cond", "supervised", "--q", "fixed", "--updates", "1500", "--out", "rounds/r17_navigate_commit/runs/sup_fixed")),
           Step("train.py", ("--cond", "supervised", "--q", "grid", "--updates", "1500", "--out", "rounds/r17_navigate_commit/runs/sup_grid")),
           Step("measure_all.py"), Step("tables.py", tables=True)]),
    Round("r18", "r18_maze_belief", "Does an inferred location belief support goal-dependent decisions?", "1.5 h GPU + CPU",
          [Step("checks.py", quick=True),
           Step("train.py", ("--cond", "ppo", "--seeds", "0", "1", "2", "3", "4", "--updates", "1500", "--out", "rounds/r18_maze_belief/runs/ppo"), quick=True),
           Step("train.py", ("--cond", "frozen", "--seeds", "0", "1", "--updates", "1000", "--out", "rounds/r18_maze_belief/runs/frozen")),
           Step("train.py", ("--cond", "supervised", "--updates", "1000", "--lr", "3e-4", "--out", "rounds/r18_maze_belief/runs/supervised")),
           Step("measure_all.py"), Step("tables.py", tables=True)]),
    Round("r19", "r19_occupancy", "Is goal-conditioned occupancy represented beyond the posterior and the action values?", "25 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r18",)),
    Round("r20", "r20_belief_edit", "Does the policy use the decoded belief? Edits at the goal-free prefix interface", "20 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r18",)),
    Round("r21", "r21_pattern_specificity", "Is the Sigma-W edit specific? Replication with PCA, random and posterior-matched controls", "2 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r18", "r20")),
    Round("r22", "r22_pair_types", "Three pair types: same posterior; same action under one goal; different action. Whole and PCA patches, all goals", "2 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r18", "r21")),
    Round("r23", "r23_obs_prediction", "Reward only against reward plus next-symbol prediction, on round 22's fixed pairs", "2.5 h GPU",
          [Step("train.py", ("--aux", "0", "--seeds", *map(str, range(10)), "--out", "rounds/r23_obs_prediction/runs/reward"), quick=True),
           Step("train.py", ("--aux", "1.0", "--seeds", *map(str, range(10)), "--out", "rounds/r23_obs_prediction/runs/aux1")),
           Step("train.py", ("--aux", "0.1", "--seeds", *map(str, range(10)), "--out", "rounds/r23_obs_prediction/runs/aux01")),
           Step("measure.py"), Step("tables.py", tables=True)], needs=("r18", "r22")),
    Round("r24", "r24_head_consistency", "Does the prediction head agree on identical-posterior histories?", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r22", "r23")),
    Round("r25", "r25_balanced_prediction", "Selected-action against balanced candidate-action supervision of the prediction head", "1.5 h GPU",
          [Step("train.py", ("--aux", "1.0", "--sup", "all", "--seeds", *map(str, range(10)), "--out", "rounds/r25_balanced_prediction/runs/all"), quick=True),
           Step("train.py", ("--aux", "1.0", "--sup", "one", "--seeds", *map(str, range(10)), "--out", "rounds/r25_balanced_prediction/runs/one")),
           Step("measure.py"), Step("matched.py"), Step("tables.py", tables=True)], needs=("r23", "r24")),
    Round("r26", "r26_predictive_transfer", "Prediction-only, reward-only and random backbones, frozen, under a small goal-conditioned head trained on optimal actions", "30 min GPU",
          [Step("checks.py"),
           Step("train_pred.py", ("--k", "1", "--seeds", *map(str, range(10)), "--out", "rounds/r26_predictive_transfer/runs/predict1"), quick=True),
           Step("train_pred.py", ("--k", "2", "--seeds", *map(str, range(10)), "--out", "rounds/r26_predictive_transfer/runs/predict2")),
           Step("measure.py"), Step("tables.py", tables=True)], needs=("r22", "r23")),
    Round("r27", "r27_belief_encoding_edit", "Belief-encoding edits of the frozen state, read by round 26's frozen goal-conditioned heads", "5 min GPU",
          [Step("heads.py"), Step("run.py", ("--counts",)), Step("run.py"), Step("tables.py", tables=True)], needs=("r26",)),
    Round("r28", "r28_policy_belief_edit", "One belief-encoding edit at the pre-goal interface, read by the original policy and by the new head", "3 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27")),
    Round("r29", "r29_goal_route_selection", "Does the goal change which evidence route (pre-goal interface or direct route) the policy relies on?", "2 min GPU",
          [Step("run.py"), Step("posthoc.py"), Step("tables.py", tables=True)], needs=("r27", "r28")),
    Round("r30", "r30_goal_swap_components", "Goal swap with the history fixed: which heads or MLPs after the interface carry the switch?", "4 min GPU",
          [Step("run.py"), Step("posthoc.py"), Step("tables.py", tables=True)], needs=("r26",)),
    Round("r31", "r31_cross_history_mlp", "Cross-history patches of the goal token's MLP outputs: goal instruction, evidence-goal combination, or action preference?", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r30")),
    Round("r32", "r32_direct_belief_edit", "A shared belief encoding at the goal token's state after block 0, edited with the donor's posterior under all goals", "2 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27")),
    Round("r33", "r33_attention_belief_edit", "Round 32's belief edit one step earlier, before block 0's MLP: is the belief map shared across goals there?", "2 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r32")),
    Round("r34", "r34_nonlinear_belief_edit", "Nonlinear (MLP, per-posterior table) encodings of the posterior before block 0's MLP; goal-dependent pairs", "6 min GPU",
          [Step("run.py"), Step("posthoc.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r32", "r33")),
    Round("r35", "r35_block0_steps", "The goal x belief part at each step from block 0's attention output (self, prefix, heads) through the layer norm to the MLP input", "2 min GPU",
          [Step("run.py"), Step("identity_check.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r32")),
    Round("r36", "r36_query_swap", "Swap block 0's goal-token query, or its own key and value, to another goal's: representation and decisions", "1 min GPU",
          [Step("run.py"), Step("run.py", ("--untrained",)), Step("tables.py", tables=True)], needs=("r26", "r27")),
    Round("r37", "r37_self_value", "The goal token's own value against its own key in block 0, per head; the residual embedding as the complement", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r36")),
    Round("r38", "r38_self_value_interaction", "Does the self-value swap move the goal x belief part I(b,g) = f(b,g) - f(b) at block 0's MLP input, and after it?", "20 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r36", "r37")),
    Round("r39", "r39_mlp_bilinear", "How block 0's MLP computes J(b,g): low-rank bilinear description, second-order mechanism, hidden units, ablation", "30 min GPU",
          [Step("run.py"), Step("run.py", ("--untrained",)), Step("tables.py", tables=True)], needs=("r26", "r27", "r36")),
    Round("r40", "r40_mlp_depth", "The goal x belief part at every MLP of the goal token: bilinear description, own share, removal from one or all four", "55 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r36", "r39")),
    Round("r41", "r41_interaction_removal", "Per-history removal of the goal x history interaction from the goal token's MLP and attention outputs", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26", "r27", "r36", "r39")),
    Round("r42", "r42_additive_code", "The additive code at the goal token: logits ~ H(h) + G(g, L), and where additivity is impossible", "20 s GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("r26",)),
    Round("r43", "r43_what_is_H", "What is the history profile H: max_g Q*, goal-averaged values, P(optimal), reachability? Prediction and causal edits", "3 min GPU",
          [Step("run.py"), Step("run.py", ("--untrained",)), Step("tables.py", tables=True)], needs=("r26", "r27", "r42")),
]
BY_KEY = {r.key: r for r in ROUNDS}


def run_step(r: Round, s: Step, extra=()):
    cmd = [PY, str(ROOT / "rounds" / r.dir / s.script), *s.args, *extra]
    print(f"\n[{r.key}] $ {' '.join(os.path.relpath(c, ROOT) if c.startswith(str(ROOT)) else c for c in cmd[1:])}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    print(f"[{r.key}] {s.script} {'ok' if rc == 0 else f'FAILED ({rc})'} in {time.time() - t0:.0f}s", flush=True)
    return rc == 0


def run_round(r: Round, mode: str):
    ok = True
    for s in r.steps:
        if mode == "full":
            ok &= run_step(r, s)
        elif mode == "tables" and s.tables:
            ok &= run_step(r, s)
        elif mode == "quick" and s.quick:
            out = ROOT / "rounds" / r.dir / "_smoke" / Path(s.script).stem
            out.mkdir(parents=True, exist_ok=True)
            args = [a for a in s.args]
            if "--out" in args:                            # a quick run never writes into committed results
                i = args.index("--out"); del args[i:i + 2]
            ok &= run_step(r, Step(s.script, tuple(args)), ("--quick", "--out", str(out)))
    return ok


def check():
    ok = subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT).returncode == 0
    for r in ROUNDS:
        ok &= run_round(r, "tables")
    diff = subprocess.run(["git", "diff", "--stat", "--", "rounds/*/tables.md", "rounds/*/*/tables.md"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    print("\nrebuilt tables identical to the committed ones" if not diff else f"\nTABLES CHANGED:\n{diff}")
    ok &= not diff
    for r in ROUNDS:
        ok &= run_round(r, "quick")
    print("\nCHECK", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="rNN, all, list or check")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="smoke runs into rounds/<round>/_smoke/")
    g.add_argument("--tables", action="store_true", help="rebuild tables and figures from saved data only")
    a = ap.parse_args()
    if a.target == "list":
        print(f"{'round':6} {'cost':32} {'needs':9} question")
        for r in ROUNDS:
            print(f"{r.key:6} {r.cost:32} {','.join(r.needs) or '-':9} {r.question}")
        return
    if a.target == "check":
        sys.exit(0 if check() else 1)
    mode = "quick" if a.quick else "tables" if a.tables else "full"
    targets = ROUNDS if a.target == "all" else [BY_KEY[a.target]]
    ok = all([run_round(r, mode) for r in targets])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
