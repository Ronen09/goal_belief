"""Reproduce any experiment of the project with one command.

    python reproduce.py list                     # every experiment, by study: question, cost, dependencies
    python reproduce.py filter_state             # rerun one experiment end to end (training + measurement + tables)
    python reproduce.py filter_state --tables    # rebuild its tables and figures from its saved data
    python reproduce.py filter_state --quick     # a minutes-long smoke run into its _smoke/ directory
    python reproduce.py all [--quick|--tables]
    python reproduce.py check                    # tests + rebuild every table that has saved data + smoke runs;
                                                 # reports any table that differs from the committed one

Every step is a script inside the experiment's own directory (studies/<study>/<experiment>/), run from the
repository root. Experiments that reuse another experiment's models or pairs name it in `needs`; those inputs are
committed, so an experiment can be rerun on its own.
"""

from __future__ import annotations

import argparse, os, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


@dataclass
class Step:
    script: str                     # file inside the experiment directory
    args: tuple = ()
    quick: bool = False             # accepts --quick --out DIR
    tables: bool = False            # rebuilds tables/figures from saved data (no training)


@dataclass
class Experiment:
    key: str
    dir: str
    question: str
    cost: str
    steps: list = field(default_factory=list)
    needs: tuple = ()


EXPERIMENTS = [
    Experiment("occupancy", "1_representation_geometry/occupancy", "Does a goal-conditioned policy learn goal-occupancy geometry?", "6 min CPU",
          [Step("run.py", quick=True)]),
    Experiment("policy_quotient", "1_representation_geometry/policy_quotient", "Policy quotient vs occupancy geometry; soft targets", "15 min CPU",
          [Step("run.py", quick=True), Step("run.py", ("--targets", "boltzmann", "--sweep-only", "--out", "studies/1_representation_geometry/policy_quotient/boltzmann"))]),
    Experiment("supervision", "1_representation_geometry/supervision", "The supervision bottleneck: what the target keeps", "18 min CPU",
          [Step("run.py", quick=True)]),
    Experiment("hmm_objectives", "1_representation_geometry/hmm_objectives", "One-step vs k-step vs sequential objectives on an HMM", "5 min",
          [Step("run.py", ("--device", "cuda"), quick=True)]),
    Experiment("prominence", "1_representation_geometry/prominence", "What makes information geometrically prominent?", "20 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8", "--grid", "--dynamics"), quick=True)]),
    Experiment("readout_scale", "1_representation_geometry/readout_scale", "Readout gain vs hidden separation", "8 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8"), quick=True)]),
    Experiment("allocation", "1_representation_geometry/allocation", "What sets the gain/separation split", "6 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8"), quick=True)]),
    Experiment("invariants", "1_representation_geometry/invariants", "Which representation measures are invariant", "13 min, 8 CPU workers",
          [Step("run.py", ("--jobs", "8"), quick=True), Step("followup.py")]),
    Experiment("interventions", "1_representation_geometry/interventions", "Intervention equivalence and local metrics", "5 min",
          [Step("run.py", ("--jobs", "6"), quick=True)], needs=("invariants",)),
    Experiment("transformer", "1_representation_geometry/transformer", "Transformer reproduction of the prominence to intervention-equivalence experiments", "10 min GPU",
          [Step("run.py", ("--jobs", "4"), quick=True), Step("followup.py")]),
    Experiment("implementation_freedom", "1_representation_geometry/implementation_freedom", "Cut identifiability and factorisation freedom", "1.5 h GPU",
          [Step("run.py", ("--exp", "all"), quick=True), Step("followup.py")]),
    Experiment("hidden_goal", "2_belief_state/hidden_goal", "A hidden goal: is the posterior affinely recoverable, and is it the causal state?",
          "40 min GPU + CPU, then 6 min", [Step("run.py", quick=True), Step("followup.py"), Step("tables.py", tables=True),
                                          Step("causal_run.py", quick=True), Step("causal_tables.py", tables=True)]),
    Experiment("filter_state", "2_belief_state/filter_state", "Is the recurrent state the environment's minimal predictive state?", "7 min, 96 CPU workers",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("hidden_goal",)),
    Experiment("window_carry", "2_belief_state/window_carry", "Does a narrow attention window force a steerable belief state?", "31 min CPU workers",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("hidden_goal", "filter_state")),
    Experiment("prior_vs_recompute", "2_belief_state/prior_vs_recompute", "Does a next-token transformer use its previous belief as a prior?", "11 min GPU",
          [Step("run.py", quick=True), Step("recompute.py", tables=True), Step("tables.py", tables=True)]),
    Experiment("kv_dropout", "2_belief_state/kv_dropout", "Can K/V dropout induce recurrence continuously?", "1 h GPU + CPU workers",
          [Step("run.py", quick=True),
           Step("run.py", ("--out", "studies/2_belief_state/kv_dropout/fine", "--ps", "0.05", "0.1", "0.2", "0.3", "--plain-ps", "0.1", "0.25")),
           Step("tables.py", tables=True)], needs=("prior_vs_recompute",)),
    Experiment("read_cost", "2_belief_state/read_cost", "Does a price on reading the history induce a selective, recurrent belief state?", "3 h (32 min GPU + carry family on CPU workers)",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("kv_dropout",)),
    Experiment("navigate_commit", "3_reward_trained_agents/navigate_commit", "How does a full-attention transformer come to build and use a belief while its policy improves?",
          "4 h GPU",
          [Step("tune.py", quick=True),
           Step("train.py", ("--cond", "ppo", "--q", "fixed", "--seeds", "0", "1", "2", "3", "4", "--updates", "3000",
                             "--dense", "600", "1800", "100", "--out", "studies/3_reward_trained_agents/navigate_commit/runs/ppo_fixed"), quick=True),
           Step("train.py", ("--cond", "frozen", "--q", "fixed", "--seeds", "0", "1", "--updates", "2000",
                             "--out", "studies/3_reward_trained_agents/navigate_commit/runs/frozen_fixed")),
           Step("train.py", ("--cond", "ppo", "--q", "grid", "--seeds", "0", "1", "2", "--updates", "3500",
                             "--out", "studies/3_reward_trained_agents/navigate_commit/runs/ppo_grid")),
           Step("train.py", ("--cond", "supervised", "--q", "fixed", "--updates", "1500", "--out", "studies/3_reward_trained_agents/navigate_commit/runs/sup_fixed")),
           Step("train.py", ("--cond", "supervised", "--q", "grid", "--updates", "1500", "--out", "studies/3_reward_trained_agents/navigate_commit/runs/sup_grid")),
           Step("measure_all.py"), Step("tables.py", tables=True)]),
    Experiment("maze_belief", "3_reward_trained_agents/maze_belief", "Does an inferred location belief support goal-dependent decisions?", "1.5 h GPU + CPU",
          [Step("checks.py", quick=True),
           Step("train.py", ("--cond", "ppo", "--seeds", "0", "1", "2", "3", "4", "--updates", "1500", "--out", "studies/3_reward_trained_agents/maze_belief/runs/ppo"), quick=True),
           Step("train.py", ("--cond", "frozen", "--seeds", "0", "1", "--updates", "1000", "--out", "studies/3_reward_trained_agents/maze_belief/runs/frozen")),
           Step("train.py", ("--cond", "supervised", "--updates", "1000", "--lr", "3e-4", "--out", "studies/3_reward_trained_agents/maze_belief/runs/supervised")),
           Step("measure_all.py"), Step("tables.py", tables=True)]),
    Experiment("maze_occupancy", "3_reward_trained_agents/maze_occupancy", "Is goal-conditioned occupancy represented beyond the posterior and the action values?", "25 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("maze_belief",)),
    Experiment("belief_edit", "3_reward_trained_agents/belief_edit", "Does the policy use the decoded belief? Edits at the goal-free prefix interface", "20 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("maze_belief",)),
    Experiment("pattern_specificity", "3_reward_trained_agents/pattern_specificity", "Is the Sigma-W edit specific? Replication with PCA, random and posterior-matched controls", "2 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("maze_belief", "belief_edit")),
    Experiment("pair_types", "3_reward_trained_agents/pair_types", "Three pair types: same posterior; same action under one goal; different action. Whole and PCA patches, all goals", "2 min GPU",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("maze_belief", "pattern_specificity")),
    Experiment("obs_prediction", "3_reward_trained_agents/obs_prediction", "Reward only against reward plus next-symbol prediction, on the pair-types experiment's fixed pairs", "2.5 h GPU",
          [Step("train.py", ("--aux", "0", "--seeds", *map(str, range(10)), "--out", "studies/3_reward_trained_agents/obs_prediction/runs/reward"), quick=True),
           Step("train.py", ("--aux", "1.0", "--seeds", *map(str, range(10)), "--out", "studies/3_reward_trained_agents/obs_prediction/runs/aux1")),
           Step("train.py", ("--aux", "0.1", "--seeds", *map(str, range(10)), "--out", "studies/3_reward_trained_agents/obs_prediction/runs/aux01")),
           Step("measure.py"), Step("tables.py", tables=True)], needs=("maze_belief", "pair_types")),
    Experiment("head_consistency", "3_reward_trained_agents/head_consistency", "Does the prediction head agree on identical-posterior histories?", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("pair_types", "obs_prediction")),
    Experiment("balanced_prediction", "3_reward_trained_agents/balanced_prediction", "Selected-action against balanced candidate-action supervision of the prediction head", "1.5 h GPU",
          [Step("train.py", ("--aux", "1.0", "--sup", "all", "--seeds", *map(str, range(10)), "--out", "studies/3_reward_trained_agents/balanced_prediction/runs/all"), quick=True),
           Step("train.py", ("--aux", "1.0", "--sup", "one", "--seeds", *map(str, range(10)), "--out", "studies/3_reward_trained_agents/balanced_prediction/runs/one")),
           Step("measure.py"), Step("matched.py"), Step("tables.py", tables=True)], needs=("obs_prediction", "head_consistency")),
    Experiment("predictive_transfer", "4_predictive_pretraining/predictive_transfer", "Prediction-only, reward-only and random backbones, frozen, under a small goal-conditioned head trained on optimal actions", "30 min GPU",
          [Step("checks.py"),
           Step("train_pred.py", ("--k", "1", "--seeds", *map(str, range(10)), "--out", "studies/4_predictive_pretraining/predictive_transfer/runs/predict1"), quick=True),
           Step("train_pred.py", ("--k", "2", "--seeds", *map(str, range(10)), "--out", "studies/4_predictive_pretraining/predictive_transfer/runs/predict2")),
           Step("measure.py"), Step("tables.py", tables=True)], needs=("pair_types", "obs_prediction")),
    Experiment("belief_encoding_edit", "4_predictive_pretraining/belief_encoding_edit", "Belief-encoding edits of the frozen state, read by the predictive-transfer experiment's frozen goal-conditioned heads", "5 min GPU",
          [Step("heads.py"), Step("run.py", ("--counts",)), Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer",)),
    Experiment("policy_belief_edit", "4_predictive_pretraining/policy_belief_edit", "One belief-encoding edit at the pre-goal interface, read by the original policy and by the new head", "3 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit")),
    Experiment("goal_route_selection", "5_goal_belief_mechanism/goal_route_selection", "Does the goal change which evidence route (pre-goal interface or direct route) the policy relies on?", "2 min GPU",
          [Step("run.py"), Step("posthoc.py"), Step("tables.py", tables=True)], needs=("belief_encoding_edit", "policy_belief_edit")),
    Experiment("goal_swap_components", "5_goal_belief_mechanism/goal_swap_components", "Goal swap with the history fixed: which heads or MLPs after the interface carry the switch?", "4 min GPU",
          [Step("run.py"), Step("posthoc.py"), Step("tables.py", tables=True)], needs=("predictive_transfer",)),
    Experiment("cross_history_mlp", "5_goal_belief_mechanism/cross_history_mlp", "Cross-history patches of the goal token's MLP outputs: goal instruction, evidence-goal combination, or action preference?", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "goal_swap_components")),
    Experiment("direct_belief_edit", "5_goal_belief_mechanism/direct_belief_edit", "A shared belief encoding at the goal token's state after block 0, edited with the donor's posterior under all goals", "2 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit")),
    Experiment("attention_belief_edit", "5_goal_belief_mechanism/attention_belief_edit", "The direct-belief-edit experiment's belief edit one step earlier, before block 0's MLP: is the belief map shared across goals there?", "2 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "direct_belief_edit")),
    Experiment("nonlinear_belief_edit", "5_goal_belief_mechanism/nonlinear_belief_edit", "Nonlinear (MLP, per-posterior table) encodings of the posterior before block 0's MLP; goal-dependent pairs", "6 min GPU",
          [Step("run.py"), Step("posthoc.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "direct_belief_edit", "attention_belief_edit")),
    Experiment("block0_steps", "5_goal_belief_mechanism/block0_steps", "The goal x belief part at each step from block 0's attention output (self, prefix, heads) through the layer norm to the MLP input", "2 min GPU",
          [Step("run.py"), Step("identity_check.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "direct_belief_edit")),
    Experiment("query_swap", "5_goal_belief_mechanism/query_swap", "Swap block 0's goal-token query, or its own key and value, to another goal's: representation and decisions", "1 min GPU",
          [Step("run.py"), Step("run.py", ("--untrained",)), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit")),
    Experiment("self_value", "5_goal_belief_mechanism/self_value", "The goal token's own value against its own key in block 0, per head; the residual embedding as the complement", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "query_swap")),
    Experiment("self_value_interaction", "5_goal_belief_mechanism/self_value_interaction", "Does the self-value swap move the goal x belief part I(b,g) = f(b,g) - f(b) at block 0's MLP input, and after it?", "20 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "query_swap", "self_value")),
    Experiment("mlp_bilinear", "5_goal_belief_mechanism/mlp_bilinear", "How block 0's MLP computes J(b,g): low-rank bilinear description, second-order mechanism, hidden units, ablation", "30 min GPU",
          [Step("run.py"), Step("run.py", ("--untrained",)), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "query_swap")),
    Experiment("mlp_depth", "5_goal_belief_mechanism/mlp_depth", "The goal x belief part at every MLP of the goal token: bilinear description, own share, removal from one or all four", "55 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "query_swap", "mlp_bilinear")),
    Experiment("interaction_removal", "5_goal_belief_mechanism/interaction_removal", "Per-history removal of the goal x history interaction from the goal token's MLP and attention outputs", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "query_swap", "mlp_bilinear")),
    Experiment("additive_code", "5_goal_belief_mechanism/additive_code", "The additive code at the goal token: logits ~ H(h) + G(g, L), and where additivity is impossible", "20 s GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer",)),
    Experiment("what_is_H", "5_goal_belief_mechanism/what_is_H", "What is the history profile H: max_g Q*, goal-averaged values, P(optimal), reachability? Prediction and causal edits", "3 min GPU",
          [Step("run.py"), Step("run.py", ("--untrained",)), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "additive_code")),
    Experiment("H_mixture", "5_goal_belief_mechanism/H_mixture", "H as a mixture of P(optimal) and reachability; goal weights inside H against uniform training frequency", "1 min GPU",
          [Step("run.py"), Step("tables.py", tables=True)], needs=("predictive_transfer", "belief_encoding_edit", "additive_code", "what_is_H")),
    Experiment("hard_cases", "6_hard_cases_and_tasks/hard_cases", "Exploratory: hallmarks of the additively unsolvable cases, more of them after the reveal, fine-tuning and retraining on them, a matched-episode simulation, and a model-free screen of harder tasks (multi-goal collection)", "3.5 h GPU",
          [Step("hallmarks.py"), Step("hallmark_beliefs.py"), Step("later_cases.py"), Step("finetune.py"), Step("finetune_diag.py"),
           Step("train_cases.py", ("--arm", "targeted")), Step("train_cases.py", ("--arm", "control")), Step("eval_scratch.py"), Step("policy_diff.py"),
           Step("sim_data.py", ("r23", "0")), Step("sim_data.py", ("targeted", "0")), Step("sim_data.py", ("control", "0")), Step("build_sim.py", tables=True),
           Step("hardness.py"), Step("screen.py", ("baseline",)), Step("screen.py", ("search",)), Step("screen_multi.py")], needs=("obs_prediction", "additive_code")),
]
BY_KEY = {r.key: r for r in EXPERIMENTS}


def run_step(r: Experiment, s: Step, extra=()):
    cmd = [PY, str(ROOT / "studies" / r.dir / s.script), *s.args, *extra]
    print(f"\n[{r.key}] $ {' '.join(os.path.relpath(c, ROOT) if c.startswith(str(ROOT)) else c for c in cmd[1:])}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    print(f"[{r.key}] {s.script} {'ok' if rc == 0 else f'FAILED ({rc})'} in {time.time() - t0:.0f}s", flush=True)
    return rc == 0


def run_experiment(r: Experiment, mode: str):
    ok = True
    for s in r.steps:
        if mode == "full":
            ok &= run_step(r, s)
        elif mode == "tables" and s.tables:
            ok &= run_step(r, s)
        elif mode == "quick" and s.quick:
            out = ROOT / "studies" / r.dir / "_smoke" / Path(s.script).stem
            out.mkdir(parents=True, exist_ok=True)
            args = [a for a in s.args]
            if "--out" in args:                            # a quick run never writes into committed results
                i = args.index("--out"); del args[i:i + 2]
            ok &= run_step(r, Step(s.script, tuple(args)), ("--quick", "--out", str(out)))
    return ok


def check():
    ok = subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT).returncode == 0
    for r in EXPERIMENTS:
        ok &= run_experiment(r, "tables")
    diff = subprocess.run(["git", "diff", "--stat", "--", "studies/*/*/tables.md", "studies/*/*/*/tables.md"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    print("\nrebuilt tables identical to the committed ones" if not diff else f"\nTABLES CHANGED:\n{diff}")
    ok &= not diff
    for r in EXPERIMENTS:
        ok &= run_experiment(r, "quick")
    print("\nCHECK", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="an experiment name (see list), all, list or check")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="smoke runs into studies/<study>/<experiment>/_smoke/")
    g.add_argument("--tables", action="store_true", help="rebuild tables and figures from saved data only")
    a = ap.parse_args()
    if a.target == "list":
        print(f"{'experiment':24} {'cost':32} {'needs':28} question")
        study = None
        for r in EXPERIMENTS:
            if r.dir.split("/")[0] != study:
                study = r.dir.split("/")[0]; print(f"\n{study}")
            print(f"  {r.key:22} {r.cost:32} {','.join(r.needs) or '-':28} {r.question}")
        return
    if a.target == "check":
        sys.exit(0 if check() else 1)
    mode = "quick" if a.quick else "tables" if a.tables else "full"
    targets = EXPERIMENTS if a.target == "all" else [BY_KEY[a.target]]
    ok = all([run_experiment(r, mode) for r in targets])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
