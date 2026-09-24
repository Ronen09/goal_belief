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
    Round("r17", "r17_read_cost", "Does a price on reading the history induce a selective, recurrent belief state?", "1 h GPU + CPU workers",
          [Step("run.py", quick=True), Step("tables.py", tables=True)], needs=("r16",)),
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
