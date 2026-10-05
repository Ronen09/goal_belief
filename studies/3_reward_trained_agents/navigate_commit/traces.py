"""The navigate-commit experiment: worked computational traces. For a few decisions of one trained model: the history, the exact belief
and optimal actions, the model's action distribution, the decoded marginals along the residual stream (one shared
decoder per block), what each head moves from each report token, and the same for the twin with one report
flipped. Writes traces.md in the run directory."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from goalgeo import navcausal as C, navcommit as NC, navmodel as NM, navprobe as PR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as M, train as T                                      # noqa: E402

REP = ("", "LEFT", "RIGHT", "TOP", "BOTTOM")
CORNER = ("top-left", "top-right", "bottom-left", "bottom-right")


def cellname(c):
    return f"({c // 5},{c % 5})"


def history(tok, q, pos):
    out = [f"H station at {cellname(tok[0, NM.F_CELL].item() - 1)} (q = {q[0].item():.2f}), "
           f"V station at {cellname(tok[1, NM.F_CELL].item() - 1)} (q = {q[1].item():.2f})"]
    steps = []
    for p in range(2, pos + 1):
        t = tok[p]
        if t[NM.F_TYPE] == NM.DEC:
            steps.append(f"[{p}] at {cellname(t[NM.F_CELL].item() - 1)}, {t[NM.F_REM].item() - 1} left")
        else:
            a = NC.ACTION_NAMES[t[NM.F_ACT].item() - 1]
            steps.append(f"[{p}] {a}" + (f" → **{REP[t[NM.F_REP].item()]}**" if t[NM.F_REP] > 0 else ""))
    return out + [" · ".join(steps)]


def dist(p, legal):
    return ", ".join(f"{NC.ACTION_NAMES[a]} {p[a]:.2f}" for a in range(6) if legal[a] and p[a] >= 0.005)


@torch.no_grad()
def trace(net, decs, pairs, i, title):
    n = torch.tensor([i], device=pairs["tok"].device)
    L = [f"## {title}", ""]
    for label, tok in (("history", pairs["tok"][n]), ("twin (one report flipped)", pairs["tok_twin"][n])):
        q, pos = pairs["q"][n], pairs["pos"][n]
        lg, _, rec = net(tok, q, record=True)
        legal = pairs["legal"][i]
        p = lg[0, pos[0]].masked_fill(~legal, -1e9).softmax(-1)
        twin = label != "history"
        marg = pairs["marg_cf" if twin else "marg"][i]
        opt = pairs["opt_cf" if twin else "opt"][i]
        L += [f"**{label}.** " + history(tok[0], q[0], pos[0].item())[0], "", history(tok[0], q[0], pos[0].item())[1], "",
              f"Exact: P(right) = {marg[0]:.2f}, P(bottom) = {marg[1]:.2f}; optimal: "
              + ", ".join(NC.ACTION_NAMES[a] for a in range(6) if opt[a]) + f". Model: {dist(p, legal)}.", "",
              "| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |", "|---|---|---|---|"]
        for l in range(net.nl):
            d = decs[l]
            g = lambda x: d(x[0, pos[0]][None])[0, 4:6]
            a, b_, c_ = g(rec["resid"][l]), g(rec["mid"][l]), g(rec["resid"][l + 1])
            L.append(f"| {l} | ({a[0]:.2f}, {a[1]:.2f}) → ({b_[0]:.2f}, {b_[1]:.2f}) → ({c_[0]:.2f}, {c_[1]:.2f}) | "
                     f"({b_[0] - a[0]:+.2f}, {b_[1] - a[1]:+.2f}) | ({c_[0] - b_[0]:+.2f}, {c_[1] - b_[1]:+.2f}) |")
        contrib, rec2 = C.source_contributions(net, decs, tok, q, pos)
        reps = [(p_, REP[tok[0, p_, NM.F_REP].item()]) for p_ in range(pos[0].item()) if tok[0, p_, NM.F_REP] > 0]
        L += ["", "Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded "
              "displacement of P(right), P(bottom)):", ""]
        for l in range(net.nl):
            for h in range(net.nh):
                for p_, name in reps:
                    c = contrib[l, 0, h, p_]
                    if c.abs().max() > 0.02:
                        L.append(f"* L{l}.H{h} ← token {p_} ({name}): attention {rec2['pattern'][l][0, h, pos[0], p_]:.2f}, "
                                 f"({c[0]:+.2f}, {c[1]:+.2f})")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run = Path(a.run)
    ctx = M.Context("fixed", a.device)
    net = NM.Net(T.ENV["n"] ** 2, T.ENV["H"]).to(a.device).eval()
    net.load_state_dict(torch.load(sorted((run / "ckpt").glob("u*.pt"))[-1]))
    acts, _ = PR.collect(net, ctx.bank["broad"], ctx.d)
    _, decs = M.block_writes(net, ctx, acts, ctx.pairs["eval"])
    p = ctx.pairs["eval"]
    dd = ctx.dd
    sh, sv, order = dd["sh"][p["dec"]], dd["sv"][p["dec"]], dd["order"][p["dec"]]
    decisive = ~(p["opt"] & p["opt_cf"]).any(1)
    interior = ~p["legal"][:, NC.COMMIT] & (p["legal"][:, :4].sum(1) == 4)
    def pick(m):
        for extra in (decisive & interior, decisive, torch.ones_like(decisive)):
            hit = torch.nonzero(m & extra)
            if len(hit):
                return hit[0].item()
    L = ["# The navigate-commit experiment: worked traces", "", f"Model: `{run}`, final checkpoint. Decoders: one shared affine decoder per block, fitted on "
         "the block's input, middle and output on the probe-fitting layouts. Token indices: 0, 1 are the station records; "
         "even indices from 2 are decision tokens, odd ones the events after them.", ""]
    L += trace(net, decs, p, pick((sv == 0) & (sh > 0)), "1. One clue (horizontal)")
    L += trace(net, decs, p, pick(order == 3), "2. Two clues, horizontal first")
    L += trace(net, decs, p, pick(order == 4), "3. Two clues, vertical first")
    (run / "traces.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
