from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, torch, importlib.util
sys.argv = ["x"]
SP = __HERE__ + ""
src = open(SP + "finetune.py").read().split("res = {}")[0]
exec(src)
base, _ = M26.load("reward", 0, t, dev, M26.R23 / "runs")
nets = {"before": base, "control": finetune(base, "control", 0)}
for name, net in nets.items():
    gen = torch.Generator(device=dev); gen.manual_seed(99)
    b = P.rollout(net, t, 16384, gen, greedy=True)
    reg = (b.regret * b.alive).sum(0) / b.alive.sum(0).clamp(min=1)
    disc = (t.gamma ** torch.arange(t.H, device=dev)) * (b.regret * b.alive).mean(0)
    s = P.summarize(b, t, 1)[0]
    # share of decisions whose state was 'covered': graph nodes seen in the fine-tune data
    seen = torch.zeros(len(t.belief), dtype=torch.bool, device=dev); seen[d["nodes"].flatten()] = True
    cov = seen[b.node[b.alive]].float().mean()
    print(name, "regret", round(s["regret"], 4), "success", round(s.get("success", float('nan')), 3),
          "| mean local regret by step", [round(x, 3) for x in reg.tolist()],
          "| discounted contribution by step", [round(x, 4) for x in disc.tolist()], f"| nodes covered {cov:.2f}")
    # first decision: optimal rate, and by goal
    a0 = b.act[:, 0]; ok0 = (t.Q[b.node[:, 0]].gather(1, a0[:, None]).squeeze(1) >= t.V[b.node[:, 0]] - 1e-6)
    print("   first-decision optimal", round(ok0.float().mean().item(), 3), "by goal", [round(ok0[b.goal == g].float().mean().item(), 3) for g in range(3)],
          "| regret by goal", [round(P.summarize(b, t, 1)[0].get(f"G{g+1}", float('nan')), 4) if f"G{g+1}" in s else None for g in range(3)])
