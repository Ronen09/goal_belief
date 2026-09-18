"""The stacked trainer must be the per-model one: identical forward pass, identical attention
weights, and an identical optimisation trajectory given the same data order."""

import numpy as np
import torch

from goalgeo import cuts as Ct, hmm4 as H4, tfm as Tf, tfm_batched as Tb

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KW = dict(d=16, n_heads=2, n_layers=2, mlp=32, T=16)


def _specs():
    return [Tb.Spec(seed=0, norm="learn"), Tb.Spec(seed=1, norm="none"), Tb.Spec(seed=2, norm="frozen"),
            Tb.Spec(seed=3, norm="learn", attn_diag=(0,)), Tb.Spec(seed=4, norm="learn", attn_diag=(1,))]


def test_batched_forward_matches_each_model():
    specs = _specs()
    bt = Tb.BatchedTransformer(specs, device=DEV, **KW)
    X = torch.randint(0, H4.V, (len(specs), 6, KW["T"]), device=DEV)
    with torch.no_grad():
        zb, attn = bt.logits(X, need_weights=True)
        for j, net in enumerate(bt.to_nets()):
            net = net.to(DEV).eval()
            res, aw = net.residuals(X[j], need_weights=True)
            z = net.out(net.ln_f(res[-1]))
            assert torch.allclose(z, zb[j], atol=2e-5), (j, (z - zb[j]).abs().max().item())
            for i in (0, 1):
                assert torch.allclose(aw[i], attn[i][j], atol=2e-5)


def test_batched_forward_with_fixed_gain_readout():
    specs = [Tb.Spec(seed=s, gain=g, norm="frozen") for s, g in ((0, 0.5), (1, 2.0))]
    bt = Tb.BatchedTransformer(specs, device=DEV, **KW)
    X = torch.randint(0, H4.V, (2, 4, KW["T"]), device=DEV)
    with torch.no_grad():
        zb = bt.logits(X)
        for j, net in enumerate(bt.to_nets()):
            assert torch.allclose(net.to(DEV).eval().forward_all(X[j]), zb[j], atol=2e-5)
            assert abs(float(net.out.weight.norm(dim=1).mean()) - specs[j].gain) < 1e-5


def test_batched_training_matches_per_model_training():
    """Same init, same data order, same optimiser: the stacked run must track the single run."""
    steps = 25
    specs = [Tb.Spec(seed=0, norm="learn"), Tb.Spec(seed=1, norm="none", gain=None)]
    bt = Tb.BatchedTransformer(specs, device=DEV, **KW)
    Xp, Yp = Tb.make_data(specs, pool=200, T=KW["T"], device=DEV)
    hist_b = Tb.train_batched(bt, Xp, Yp, steps=steps, batch=16, lr=1e-3, tf32=False)
    nets_b = bt.to_nets()
    m = H4.make_hmm4(1.0, 0.4, 1)
    for j, s in enumerate(specs):
        net = Tf.CausalTransformer(n_vocab=H4.V, seed=s.seed, final_norm=s.norm, **KW).to(DEV)
        hist = Ct.train_routed(net, m, steps=steps, batch=16, seed=s.seed, T=KW["T"], pool=200, lr=1e-3)
        assert abs(hist[-1] - hist_b[-1, j]) < 2e-3, (j, hist[-1], hist_b[-1, j])
        for (k, a), b in zip(net.state_dict().items(), nets_b[j].state_dict().values()):
            assert torch.allclose(a.cpu(), b.cpu(), atol=2e-3), k


def test_tf32_training_stays_close_to_fp32_training():
    """Production runs use TF32 for speed; it must move the trajectory only at the 1e-3 level."""
    specs = [Tb.Spec(seed=0, norm="learn")]
    Xp, Yp = Tb.make_data(specs, pool=200, T=KW["T"], device=DEV)
    out = []
    for tf32 in (False, True):
        bt = Tb.BatchedTransformer(specs, device=DEV, **KW)
        out.append(Tb.train_batched(bt, Xp, Yp, steps=25, batch=16, lr=1e-3, tf32=tf32)[-1, 0])
    assert abs(out[0] - out[1]) < 5e-3, out


def test_attention_penalty_reduces_the_penalised_route():
    specs = [Tb.Spec(seed=0, mu_l1=0.0), Tb.Spec(seed=0, mu_l1=5.0)]
    bt = Tb.BatchedTransformer(specs, device=DEV, **KW)
    Xp, Yp = Tb.make_data(specs, pool=200, T=KW["T"], device=DEV)
    Tb.train_batched(bt, Xp, Yp, steps=60, batch=16, lr=3e-3)
    free, pen = bt.to_nets()
    X = Xp[0, :32].cpu()
    off = []
    for net in (free, pen):
        _, attn = net.cpu().eval().residuals(X, need_weights=True)
        eye = torch.eye(KW["T"], dtype=torch.bool)
        off.append(float(attn[0].masked_fill(eye, 0.0).sum(-1).mean()))
    assert off[1] < off[0]
