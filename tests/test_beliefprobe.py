"""Probe identities: exact recovery from the target itself, invariance to affine maps of h,
and the masked stacked trainer producing ordinary CausalTransformers."""

import numpy as np
import torch

from goalgeo import belief_train as BT, beliefprobe as BP, latentgoal as LG

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_probe_recovers_log_odds_from_themselves_and_counts_in_iid():
    E = BP.EvalSet("iid", 4, n=300, seed=3)
    m = BP.measure_site(E.y, E)
    for s in ("IID", "EXT", "CONF", "TIME"):
        assert m[f"{s}_r2y"] > 1 - 1e-9
    assert BP.measure_site(E.counts, E)["IID_r2y"] > 1 - 1e-9      # iid: y is affine in counts


def test_probe_is_invariant_to_invertible_affine_maps():
    E = BP.EvalSet("channel", 4, n=300, seed=4)
    rng = np.random.default_rng(0)
    H = np.c_[E.counts, np.tanh(E.y), rng.normal(size=(len(E.y), 5))]
    A = rng.normal(size=(H.shape[1], H.shape[1])); c = rng.normal(size=H.shape[1])
    m1, m2 = BP.measure_site(H, E), BP.measure_site(H @ A + c, E)
    for k in m1:
        assert abs(m1[k] - m2[k]) < 1e-6, k


def test_masked_stack_unstacks_to_the_same_function():
    jobs = [BT.Job("tfm", "iid", 3, "goal", 0), BT.Job("tfm", "iid", 3, "next_obs", 1)]
    nets, _ = BT.train_tfm_stack(jobs, steps=3, n_pool=64, batch=8, device=DEV)
    X, _, _ = LG.sample(LG.make_env("iid", 3), 5, seed=9)
    for net, j in zip(nets, jobs):
        p = BP.tfm_sites(net, X, device=DEV)["p"]
        assert p.shape[1] == LG.out_dim(LG.make_env("iid", 3), j.objective)
        assert np.allclose(p.sum(1), 1, atol=1e-6)
        ref = BT.load_net(j)
        ref.load_state_dict(net.state_dict()); ref.out_mask = net.out_mask
        q = BP.tfm_sites(ref, X, device=DEV)["p"]
        assert np.allclose(p, q, atol=1e-6)


def test_window_transformer_edit_identity_and_receptive_field():
    from goalgeo import wtfm as W
    X, _, _ = LG.sample(LG.make_env("iid", 3), 6, seed=2)
    Xt = torch.as_tensor(X)
    for carry in (False, True):
        net = W.WindowTransformer(6, 3, 2, carry, seed=0).eval()
        with torch.no_grad():
            p = torch.softmax(net.forward_all(Xt), -1).double().numpy()
        assert np.allclose(net.run_edited(Xt, 5), p[:, 5:], atol=1e-6)
    net = W.WindowTransformer(6, 3, 2, False, seed=0).eval()        # no carry: sees 3 tokens back at most
    X2 = X.copy(); X2[:, 1:10] = 1
    with torch.no_grad():
        a = net.forward_all(Xt)[:, 14]; b = net.forward_all(torch.as_tensor(X2))[:, 14]
    assert torch.allclose(a, b)


def test_forward_query_reproduces_the_full_forward_pass():
    from goalgeo import kvprior as KP
    env = LG.make_env("channel", 4)
    X, _ = KP.lm_data(env, 8, 24, 0)
    for L in (2, 4):
        net = KP.make_net(KP.LMSpec(L, 24, 0)).to(DEV).eval()
        R = KP.residuals(net, X); t = 9
        res, u, p, _ = KP.forward_query(net, [r[:, :t + 1] for r in R[:L]], X[:, t + 1], t + 1)
        assert np.abs(res[-1] - R[L][:, t + 1]).max() < 1e-4
        assert np.abs(p - KP.predictive(net, X)[:, t + 1]).max() < 1e-5
