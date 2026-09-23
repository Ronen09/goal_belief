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
