import numpy as np
import torch

from goalgeo.envs import make_env
from goalgeo.planning import solve_all_goals, optimal_dataset
from goalgeo.occupancy import goal_occupancy
from goalgeo.model import PolicyNet
from goalgeo.train import train_bc
from goalgeo import analysis as A


def _setup():
    env = make_env("base")
    sol = solve_all_goals(env, 0.9)
    occ = goal_occupancy(env, sol.pi, 0.9)
    return env, sol, occ


def test_hypothesis_rdms_have_state_shape_and_expected_keys():
    env, sol, occ = _setup()
    hyps = A.hypothesis_rdms(env, occ, gamma=0.9)
    assert {"occupancy", "occupancy_state", "spatial", "geodesic", "SR", "identity"} <= set(hyps)
    for R in hyps.values():
        assert R.shape == (env.n_states, env.n_states)


def test_steer_alpha_zero_matches_unpatched_actions():
    env, sol, occ = _setup()
    net = PolicyNet(env, emb_dim=8, hidden=16, seed=1)
    res = A.steer(net, sol, layer="h2", g1=0, g2=3, alphas=[0.0])
    base_actions = net.all_activations()["logits"][:, 0, :].argmax(-1)
    assert np.array_equal(res["actions"][0], base_actions)


def test_steer_on_trained_net_moves_behaviour_toward_g2():
    env, sol, occ = _setup()
    s, g, y = optimal_dataset(env, sol)
    net = PolicyNet(env, emb_dim=16, hidden=64, seed=0)
    train_bc(net, s, g, y, steps=800, lr=3e-3)
    # at the embedding layer the mean difference vector is exactly [0, E_g2 - E_g1],
    # so alpha=1 reproduces the g2 input and behaviour must match pi*_g2 wherever
    # the network is accurate; alpha=0 must match pi*_g1.
    res = A.steer(net, sol, layer="emb", g1=0, g2=3, alphas=[0.0, 1.0])
    assert res["n_differ"] > 5
    assert res["agree_g1"][0] > 0.95 and res["agree_g2_excl"][0] == 0.0
    assert res["agree_g2"][1] > 0.95 and res["agree_g2_excl"][1] > 0.3


def test_disagreement_pairs_are_extremes_of_rdm_difference():
    rng = np.random.default_rng(0)
    R1, R2 = np.abs(rng.normal(size=(12, 12))), np.abs(rng.normal(size=(12, 12)))
    R1, R2 = (R1 + R1.T), (R2 + R2.T)
    idx_occ_far, idx_sp_far = A.disagreement_pairs(R1, R2, quantile=0.25)
    n_pairs = 12 * 11 // 2
    assert 0 < len(idx_occ_far) <= n_pairs // 4 + 1
    assert set(idx_occ_far).isdisjoint(set(idx_sp_far))


def test_hypothesis_rdms_include_policy_table():
    env, sol, occ = _setup()
    hyps = A.hypothesis_rdms(env, occ, gamma=0.9, sol=sol)
    assert "policy" in hyps and hyps["policy"].shape == (env.n_states, env.n_states)


def test_steer_reports_occupancy_predicted_alpha():
    env, sol, occ = _setup()
    net = PolicyNet(env, emb_dim=8, hidden=16, seed=1)
    rep = A.steering_report(net, sol, occ, layers=("h2",), n_goal_pairs=3, alphas=np.linspace(0, 2, 5))
    pred = np.array(rep["layers"]["h2"]["alpha_pred"])
    assert len(pred) == len(rep["layers"]["h2"]["flip_alpha"]) > 0
    assert np.all((pred >= 0) & (pred <= 1))
