import numpy as np
import torch

from goalgeo.envs import make_switch_env, make_corridor_env, make_env
from goalgeo.planning import solve_all_goals, optimal_dataset
from goalgeo.occupancy import goal_occupancy
from goalgeo.model import PolicyNet
from goalgeo.train import train_bc
from goalgeo import quotient as Qt


def test_ground_truth_sweep_locates_switch_boundary():
    lams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    gt = Qt.sweep_ground_truth(lams, gamma=0.9)
    e = gt["envs"][0]
    s = e.state_of(4, 0); g = e.goal_coords.index((4, 12))
    # margin is signed relative to the lambda=0 optimal action: positive before the switch, negative after
    m = gt["signed_margin"][:, g, s]
    assert m[0] > 0 and m[-1] < 0
    b = gt["boundaries"][g, s]           # list of lambda-interval indices where the argmax set changes
    assert len(b) == 1 and 0 < b[0] < len(lams) - 1
    assert gt["policy_tables"].shape == (len(lams), e.n_states, e.n_goals * 4)


def test_quotient_pairs_in_corridor():
    e = make_corridor_env(); sol = solve_all_goals(e, 0.9); occ = goal_occupancy(e, sol.pi, 0.9)
    pairs = Qt.quotient_pairs(e, sol, occ)
    A, B = pairs["case_A"], pairs["case_B"]
    assert len(A) > 0 and len(B) > 0
    tables = np.transpose(sol.pi > 0, (1, 0, 2)).reshape(e.n_states, -1)
    for i, j in A:
        assert np.array_equal(tables[i], tables[j])
    for i, j in B:
        assert not np.array_equal(tables[i], tables[j])
    # case A pairs are farther in occupancy than case B pairs
    Docc = np.linalg.norm(occ.Z_sa[:, None] - occ.Z_sa[None], axis=-1)
    assert np.median([Docc[i, j] for i, j in A]) > np.median([Docc[i, j] for i, j in B])


def _trained_base():
    e = make_env("base"); sol = solve_all_goals(e, 0.9); occ = goal_occupancy(e, sol.pi, 0.9)
    s, g, y = optimal_dataset(e, sol)
    net = PolicyNet(e, emb_dim=16, hidden=64, seed=0); train_bc(net, s, g, y, steps=800, lr=3e-3)
    return e, sol, occ, net, (s, g, y)


def test_interaction_ablation_identity_and_effect():
    e, sol, occ, net, data = _trained_base()
    res = Qt.interaction_ablation(net, sol, occ, layer="h2", n_random=2, seed=0)
    assert res["baseline"]["accuracy"] > 0.95
    assert 0.0 <= res["remove_component"]["accuracy"] <= 1.0
    assert set(res["remove_component"]) >= {"accuracy", "goal_sensitivity", "steer_success"}
    assert res["random_component"]["accuracy"] >= 0.0


def test_steering_predictors_linearised_alpha_is_exact_at_last_layer():
    e, sol, occ, net, data = _trained_base()
    tab = Qt.steering_predictors(net, sol, occ, layer="h3", n_goal_pairs=4, alphas=np.linspace(0, 2, 41), seed=0)
    fin = np.isfinite(tab["alpha_star"]) & np.isfinite(tab["alpha_lin"])
    assert fin.sum() > 10
    assert np.max(np.abs(tab["alpha_star"][fin] - tab["alpha_lin"][fin])) <= 0.05 + 1e-9
