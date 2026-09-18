import numpy as np
import torch

from goalgeo.envs import make_env, make_ring_env
from goalgeo.planning import solve_all_goals
from goalgeo.occupancy import goal_occupancy
from goalgeo import targets as T
from goalgeo import models2 as M
from goalgeo import supervision as Sv
from goalgeo import geometry as G


def _base():
    e = make_env("base"); sol = solve_all_goals(e, 0.9); occ = goal_occupancy(e, sol.pi, 0.9)
    return e, sol, occ


def test_boltzmann_targets_normalise_and_approach_hard_as_tau_to_zero():
    e, sol, occ = _base()
    hard = T.build_targets(e, sol, occ, "hard")
    cold = T.build_targets(e, sol, occ, "boltzmann", tau=1e-4)
    assert np.allclose(cold.Y.sum(1), 1.0)
    assert np.abs(cold.Y - hard.Y).max() < 1e-3
    assert hard.out_dim == 4 and hard.loss == "ce"


def test_regression_targets_have_expected_dims():
    e, sol, occ = _base()
    assert T.build_targets(e, sol, occ, "advantage").Y.shape[1] == 4
    assert T.build_targets(e, sol, occ, "q").Y.shape[1] == 4
    assert T.build_targets(e, sol, occ, "occupancy").Y.shape[1] == 4 * e.n_goals
    assert T.build_targets(e, sol, occ, "advantage").loss == "mse"


def test_kl_loss_has_same_gradient_as_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 4, requires_grad=True); p = torch.softmax(torch.randn(8, 4), -1)
    g_ce = torch.autograd.grad(T.LOSSES["ce"](logits, p), logits)[0]
    g_kl = torch.autograd.grad(T.LOSSES["kl"](logits, p), logits)[0]
    assert torch.allclose(g_ce, g_kl, atol=1e-6)


def test_generic_training_supports_regression_heads():
    e, sol, occ = _base()
    tg = T.build_targets(e, sol, occ, "q")
    net = M.make_model("mlp3", e, out_dim=tg.out_dim, hidden=32, emb_dim=8, seed=0)
    hist = T.train_targets(net, tg, steps=200, lr=3e-3, checkpoint_steps=[0, 200])
    assert hist.loss[-1] < hist.loss[0] and set(hist.checkpoints) == {0, 200}


def test_architectures_expose_activations():
    e, sol, occ = _base()
    s = torch.arange(e.n_states); g = torch.zeros(e.n_states, dtype=torch.long)
    for arch in ("mlp3", "mlp6", "resmlp", "transformer"):
        net = M.make_model(arch, e, out_dim=4, hidden=32, emb_dim=16, seed=0)
        acts = net.activations(s, g)
        assert "logits" in acts and acts["logits"].shape == (e.n_states, 4)
        assert len(net.hidden_layers) >= 2
        for l in net.hidden_layers:
            assert acts[l].shape[0] == e.n_states
        H = net.all_activations()
        assert H[net.hidden_layers[-1]].shape[:2] == (e.n_states, e.n_goals)


def test_synthetic_task_probe_rows_share_argmax_with_reference():
    task = M.SyntheticTask(n_background=50, margins=[0.01, 0.1, 0.5], seed=0)
    assert task.n_states == 50 + 3 + 1 and task.n_goals == 1
    ref = task.Q[task.reference]
    for m, row in zip(task.margins, task.probes):
        q = task.Q[row]
        assert q.argmax() == ref.argmax()
        assert np.isclose(np.sort(q)[-1] - np.sort(q)[-2], m)


def test_equivalence_classes_and_bottleneck_stats():
    e, sol, occ = _base()
    hard = T.build_targets(e, sol, occ, "hard"); soft = T.build_targets(e, sol, occ, "boltzmann", tau=0.05)
    n_hard = Sv.n_equivalence_classes(hard.Y); n_soft = Sv.n_equivalence_classes(soft.Y)
    assert 1 < n_hard < 16 and n_soft > n_hard
    stats = Sv.bottleneck_stats(hard, sol, occ)
    assert set(stats) >= {"rank", "participation_ratio", "n_classes", "rsa_with_Q", "recover_Q_r2", "recover_adv_r2", "recover_occ_r2"}


def test_unique_interaction_is_orthogonal_to_additive_span():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(9, 5, 7))
    dec = Sv.additive_decomposition(H)
    a, b, r = dec["a_s"], dec["b_g"], dec["r"]
    assert np.allclose(a.sum(0), 0) and np.allclose(b.sum(0), 0)
    assert np.allclose(r.sum(0), 0) and np.allclose(r.sum(1), 0)
    ru = dec["r_unique"]
    span = np.vstack([a, b])                     # additive feature directions
    assert np.abs(ru.reshape(-1, 7) @ span.T).max() < 1e-8
    assert 0 <= dec["unique_fraction"] <= 1


def test_within_between_class_ratio():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(20, 3)), rng.normal(size=(20, 3)) + 10])
    classes = np.array([0] * 20 + [1] * 20)
    r = Sv.within_between(X, classes)
    assert r["ratio"] < 0.3 and r["between"] > r["within"]


def test_ring_env_has_all_three_pair_types():
    e = make_ring_env(); sol = solve_all_goals(e, 0.9); occ = goal_occupancy(e, sol.pi, 0.9)
    hard = T.build_targets(e, sol, occ, "hard")
    pairs = Sv.abc_pairs(hard, tau=0.02)
    st = pairs["stats"]
    assert st["A"]["n"] > 0 and st["B"]["n"] > 0 and st["C"]["n"] > 0
    assert st["A"]["dQ"] > st["C"]["dQ"]
    assert st["C"]["dS"] / st["C"]["dQ"] > st["A"]["dS"] / st["A"]["dQ"]   # C: soft targets differ more per unit of Q difference
    assert st["B"]["dQ"] < st["A"]["dQ"]
