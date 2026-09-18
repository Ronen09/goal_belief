import numpy as np
import torch

from goalgeo.gridworld import GridWorld
from goalgeo.planning import solve_all_goals, optimal_dataset
from goalgeo.model import PolicyNet, LAYERS
from goalgeo.train import train_bc, evaluate_accuracy

MAP = """
..#.
.G..
#..G
....
"""


def _env():
    return GridWorld.from_ascii(MAP)


def test_activations_have_expected_shapes():
    env = _env()
    net = PolicyNet(env, encoding="onehot", emb_dim=8, hidden=16, n_hidden=3)
    s = torch.arange(env.n_states)
    g = torch.zeros(env.n_states, dtype=torch.long)
    acts = net.activations(s, g)
    assert list(acts.keys()) == LAYERS
    assert acts["emb"].shape == (env.n_states, 16)
    assert acts["h1"].shape == (env.n_states, 16)
    assert acts["h3"].shape == (env.n_states, 16)
    assert acts["logits"].shape == (env.n_states, 4)


def test_forward_from_layer_matches_forward_when_unpatched():
    env = _env()
    net = PolicyNet(env, encoding="coord", emb_dim=8, hidden=16, n_hidden=3)
    s = torch.arange(env.n_states); g = torch.ones(env.n_states, dtype=torch.long)
    acts = net.activations(s, g)
    for layer in ["emb", "h1", "h2", "h3"]:
        out = net.forward_from(layer, acts[layer])
        assert torch.allclose(out, acts["logits"], atol=1e-6), layer


def test_all_goal_activation_tensor_shape():
    env = _env()
    net = PolicyNet(env, emb_dim=8, hidden=16)
    H = net.all_activations()  # dict layer -> [S, K, d]
    assert H["h2"].shape == (env.n_states, env.n_goals, 16)


def test_training_reaches_high_accuracy():
    env = _env()
    sol = solve_all_goals(env, gamma=0.9)
    s, g, y = optimal_dataset(env, sol)
    net = PolicyNet(env, emb_dim=8, hidden=32, seed=0)
    hist = train_bc(net, s, g, y, steps=600, lr=1e-2, checkpoint_steps=[10, 600])
    assert evaluate_accuracy(net, s, g, y) > 0.99
    assert set(hist.checkpoints) == {10, 600}
    assert hist.loss[-1] < hist.loss[0]
