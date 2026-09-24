"""The read gate: exact 0/1 forward, sigmoid straight-through backward, penalty over positions >= 2."""
import torch

from goalgeo import readgate as RG


def test_eval_gate_is_the_sign_of_the_logit():
    a = torch.tensor([-2.0, -1e-3, 0.0, 1e-3, 5.0])
    g, p = RG.gate(a, train=False)
    assert g.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert torch.allclose(p, torch.sigmoid(a))


def test_train_gate_is_binary_with_sigmoid_gradient_and_open_probability_sigmoid():
    torch.manual_seed(0)
    a = torch.full((20000,), 1.0, requires_grad=True)
    g, p = RG.gate(a, train=True, gen=torch.Generator().manual_seed(1))
    assert set(g.detach().unique().tolist()) <= {0.0, 1.0}
    assert abs(g.detach().mean().item() - torch.sigmoid(torch.tensor(1.0)).item()) < 0.02   # P(open) = sigma(a)
    g.sum().backward()
    assert a.grad.abs().sum() > 0                                                         # straight-through path exists
    assert torch.allclose(p, torch.sigmoid(a.detach()))


def test_mix_is_exact_at_the_endpoints():
    o1, o0 = torch.randn(3, 5), torch.randn(3, 5)
    assert torch.equal(RG.mix(torch.ones(3), o1, o0), o1)
    assert torch.equal(RG.mix(torch.zeros(3), o1, o0), o0)


def test_penalty_positions():
    p = torch.zeros(2, 6, 3); p[:, :2] = 1.0; p[:, 2:] = 0.25          # [B, T, L]: positions 0, 1 fully open
    assert abs(RG.penalty(p).item() - 0.25) < 1e-7                      # they do not count
    assert abs(RG.penalty(p, first=0).item() - (2 * 1.0 + 4 * 0.25) / 6) < 1e-7
