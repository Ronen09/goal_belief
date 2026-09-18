import numpy as np
import torch

from goalgeo import hmm4 as H4, seqmodels as Sm


def test_weighted_loss_zero_weight_removes_relevant_gradient():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    X, Z = H4.sample(m, 8, 12, seed=0)
    Y = H4.next_token(m, H4.beliefs_seq(m, X))[:, :-1]
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y, dtype=torch.float32)
    rel = torch.as_tensor(m.relevant[Z[:, 1:]])
    # weight 1 everywhere except relevant positions -> loss must not depend on the relevant targets
    W = torch.where(rel, 0.0, 1.0)
    l1 = Sm.weighted_seq_loss(net, Xt, Yt, W)
    Y2 = Yt.clone(); Y2[rel] = torch.roll(Y2[rel], 1, dims=-1)
    l2 = Sm.weighted_seq_loss(net, Xt, Y2, W)
    assert torch.isclose(l1, l2)
    W1 = torch.ones_like(W)
    assert not torch.isclose(Sm.weighted_seq_loss(net, Xt, Yt, W1), Sm.weighted_seq_loss(net, Xt, Y2, W1))


def test_train_weighted_learns_and_returns_checkpoints():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    hist, ckpt = Sm.train_weighted(net, m, lam=2.0, steps=80, batch=32, seed=0, T=16, pool=200, checkpoints=(0, 40, 80))
    assert len(hist) == 80 and np.mean(hist[-10:]) < np.mean(hist[:10])
    assert sorted(ckpt) == [0, 40, 80]
    assert not torch.equal(ckpt[0]["out.weight"], ckpt[80]["out.weight"])
    assert torch.equal(ckpt[80]["out.weight"], net.state_dict()["out.weight"].cpu())


from goalgeo import prominence as Pr  # noqa: E402


def test_linear_cv_accuracy_on_separable_and_random_labels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 8)); y = X[:, 0] > 0
    assert Pr.linear_cv_accuracy(X, y) > 0.9
    assert abs(Pr.linear_cv_accuracy(X, rng.random(400) > 0.5) - 0.5) < 0.15


def test_positions_and_anchor_geometry():
    m = H4.make_hmm4(1.0, 0.4, 3)
    ev = Pr.make_eval(m, n=50, T=30, seed=0)
    P, anchors = Pr.positions(m, ev["Z"])
    assert P["cueA"].sum() > 0 and P["preB"].sum() > 0 and P["ctrlU"].sum() > 0
    assert len(anchors) > 0
    for i, t in anchors:
        assert ev["Z"][i, t] in (m.S["P"], m.S["Q"])
        assert m.relevant[ev["Z"][i, t + m.k + 1]] and t + m.k <= 30 - 2


def test_measure_returns_all_keys_and_sane_values():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    ev = Pr.make_eval(m, n=120, T=24, seed=0)
    res = Pr.measure(net, m, ev, n_rsa=100, n_dec=300)
    for key in ("P_metric", "P_metric_pre", "belief_r2", "branch_acc_pre", "rsa_belief", "rsa_P1", "rsa_P2", "rsa_future",
                "kl", "kl_relevant", "G_rel", "Gpar_rel", "G_imm", "Gpar_imm", "var_ratio_delayed_vs_random"):
        assert key in res and np.isfinite(res[key]), key
    assert res["P_metric"] > 0 and res["kl"] >= 0
    assert res["Gpar_rel"] <= res["G_rel"] + 1e-9 and res["Gpar_imm"] <= res["G_imm"] + 1e-9


def test_cross_distance_of_identical_groups_is_zero():
    h = np.ones((10, 4))
    assert Pr.cross_distance(h, h) == 0.0


def test_gradients_match_finite_differences_and_are_nonzero():
    m = H4.make_hmm4(1.0, 0.4, 2)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=12, emb=6, out_dim=H4.V, seed=1)
    ev = Pr.make_eval(m, n=40, T=20, seed=3)
    P, anchors = Pr.positions(m, ev["Z"]); anchors = anchors[:1]
    dhat = np.zeros(12); dhat[0] = 1.0
    g = Pr.gradients(net, ev["X"], ev["Y"], anchors, m.k, dhat)
    assert g["G_rel"] > 1e-4 and g["G_imm"] > 1e-4
    # finite difference of the relevant loss along dhat
    i, t = anchors[0]
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(ev["X"])).numpy()
    def rel_loss(h):
        win = torch.as_tensor(ev["X"][i, t + 1:t + m.k + 1])[None]
        hk, _ = net.gru(net.emb(win), torch.as_tensor(h, dtype=torch.float32)[None, None])
        y = torch.as_tensor(ev["Y"][i, t + m.k], dtype=torch.float32)
        return float((-(y * torch.log_softmax(net.out(hk[0, -1]), -1)).sum()).detach())
    eps = 1e-3; h = Hs[i, t].astype(np.float64)
    fd = (rel_loss(h + eps * dhat) - rel_loss(h - eps * dhat)) / (2 * eps)
    assert np.isclose(abs(fd), g["Gpar_rel"], rtol=1e-2, atol=1e-4), (fd, g)


def test_fixed_gain_readout_keeps_row_norms_after_training():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0, gain=2.0)
    Sm.train_weighted(net, m, steps=30, batch=32, seed=0, T=16, pool=100, checkpoints=(0, 30))
    W = net.out.weight.detach()
    assert W.shape == (H4.V, 16) and torch.allclose(W.norm(dim=1), torch.full((H4.V,), 2.0), atol=1e-5)
    free = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    assert isinstance(free.out, torch.nn.Linear)


def test_measure_reports_readout_gain_and_achieved_gap_identity():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0, gain=1.5)
    ev = Pr.make_eval(m, n=120, T=24, seed=0)
    res = Pr.measure(net, m, ev, n_rsa=100, n_dec=300)
    assert np.isclose(res["g_eff"], float((net.out.weight[H4.TOK["x"]] - net.out.weight[H4.TOK["y"]]).detach().norm()), atol=1e-5)
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(ev["X"])).numpy()
    P, _ = Pr.positions(m, ev["Z"])
    d = Hs[P["preA"]].mean(0) - Hs[P["preB"]].mean(0)
    wd = (net.out.weight[H4.TOK["x"]] - net.out.weight[H4.TOK["y"]]).detach().numpy()
    assert np.isclose(res["achieved_gap_pre"], float(wd @ d), atol=1e-5)
    assert -1 <= res["cos_readout_pre"] <= 1 and np.isfinite(res["achieved_gap_control"])
    assert np.isclose(res["required_gap"], 2 * np.log(0.9 / 0.1))


def test_readout_init_scale_and_separate_readout_lr():
    m = H4.make_hmm4(1.0, 0.4, 1)
    a = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    b = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0, out_scale=10.0)
    assert torch.allclose(b.out.weight, 10 * a.out.weight) and torch.allclose(b.gru.weight_hh_l0, a.gru.weight_hh_l0)
    w0 = a.out.weight.detach().clone(); g0 = a.gru.weight_hh_l0.detach().clone()
    Sm.train_weighted(a, m, steps=5, batch=16, seed=0, T=12, pool=50, checkpoints=(0,), lr=1e-3, lr_out=0.0)
    assert torch.equal(a.out.weight, w0) and not torch.equal(a.gru.weight_hh_l0, g0)


def test_control_contrast_parameter():
    m = H4.make_hmm4(1.0, 0.4, 1, ctrl=0.6)
    assert np.isclose(m.E[m.S["A'"], H4.TOK["x"]], 0.6) and np.isclose(m.E[m.S["B'"], H4.TOK["y"]], 0.6)
    assert np.isclose(H4.make_hmm4(1.0, 0.4, 1).E[H4.make_hmm4(1.0, 0.4, 1).S["A'"], H4.TOK["x"]], 0.9)
