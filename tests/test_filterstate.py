"""The full-state coordinates are an invertible reparametrisation of the joint filter."""

import numpy as np

from goalgeo import filterstate as FS, latentgoal as LG


def test_coords_invert_the_joint_filter():
    env = LG.make_env("channel", 4)
    X, _, _ = LG.sample(env, 100, seed=6)
    J = LG.filter_joint(env, X)[:, 1:].reshape(-1, 4, 2)
    J = J[(J > 1e-5).all((1, 2))]                      # away from the logit clip
    y, l = FS.coords(J)
    assert np.allclose(FS.joint_from(y, l), J, atol=1e-9)
