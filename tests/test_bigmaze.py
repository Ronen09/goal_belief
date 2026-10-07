"""The larger maze: every cell a goal (the random-goals experiment) against fixed goals."""

import torch

from goalgeo import bigmaze as BM


def test_all_goals_env_spawn_prior_and_filter():
    m4, mall = BM.make(pairs=1, eps=0.4), BM.make(pairs=1, eps=0.4, all_goals=True)
    assert mall.n == m4.n and list(mall.landmarks) == list(m4.landmarks) and (mall.sym == m4.sym).all()
    assert len(mall.goals) == mall.n
    t = BM.Sim(mall, "cpu")
    gen = torch.Generator(); gen.manual_seed(0)
    env = BM.Env(t, 2000, gen)
    assert (env.cell != env.goal).all()
    assert env.goal.min() == 0 and env.goal.max() == t.n - 1
    n = torch.arange(env.N)
    assert (env.belief[n, env.goal] == 0).all()
    assert (env.belief[n, env.cell] > 0).all()
    assert torch.allclose(env.belief.sum(1), torch.ones(env.N))
    for _ in range(10):
        env.step(torch.randint(4, (env.N,), generator=gen))
        live = ~env.done
        assert (env.belief[n, env.cell][live] > 0).all()
        assert torch.allclose(env.belief[live].sum(1), torch.ones(int(live.sum())))
    # fixed episodes are reproducible, and a goal passed in is kept
    gen.manual_seed(0); e1 = BM.Env(t, 50, gen)
    gen.manual_seed(0); e2 = BM.Env(t, 50, gen)
    assert (e1.goal == e2.goal).all() and (e1.cell == e2.cell).all()
    e3 = BM.Env(t, 50, gen, goal=e1.goal, cell=e1.cell)
    assert (e3.goal == e1.goal).all() and (e3.cell == e1.cell).all()


def test_fixed_goals_unchanged():
    t = BM.Sim(BM.make(pairs=1, eps=0.4), "cpu")
    assert not t.all_goals and len(t.starts) == t.n - 4
    gen = torch.Generator(); gen.manual_seed(0)
    env = BM.Env(t, 500, gen)
    assert not any(c in t.maze.goals for c in env.cell.tolist())
    assert (env.belief[:, list(t.maze.goals)] == 0).all()
