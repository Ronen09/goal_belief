# Round 12: make the goal hidden

Right now, the goal is supplied to the agent. That means there isn't really a recursively updated "goal belief"; there is just a representation conditioned on a known goal. That makes the phrase goal geometry slightly misleading.

Instead construct an environment where there is a latent goal \(G\), the agent gets observations informative about \(G\), and must act while maintaining

$$ b_t(g)=P(G=g\mid o_{\le t},a_{<t}). $$

Now there is a mathematically privileged internal object to look for.

I would make the first experiment extremely synthetic:

3–5 possible goals.
At the start of an episode, sample one goal.
Give noisy evidence about the goal over several timesteps.
Make the optimal action later depend on the posterior over goals.
Calculate the exact Bayesian posterior ourselves.

Then test three increasingly strong claims.

1. Recoverability. Can the posterior \(b_t\), or preferably its \(K-1\) log-odds coordinates, be recovered by an affine map from the residual state?

Don't start with RSA. Your own results have basically shown why that's dangerous: metric geometry can vary while functionally meaningful quantities stay fixed.

Test something like

$$ Ah_t+c \approx \left[ \log \frac{b_t(g_1)}{b_t(g_K)},\ldots, \log\frac{b_t(g_{K-1})}{b_t(g_K)} \right]. $$

Generalisation matters more than train \(R^2\): hold out posterior values/evidence sequences.

(Claims 2 and 3 were not included in the brief as given.)
