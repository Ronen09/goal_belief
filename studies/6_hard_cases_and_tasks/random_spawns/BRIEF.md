# Random spawns: is the additive policy a consequence of where the agent starts?

(Follows the additive-code and additive-ablation experiments. New reward-trained models.)

The brief as sent: "I also just realized that the additive policy might just be because of how the goals are placed,
can we try randomizing agent spawns" and, when I started to combine it with removing the passive prefix: "Dont combine
them lets do spawns first".

The concern: the maze-belief task starts the agent in one of four corridor cells, two on each arm, and has three fixed
goals. A goal bias that is the same for every history ("G3: go right") may work only because the agent always starts in
the same few places relative to the goals. With spawns anywhere, the right move toward a goal depends on where the
agent is, and a fixed bias per goal might no longer do.
