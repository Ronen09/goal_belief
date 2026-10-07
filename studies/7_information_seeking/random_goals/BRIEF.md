# Random goals: a goal drawn from every cell

(Follows the interior-goals experiment. New models.)

The maze10 report's proposed next step was the same maze "with goals in the interior, or a goal drawn from all cells,
where no fixed direction points to a goal". The interior-goals experiment took the first half (six junction goals; the
additive policy stayed at 0.75). This is the second: every cell a possible goal, one per episode.

The brief as sent: "run the additive ceiling script yes sure and also i want us to try random goal spawns". The
additive ceiling script (`../additive_ceiling/run.py`, no network) was run first; its row for the larger maze with
every cell a goal is the design check in `PLAN.md`.
