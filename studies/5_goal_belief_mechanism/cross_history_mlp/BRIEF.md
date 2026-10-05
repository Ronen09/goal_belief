# Cross-history MLP: what do the goal token's MLPs carry — a goal instruction, an evidence–goal combination, or an action preference?

(Follows the goal-swap-components experiment; the observation-prediction experiment's reward models, frozen. No training.) The brief as sent:

The next question is what those MLPs carry: a goal instruction, an evidence–goal combination, or an action preference.
Generalizing the selected component locations to new histories does not yet answer that—the donor activation is still
generated separately for each history.

I would now do the cross-history test:
- Take an MLP output from history A under G2, where the model correctly chooses left.
- Patch it into history B under G1, where the model correctly chooses up, but would choose right under G2.
- Ask whether the patched model chooses right, left, or retains up.
