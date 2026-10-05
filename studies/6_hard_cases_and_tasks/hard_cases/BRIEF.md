# Hard cases: the additively impossible cases, training on them, and harder tasks

(Follows the additive-code experiment. Given in conversation, one question at a time; nothing here was pre-registered.)

1. What are the hallmarks of the cases where no additive code H(history) + G(goal) can be right?
2. Can we generate more of them? (Chosen: decisions after the goal reveal, same models, no retraining.)
3. Use them for training: first by fine-tuning the observation-prediction experiment's models, then ("why fine-tune when we can train again")
   by training from scratch with them.
4. Is there any difference between the policies?
5. A visual simulation of how the model does it compared with the Bayes-optimal version.
6. A harder problem that forces a better policy: screen task variants; then multiple goals with random values shown
   at the reveal, collected to maximise reward (chosen: fixed positions, values shown).
