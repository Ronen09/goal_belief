# Round 22: three types of donor–recipient pair

(Follows round 21; same task, same trained models, no new training.) The brief as sent:

Construct three types of donor–recipient pairs, matching prefix length, remaining horizon, and any other
decision-relevant variables:

| pair type | what it tests |
|---|---|
| different histories, same posterior | does history matter beyond belief? |
| different posteriors, same optimal action for one goal | does the representation preserve information beyond that particular action? |
| different posteriors, different optimal actions | does replacing evidence produce the predicted decision change? |

Use whole-prefix patches and your existing rank-13 PCA patches, and evaluate each pair under all three goals. Keep
the natural-access and direct-route-removed conditions separate.
