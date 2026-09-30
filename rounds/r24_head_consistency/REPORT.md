# Does the prediction head agree on identical-posterior histories? (round 24)

Run date: 2026-09-30. Brief: `BRIEF.md`. Measures and criterion: `PLAN.md`, committed (`4e6d078`) **before the head's
outputs on these pairs were computed**. No training: round 23's twenty models with a trained head, round 22's pairs.
Numbers from `tables.md`; data in `results.json`. Reproduce: `reproduce.py r24` (1 min on one GPU).

**Registered and post hoc.** The registered measure reads the head at the goal token and averages the four
candidate moves equally. After the reveal the head is trained only on the move the policy took, so I split that
average into the greedy move and the other three; the split is post hoc. The last prefix token was registered.

**What is matched.** Type A pairs: identical posterior (the same node of the belief graph), different tokens, the
same prefix length and token position; at the goal token also the same goal and 12 moves left; the same candidate
move. The exact predictive distribution is then the same for the two histories. Total variation over the three
symbols; median (range) over ten seeds, coefficient 1:

| where the head is read | D_same: between the two histories | error against exact | D_diff: type C pairs | relative inconsistency | as a share of the policy's |
|---|---|---|---|---|---|
| goal token, all four moves (registered) | 0.026 (0.022–0.040) | 0.043 | 0.452 (exact 0.456) | 0.058 | 0.82 (0.65–0.93) |
| goal token, the greedy move | 0.014 (0.011–0.019) | 0.020 | 0.346 (exact 0.339) | 0.042 | 0.54 (0.46–0.58) |
| goal token, the other three moves | 0.031 (0.025–0.047) | 0.051 | 0.488 (exact 0.495) | 0.062 | 0.87 |
| last prefix token, lengths 2–3 | 0.012 (0.011–0.014) | 0.015 | 0.466 (exact 0.469) | 0.025 | 0.33 (0.25–0.43) |
| policy, the same cells | 0.072 (0.057–0.091) | | 0.927 | 0.078 | 1 |

Relative inconsistency is D_same / D_diff: how much two histories with the same posterior differ, against how much
two histories with different posteriors do.

## Findings

* **By the registered criterion, the first branch: the head's predictions are themselves history-dependent.** At
  the goal token its relative inconsistency is 0.058, 0.82 of the policy's 0.078 (the criterion's threshold was a
  half). Two histories with the identical posterior get predictions 0.026 apart where the exact answer is one and
  the same.
* **It is smaller where the head is better trained, and does not vanish.** For the move the policy takes it is 0.014
  (0.54 of the policy's relative inconsistency, in every seed between 0.46 and 0.58). At prefix tokens, where moves
  are imposed at random and every candidate is trained, it is 0.012 (0.33). Untaken moves after the reveal are
  predicted worse (error 0.051) and less consistently (0.031).
* **The head is belief-consistent only to within its accuracy, and its accuracy is modest.** D_same is below the
  error against the exact distribution in all ten seeds at every site (about 0.6–0.8 of it): the head's errors are
  in larger part a function of the posterior and in smaller part of the history. An average excess cross-entropy of
  0.001 nats, which looked like an exact predictor in round 23, is an error of 0.02 in total variation on the taken
  move and 0.04 over all moves.
* **The head's inconsistency and the policy's occur together.** In the 7 % of type A cells where the greedy action
  differs between the two histories, D_same is 0.044 (0.030–0.065); where it is the same, 0.026 (0.021–0.037). I
  expected no difference. Whatever distinguishes the two histories reaches both outputs.
* **The head tracks real changes of posterior accurately**: for type C pairs its predictions differ by 0.452 where
  the exact predictive differs by 0.456.
* **Coefficient 0.1**: the head is less accurate (error 0.096 over all moves, 0.025 on the greedy move) and less
  consistent (D_same 0.046; relative inconsistency 0.096, 1.2 times the policy's). On the greedy move 0.019, 0.61 of
  the policy's.
* At the last prefix token of length-4 prefixes, where the head never had a target, D_same is 0.027.

## Reading, by the brief's two branches

1. **Neither branch cleanly; nearer the first.** The auxiliary task is solved with approximations that depend on
   the history at about the same relative size as the policy's (0.8 of it over all moves, a half on the taken move,
   a third at prefix tokens). The second branch — belief-consistent predictions with a policy that is not — is not
   what the data show: there is no site where the head is consistent and the policy is not.
2. **So round 23's result is less surprising.** The prediction objective did not reduce the policy's dependence on
   history because, as learned, it does not enforce a history-independent code: a loss within 0.001 nats of exact is
   reached by a predictor whose outputs still differ between same-posterior histories by 1–3 % in probability.
3. **Both outputs read a shared state that is a history summary close to, and not equal to, the posterior.** This
   agrees with rounds 21–22 (the prefix states are an evidence-dependent subspace; same-posterior histories differ
   inside it) and adds that the difference shows in a second, independently supervised read-out, largest in the
   same cells.

What this suggests for training: more weight on the same prediction loss is unlikely to help much, since the loss
is already near its floor. A loss that targets the inconsistency directly would be needed — a consistency term
between histories known to share a posterior, or more prediction targets per token (all four candidate moves, or
several steps ahead), which constrain more of the posterior than one sampled symbol does.

Limits: one maze; the comparison of the two outputs rests on a ratio chosen to put a 3-symbol and a 4-move
distribution on one scale; type A's posteriors are the 297 that recur with different histories; decisions and
predictions at the reveal only.
