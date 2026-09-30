# Round 21: is the ΣW intervention specific?

(Follows round 20; same task, same trained models, no new training.) The brief as sent:

My next step would be a narrow replication of the ΣW intervention, with specificity controls. Freeze its
construction using separate fitting data, then compare it with:

- **Top principal components of equal rank**: does ordinary high-variance history replacement work just as well?
- **Random subspaces of equal rank**: report intervention magnitude and captured donor difference, since rank alone
  is an inadequate match here.
- **Closely posterior-matched histories**: does the intervention still change actions when the posterior barely
  changes?
- **The same edit across goals**: does it move actions toward the donor's goal-specific optimal distribution, rather
  than merely producing different actions?

For uncertainty versus mode pairs, match the size of the optimal action-distribution change before interpreting the
reversal. Changing the most likely cell need not change the optimal action much; redistributing uncertainty can
change it substantially.

The key question now is: does the posterior-associated intervention explain behaviour more specifically than a
generic replacement of the dominant history representation?
