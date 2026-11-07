# Route A findings

2026-11-07, P0 v0.2.0: frozen baseline reproduction completed. Fresh same-family review: baseline reproduction=yes (17 replays, two fixed seed-0 refits); diagnostic attribution=partial; integrity=WARN; acceptance=provisional. Numerical checks passed. Training sufficiency and causal attribution remain untested; no new-method effectiveness claim.

Supported retrospective observations: periodic NWP service gaps, conditional tail failures despite macro coverage, nonnested independent-CQR confidence levels, and a dataset-dependent coverage/width/IS tradeoff. Preserve all failure groups. Next hypothesis is coherent full-distribution calibration with frozen controls; no P1 experiment has run.

Evidence: [report](../docs/results/v0.2.0/REPORT.md), [claim judgment](../docs/results/v0.2.0/CLAIMS_FROM_RESULTS.md), [audit](../docs/results/v0.2.0/EXPERIMENT_AUDIT.md).

## P1 v0.3.0

2026-11-07: completed seven fixed-forest calibration arms across 17 farms, zero new research forests. Hash continuity and P0 controls pass. Fresh same-family review: C1=yes, C2_joint_goal=no, C3_descriptive_and_mechanistic_scope=partial; integrity=WARN, acceptance=provisional. No candidate passes the validation or evaluation joint gate; retain CQR_signed84.

2014 state dual-tail improves target coverages with width/score cost; 2012 state PIT improves aggregate scores but loses near-zero coverage. Full CDF nesting is preserved; independent per-level interval calibration may remain nonnested. Do not relax the predeclared 1% width allowance or call historical single-PIT-seed results fresh confirmation. Further interventions need bounded validation-only designs and unused confirmation data; no P2 is run.

Evidence: [P1 report](../docs/results/v0.3.0/REPORT.md), [joint gates](../docs/results/v0.3.0/joint_gates.csv), [audit](../docs/results/v0.3.0/EXPERIMENT_AUDIT.md).
