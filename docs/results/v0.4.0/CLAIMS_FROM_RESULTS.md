# P2 result-to-claim judgment

Independent reviewer verdict: **WARN**; deterministic integrity checks: **PASS**. Same-family review, acceptance provisional.

No numerical/data/training integrity failure found. All independent scoring, gate and inference checks pass. Historical fixed-configuration scope limits causal/generalization claims. C1 yes, C2 no, C3 partial; retain QRF_CQR.

No required repairs remain. Preserve the historical fixed-configuration scope and retain QRF_CQR.

## C1 — yes

**what results support**

Same canonical QRF information, final dense values, complete proper-training label set, proper-training-only selection/preprocessing, fresh full refit and identical signed84 CQR. Actual neural raw outputs are finite, bounded and noncrossing.

**what results dont support**

Noncrossing calibrated heads, identical compute, extra historical encoding, independent replay of every optimizer update, or deployment availability assumptions.

**missing evidence**

Optimizer/intermediate states were not saved; current hashes alone do not independently authenticate historical freeze timing. Neither is needed to verify the narrower artifact-consistency claim.

**suggested claim revision**

Retain the explicit retrospective same-information/label-budget/raw-noncrossing wording; distinguish raw from independently calibrated cross-level nesting.

**next experiments needed**

None for this bounded claim.

**evidence**

[
  "route_a/p2_train.py:55",
  "route_a/p2_train.py:89",
  "route_a/p2_train.py:97",
  "route_a/p2_models.py:34",
  "route_a/p2_evaluate.py:122",
  "experiments/route_a/20261107_p2_v040/review_trace/independent_numeric_verification.json:134"
]

## C2 — no

**what results support**

No tested candidate meets the joint target: 0/12 seed-mean gates and 0/36 fixed-seed diagnostic gates pass. The 2014 validation rule retains QRF_CQR. TCN calibrated overall IS is worse than the MLP control on both datasets; GRU's overall calibrated IS contrasts with MLP have intervals crossing zero.

**what results dont support**

A small temporal candidate achieving simultaneous up-ramp/near-zero coverage, width and proper-score targets, or equivalence of GRU and MLP.

**missing evidence**

No passing candidate exists in this completed protocol. A stronger future effectiveness claim requires a new predeclared candidate and unused confirmation data.

**suggested claim revision**

None of these fixed small neural configurations met the joint target against frozen QRF_CQR; retain QRF_CQR and report the observed tradeoffs.

**next experiments needed**

None to document this bounded negative result. Any new candidate/search budget should be fixed before accessing fresh confirmation dates/sites.

**evidence**

[
  "experiments/route_a/20261107_p2_v040/joint_gates.csv:2",
  "experiments/route_a/20261107_p2_v040/joint_gates.csv:6",
  "experiments/route_a/20261107_p2_v040/joint_gates.csv:13",
  "experiments/route_a/20261107_p2_v040/selection.json:3",
  "experiments/route_a/20261107_p2_v040/REPORT.md:118",
  "experiments/route_a/20261107_p2_v040/REPORT.md:121",
  "experiments/route_a/20261107_p2_v040/REPORT.md:130",
  "experiments/route_a/20261107_p2_v040/REPORT.md:133"
]

## C3 — partial

**what results support**

Historical method/metric/regime tradeoffs, seed means/std, conditional paired calendar-block intervals, actual training-loss decreases, zero raw crossings and measured CQR nesting violations.

**what results dont support**

Pure temporal-structure causality, fully sufficient optimization, universal network-family superiority/inferiority, unseen-farm/date generalization, or boundary/normalization mechanisms without interventions.

**missing evidence**

Controlled parameter/normalization/sequence interventions, bounded prospective optimizer sensitivity, fresh locked dates/sites, optional forest repeats for retraining uncertainty, and real availability/latency validation for deployment.

**suggested claim revision**

Keep all statements descriptive and conditional on these fitted models, dates, farm weighting and fixed configurations; label explanations as future hypotheses.

**next experiments needed**

If pursuing mechanisms, predeclare capacity/LayerNorm-matched per-lead controls and lead-order/temporal-path interventions. If pursuing generalization, evaluate a frozen protocol on unused dates/sites.

**evidence**

[
  "experiments/route_a/20261107_p2_v040/REPORT.md:10",
  "experiments/route_a/20261107_p2_v040/REPORT.md:81",
  "experiments/route_a/20261107_p2_v040/REPORT.md:108",
  "experiments/route_a/20261107_p2_v040/REPORT.md:141",
  "experiments/route_a/20261107_p2_v040/REPORT.md:143"
]
