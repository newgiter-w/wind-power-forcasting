# P1 result-to-claim judgment

Fresh GPT-6-Astra ultra review; same-family / provisional. Numerical checks pass; study scope remains WARN.

## C1

claim_supported: **yes**

**claim**: Forest remains fixed and comparisons reproduce P0 controls

**impact**: supported

**supported_scope**: 17 saved forest hashes and inputs unchanged; all raw/CQR endpoints exactly match P0; full bootstrap weights and native quantiles independently witnessed

**not_supported**: 17 new independent fits or new holdout effectiveness

**evidence**: ["route_a/p1_run.py:166-182", "route_a/p1_run.py:203-224", "route_a/reviewer_verify_p1.py:61-99", "route_a/reviewer_verify_p1.py:128-149"]

## C2

claim_supported: **no**

**claim**: A candidate meets all joint ramp/near-zero coverage, width and score goals

**impact**: unsupported

**supported_scope**: The report's negative conclusion is supported: zero eligible passing candidates on 2014 validation or either evaluation; retain CQR_signed84

**evidence**: ["route_a/p1_analyze.py:25-41", "experiments/route_a/20261107_p1_v030/joint_gates.csv:2-29", "experiments/route_a/20261107_p1_v030/selection.json:2-6"]

## C3

claim_supported: **partial**

**claim**: Descriptive gains/tradeoffs and mechanistic interpretation

**impact**: needs_qualifier

**supported_scope**: All narrow report numbers and tradeoffs are supported. 2014 conditional DualTail coverage gains cost width and scores; 2012 conditional PIT overall WIS/CRPS gains coexist with worse near-zero coverage/IS

**not_supported**: Causal physical-zero mechanism, isolated state/boundary/randomization mechanism, general superiority, new algorithm novelty, exact CPS/time-series coverage theorem, or deployment validity

**evidence**: ["experiments/route_a/20261107_p1_v030/REPORT.md:9", "experiments/route_a/20261107_p1_v030/REPORT.md:71", "experiments/route_a/20261107_p1_v030/REPORT.md:103", "experiments/route_a/20261107_p1_v030/REPORT.md:136", "artifacts/literature/jonkers2024.txt:847-875"]

## what_results_support

A reproducible fixed-forest retrospective comparison and an honest negative joint-goal result, with dataset/arm/metric-specific tradeoffs.

## what_results_dont_support

Joint objective achieved; independent fresh confirmation; robustness over PIT/forest seeds, unseen sites or dates; isolated causal mechanism; live availability or temporal coverage theorem.

## missing_evidence

[
  "Unused dates or sites under prospective limited selection",
  "PIT-seed and conditional-pool/boundary ablations",
  "Sensitivity to block length and explicit multiplicity policy for inferential claims",
  "Measured observation and NWP release latency plus missing-input service for deployment claims"
]

## next_experiments_needed

[
  "Prespecify a limited follow-up on unused dates or sites",
  "Matched conditional-pool and physical-boundary ablations plus PIT-seed sensitivity",
  "Block-length sensitivity when formal inference is central",
  "Latency-aware operational evaluation when deployment is claimed"
]

## routing_action

retain_CQR_and_record_qualified_negative_descriptive_result; plan prospective evidence before stronger claims
