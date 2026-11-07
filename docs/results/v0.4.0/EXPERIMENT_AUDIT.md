# P2 fresh experiment-integrity and result-to-claim review

Date: 2026-11-07. Reviewer: gpt-6-astra, ultra. Review independence: same-family. Acceptance status: provisional. Project: `D:/Study/wind_probability_route_a`; reviewed run: `experiments/route_a/20261107_p2_v040`.

**Overall verdict: WARN. Integrity status: warn. Deterministic artifact/numeric checks: PASS.** No fabricated ground truth, prediction-based score normalization, missing claimed experiment, training-label leakage, seed selection, or numerical mismatch was found. The WARN concerns limited historical scope and causal/generalization interpretation; it is not a finding that the reported measurements are corrupted. The one concrete report-unit defect found during review was corrected and rechecked before this verdict.

**Claim verdicts: C1 yes; C2 no; C3 partial.** Confidence is high for these bounded judgments. Preserve the frozen QRF_CQR choice. Neither a new temporal-model success claim nor a universal failure claim follows.

All file:line references below are relative to the project root. Exact reviewed file SHA256 values, including the final report and report generator, are in `001-review.verdict.json` and `reviewer_snapshot.json` in this directory. This review judged primary code and artifacts directly. The executor supplied paths, scope and requested checks, not an interpretation to accept. A second fresh same-family subreview checked training contracts and checkpoint reality; its scope and limits are retained in `training_subreview/training_subreview.md`.

## Independently executed evidence

The reviewer ran only these guarded, project-local scripts; no installation or wind-model retraining occurred:

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/reviewer_verify_p2.py
& .venv/Scripts/python.exe tools/run_route_a.py --script experiments/route_a/20261107_p2_v040/review_trace/training_subreview/verify_training.py
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/reviewer_snapshot_p2.py
```

- Raw CSV provenance: all 29 raw-source hashes match P0. Direct timestamp indexing reproduces all 344,664 target cells and all corresponding history cells over 17 farms, including missing values and target-valid masks. This is not comparison against another model's predictions.
- Scoring: all 170 model/seed prediction sets (17 frozen QRF plus 153 neural), both validation/evaluation stages, all declared regime/power/lead/month/season masks were independently scored. The 83,100 per-farm metric rows, 2,720 proper-score rows, 2,640 boundary rows, 9,840 metric seed summaries, 3,936 metric macro rows, 320 score seed summaries and 128 score macro rows agree. Largest score-table discrepancy is 4.55e-13 in averaged count fields; metric-valued discrepancies are at ordinary floating-point roundoff.
- Calibration: an independent pool-and-rank implementation verifies 188,352 lead-specific finite-rank corrections over both farm_01 datasets, all four model families seed0, both stages and all three confidence levels. Corrections, endpoints, calibration counts and latest label-end timestamps match exactly. It does not call production `causal_cqr`.
- Inference: independently reconstructed daily sufficient statistics and a histogram-weight implementation reproduce all 420 paired contrasts and 95% intervals within 4.44e-16. Sampling uses 184/363 full calendar days for 2014/2012, circular 14-day blocks, identical draws across farms/methods/fixed seeds, zero counts on absent/unscored dates and farm-macro/seed-mean aggregation.
- Gates: all 12 seed-mean and all 36 per-seed gate rows agree within 2.45e-15. No row passes. 2014 validation therefore selects QRF_CQR.
- Training: all 153 fit identities, 10,057 learning-curve rows, selected epochs, min_delta/patience decisions and full-refit lengths agree. All 153 final training losses decrease; chosen epochs are 8–59 and no selection run reaches 160 epochs. Four training source hashes, 68 input hashes, two evaluation source hashes, the analysis source hash and all 17 forest-continuity hashes match.
- Representative reality checks: independently refitted selection/final preprocessors for both farm_01 datasets match exactly. Six neural seed0 checkpoints reproduce every saved forecast and their initial/final training losses exactly. All parameter tensors differ from initialization. Representative QRF checkpoint predictions and P0/P2 controls also agree exactly. This does not claim that 153 optimization trajectories were rerun.

Primary evidence: `route_a/reviewer_verify_p2.py:17`, `:69`, `:145`, `:159`, `:184`; `experiments/route_a/20261107_p2_v040/review_trace/independent_numeric_verification.json:5`, `:134`, `:284`, `:315`, `:328`; `experiments/route_a/20261107_p2_v040/review_trace/training_subreview/training_subreview.json:1842`, `:2052`, `:2062`; `route_a/reviewer_snapshot_p2.py:13`.

## A. Ground-truth provenance — PASS

GEFCom2014 labels come from Task15 training CSV TARGETVAR and the supplied Task15 solution CSV, joined by zone and timestamp (`gefcom2014_experiment.py:38`, `:42`, `:44`, `:47`). GEFCom2012 labels come from `windpowermeasurements.csv` and farm-specific wp fields (`gefcom_protocol_v2.py:111`, `:112`, `:131`). Daily targets index the next 24 observed hours; observed history indexes the last 24 hours (`gefcom_protocol_v2.py:77`, `:79`). Input eligibility depends on history/NWP quality; target-validity remains a later fitting/scoring mask (`gefcom_protocol_v2.py:82`, `:91`, `:97`).

P0 reconstructs raw data and asserts agreement with cached labels/history and canonical/QRF features (`route_a/p0_baseline.py:204`, `:212`, `:221`, `:227`). P2 reads those labels separately from heads (`route_a/p2_evaluate.py:103`, `:111`). The independent reviewer additionally read the raw CSVs directly and reproduced all 17 farms, rather than trusting these assertions (`route_a/reviewer_verify_p2.py:58`, `:74`). Real target regimes are used only for retrospective diagnostics, not model input or forecast selection (`route_a/p0_baseline.py:101`; `experiments/route_a/20261107_p2_v040/REPORT.md:68`).

The numerical labels are the dataset's supplied normalized power values, not functions of model outputs. The source information is history/calendar/NWP: `gefcom_protocol_v2.py:61`, `strong_baselines.py:53`, `run_p2_2012.py:20`, `run_gefcom_tail_dynamic.py:56`. The target returned alongside a feature matrix is not concatenated into that matrix.

No claim of reproducing official competition scores is made. These are custom, transparently defined retrospective metrics against real dataset ground truth. The inherited zero-reporting-delay and inferred 2014 NWP issue-time assumptions remain deployment limitations (`experiments/route_a/20261107_p2_v040/REPORT.md:143`).

## B. Score normalization — PASS

Coverage, width, interval score and tail misses use fixed mathematical definitions (`route_a/p0_baseline.py:95`). WIS combines the median error and three central interval scores with fixed alpha/2 weights and denominator 3.5; mean pinball uses the seven fixed quantiles (`route_a/p2_evaluate.py:18`, `:21`, `:27`). The independent calculations reproduce WIS = 2 × pinball7 for every scored cell. There is no prediction-max/min/mean normalization, and no fabricated exact full-distribution CRPS is reported (`experiments/route_a/20261107_p2_v040/REPORT.md:55`).

Gate ratios divide an actual method's score by the corresponding frozen QRF_CQR score; raw scores are retained beside those comparative ratios (`route_a/p2_analyze.py:25`, `:34`; `experiments/route_a/20261107_p2_v040/REPORT.md:44`, `:59`). Farm equal weighting is implemented before averaging neural seeds; seed standard deviations use the three fixed fits, while QRF's one-seed standard deviation remains missing (`route_a/p2_evaluate.py:52`; `experiments/route_a/20261107_p2_v040/REPORT.md:29`).

## C. Result existence, provenance and number fidelity — PASS

There are exactly 153 neural fit/checkpoint/head identities over 17 farms × three architectures × three seeds, plus 17 frozen QRF controls. Training, evaluation and analysis completion files all say complete; current source/input/forest hashes agree with recorded manifests. Real gradients, changed weights, lower training losses and exact saved-checkpoint replay substantiate genuine learned predictions. The fresh environment witness executes forward/backward/optimizer fixtures and the isolation test; its follow-up records matching 21 package pins and resolved documentation paths (`experiments/route_a/20261107_p2_v040/environment_witness.md:96`, `:107`, `:117`). Import success was not treated as a readiness test.

The report's gate and tradeoff values match prediction-derived scores. In particular 2014 TCN_CQR has width ratio 0.9302046228728997, IS ratio 1.0920114441165174, up-ramp coverage change -0.051230066768462756 and near-zero coverage change +0.068952115524606 (`experiments/route_a/20261107_p2_v040/joint_gates.csv:6`). 2012 GRU_CQR has corresponding values 0.961129288109721, 1.0201169303813487, -0.04406005016381842 and +0.015271897307787774 (`joint_gates.csv:13` in the same run). The report rounds these correctly at `REPORT.md:10`. All six evaluation neural candidates worsen near-zero mean IS (`joint_gates.csv:5`, `:6`, `:7`, `:11`, `:12`, `:13`).

The initial report called adjacent-pair crossings “hours”. Production sums violated quantile pairs, and one hour can contribute multiple pairs (`route_a/p2_evaluate.py:38`). This was a unit-label defect, not a metric error. The final `REPORT.md:85` labels adjacent-pair counts and nesting-affected hours separately; `REPORT.md:104` explicitly explains multiplicity. The corresponding generator at `route_a/p2_report.py:79` and `:87` is also corrected. All numerical artifacts stayed unchanged, and final report/generator hashes are recorded.

The deterministic evidence precheck only verifies token existence (`experiments/route_a/20261107_p2_v040/evidence_precheck.json:74`). This review goes beyond it by recomputing the underlying values. Current hashes demonstrate consistency, not an independently authenticated historical timestamp of protocol fixation. Optimizer states and intermediate selection weights are not saved (`route_a/p2_train.py:68`); the review does not claim replay of each optimizer update.

## D. Dead code and execution path — PASS

The metric function is called inside collection for every confidence level and again for the 90% diagnostic/daily arrays (`route_a/p2_evaluate.py:21`, `:30`). Collection is called after actual forecast/calibration loading (`:133`); the metric/score/seed/macro/boundary outputs are written at `:136`–`:143`, and the bootstrap is executed at `:144`. `make_gates` is called for both aggregate and seed tables (`route_a/p2_analyze.py:110`, `:111`). The independent recomputation reaches every reported metric key and verifies that its data are present. No claimed metric was found to be an uncalled placeholder.

Real optimizer calls are present (`route_a/p2_train.py:17`, `:21`, `:22`, `:23`) and documented learning curves/checkpoints are mutually consistent. Synthetic unit fixtures are explicitly separate from empirical forecast results (`route_a/check_p2.py:1`, `:32`).

## E. Scope and identifiability — WARN, explicitly qualified

The actual evidence is 10 GEFCom2014 and seven GEFCom2012 farms, a fixed MLP/TCN/GRU configuration each, three neural seeds, one frozen QRF seed, and previously inspected dates. The report declares this at `REPORT.md:3`, `:29`, `:81`, `:108`, `:141`. No fresh holdout, deployed availability, all-model-family superiority/inferiority, fully sufficient optimization or simultaneous-trajectory-coverage claim is supported. The conditional time-block intervals do not include population-of-farms uncertainty, forest retraining variability, architecture search uncertainty or multiple-comparison correction.

The training-label budget and raw information are matched. The 80/20 split and selection preprocessing are internal to proper training (`route_a/p2_train.py:89`, `:97`, `:102`, `:105`). Each final network starts from a fresh same-seed initialization and refits all proper-training labels for the selected epoch count (`:53`, `:55`, `:60`). Calibration/validation/evaluation labels do not enter this network fitting/stopping route. Representative train budgets are 8,760 labels/365 days in 2014 and 13,152 labels/548 days in 2012.

This is not identical compute or identical architecture. Parameter counts differ: MLP/TCN/GRU are 15,719/17,031/16,375 for 2014 and 15,815/17,063/16,407 for 2012 (`REPORT.md:74`). LayerNorm, depth, convolution, recurrence and optimization behavior differ (`route_a/p2_models.py:40`, `:48`, `:67`), and chosen refit epochs also differ. These differences are correctly qualified (`REPORT.md:23`, `:141`; `docs/P2_METHODS.md:47`). TCN/GRU encode forecast-lead tokens with repeated history/full NWP curve, not an extra historical encoder (`REPORT.md:20`, `:22`). They receive no longer power history or future observed power. MLP is a useful control for these implementations but cannot by itself causally isolate temporal structure.

## F. Evaluation type — PASS: real_gt

All reported wind-performance tables use real dataset observations, with provenance and direct array equality established in A. The tiny generated fixtures in `check_p2.py` are separately labeled synthetic unit tests, are not a reference series for forecast scoring, and are not pooled into the empirical results. This review is same-family/provisional; deterministic checks do not turn it into cross-family semantic acceptance.

## Result-to-claim judgment

### C1 — yes; confidence high

Supported claim: “In this retrospective P2 implementation, the fixed neural models use the same canonical QRF information and final dense feature values, the same complete proper-training label set for final fitting, training-only preprocessing/epoch selection and the same signed84 online CQR protocol. Actual saved neural raw outputs are bounded and noncrossing.”

Support: the common noncrossing head uses a free base plus positive cumulative increments and [0,1] clipping (`route_a/p2_models.py:24`, `:34`); all 170 raw model/seed prediction sets are noncrossing. Independent-level CQR can and does violate cross-level ordering/nesting, which is reported rather than concealed. Representative exact preprocessors/checkpoints, all histories/manifests and independently replayed calibration ranks support the bounded claim. It does not mean final calibrated quantiles are noncrossing or that optimizer-update trajectories were independently reproduced.

### C2 — no; confidence high

The intended positive claim that a tested small temporal candidate meets the joint up-ramp/near-zero/width/score target is unsupported. None of the 2014 validation candidates passes, none of the evaluation dataset/candidate means passes, and no fixed-seed diagnostic passes. The declared fallback therefore remains QRF_CQR (`route_a/p2_analyze.py:43`; `experiments/route_a/20261107_p2_v040/selection.json:3`). Narrower intervals and improved near-zero coverage coexist with worse up-ramp coverage and worse overall/near-zero proper scores for the temporal candidates.

The MLP comparison does not rescue the success claim. TCN_CQR has worse overall IS than MLP_CQR on both datasets: +0.027659 [0.018822, 0.036295] in 2014 and +0.017330 [0.007343, 0.027531] in 2012 (`REPORT.md:118`, `:130`). GRU_CQR overall IS differences from MLP_CQR are small and their time-block intervals cross zero: +0.002015 [-0.010569, 0.013330] and +0.004950 [-0.003017, 0.013381] (`REPORT.md:121`, `:133`). This is not evidence of equivalence. 2012 GRU_CQR is worse than MLP_CQR on up-ramp and near-zero IS with these conditional intervals above zero (`REPORT.md:134`, `:135`).

Suggested replacement claim: “None of these fixed small neural configurations met the joint target against frozen QRF_CQR. Retain QRF_CQR; report the observed width/coverage/proper-score tradeoffs and the MLP comparison.” No further experiment is required to support this bounded negative result.

### C3 — partial; confidence high for descriptive scope

Supported: the listed historical method/metric/regime tradeoffs, their seed means/standard deviations, the conditional paired time-block comparisons, zero raw crossings and observed CQR nesting violations. The fixed configurations learned measurable training signal. In this run, TCN's worse calibrated IS relative to MLP is consistent across both datasets; GRU does not show a clear overall calibrated IS advantage over MLP.

Unsupported: “temporal structure causes the score change”; “training was sufficient because loss decreased/no epoch cap was hit”; “all TCN/GRU models are intrinsically inferior/superior”; “the mechanisms generalize to new farms/dates/weather products”; “near-zero coverage gain is improved conditional interval quality”; or an explanation of error changes by saturation/normalization without a controlled intervention. Boundary diagnostics describe clipping/zero support; they do not identify the cause of the tradeoff.

Missing evidence for stronger claims: fresh locked dates/sites, controlled parameter/normalization/input-sequence interventions, a bounded and prospectively fixed optimization-sensitivity study, repeated forest fits if claiming algorithm-training variability, and operational timestamp/latency validation. These are requirements for stronger future claims, not reasons to overwrite or selectively rerun this completed result.

## Actions and next experiments

1. Completed during review: repair the crossing-count unit label in report and generator; keep existing values. Verified final wording and snapshot hashes.
2. Retain QRF_CQR and carry the reported descriptive qualifiers into any downstream summary. Do not promote near-zero coverage or narrower width into the failed joint success claim.
3. If investigating why these configurations fail, predeclare a small capacity/LayerNorm-matched per-lead control and a lead-order/temporal-path intervention, with matched inputs and proper-training-only selection. Keep any additional learning-rate/epoch study bounded, report every configuration and avoid choosing it from these evaluation results.
4. Any effectiveness/generalization confirmation needs unused dates/sites with the protocol and search budget locked before access. Any deployment claim also needs real availability/latency evidence. Include QRF seed repetitions only if broadening from this frozen-control estimand to algorithm-retraining uncertainty.

No new training, publication or external write was performed. No unresolved numerical or implementation defect requires a repair before documenting the bounded result. Overall `claim_supported=partial` reflects supported C1/descriptive C3 and unsupported positive C2; it must not be rounded up to positive method acceptance.
