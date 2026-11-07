# Route A P1 fresh experiment audit and result-to-claim review

Date: 2026-11-07. Project: `D:/Study/wind_probability_route_a`. Run: `experiments/route_a/20261107_p1_v030`.

Reviewer: **gpt-6-astra / ultra**, fresh reviewer with direct access to primary artifacts. Review independence: **same-family**. Acceptance status: **provisional**. The inference subreview is also same-family/provisional; delegation does not upgrade independence.

**Overall verdict: WARN. Integrity status: warn. Claim supported: partial.** No substantive numerical, label-provenance, normalization, or future-target-gating defect was found. The warning concerns the limited scientific scope, not falsified results. The negative joint-goal conclusion and the narrowly stated descriptive tradeoffs are supported. The report does not establish a successful joint improvement, fresh confirmation, a physical mechanism, or operational coverage guarantees.

All file:line references below are relative to the project root unless prefixed by `RUN/`, which denotes this P1 run. I read the requested code, protocol, documentation, literature extract, manifests, completion/check records, results, and farm-01 full-law artifacts directly. I also examined the corresponding artifacts for all farms where needed. The executor did not provide an interpretation of the results to adopt.

## Independent verification actually performed

The reviewer script is `route_a/reviewer_verify_p1.py`; output is `RUN/review_trace/independent_distribution_verification.json`. It imports no P1 distribution/calibration/score helpers. It independently reconstructs bootstrap response weights, empirical inverse quantiles, the distributional transform, state pools, finite ranks, and CRPS using the pairwise expectation identity.

- All **102 input hashes**, **5 execution-source hashes**, the analyzer hash, saved/source protocol equality, and **17** model hashes continuing from P0 matched.
- **1,826,496** saved raw/CQR prediction values across 17 farms and both stages matched the P0 controls exactly. All **5,436** forecast-day calibration pools matched independently computed arrival/window membership and saved counts/end timestamps.
- Farm 1 in each dataset: **41,616 observed target cells** and **41,616 history cells** were reconstructed directly from source CSV timestamps and matched exactly. These are observations, not model outputs.
- For those two farms, all **19,704** nontraining forecast cells had their near-zero probabilities and randomized PIT recomputed. Maximum PIT difference was **3.34e-16**.
- For **24 real forecast cells**, all **500 trees** were traversed and their full bootstrap responses reconstructed directly from each tree's seed. The resulting weights matched saved sparse weights within **1.39e-17**. Native package linear quantiles and raw ECDF quantiles matched exactly; pairwise raw-law CRPS matched within **3.89e-16**.
- **287** calibration witnesses, spanning both stages, horizons, predicted states, fallback and nonfallback pools, reproduced PIT pushforward quantiles, both dual-tail variants' ranks/collapse/clipping, conditional weights, and boundary probabilities. Quantile/end-point errors were zero; largest pairwise CRPS error was **3.36e-15**. Full-CDF nesting was checked across all farms/stages.
- The inference subreview independently rebuilt **29,085 interval rows**, **952 score rows**, all macro aggregates, **102 boundary-score rows**, **102 state-support rows**, **204 paired-bootstrap rows**, and **28 joint-gate rows**. It used hourly arrays and a day-multiplicity matrix for bootstrap sums, rather than trusting saved daily sufficient statistics. Maximum discrepancies were at floating-point precision. It checked **72 report table rows** and the headline numbers. See `RUN/review_trace/p1_inference_review.md`, `p1_inference_verify.json`, and `p1_report_verify.json`.

This is a meaningful independent subset of full-law construction, together with comprehensive saved-score/inference reconstruction. It is not a replay of all full-law predictions or a retraining study.

## A. Ground-truth provenance — PASS

GEFCom2014 observations come from released `TARGETVAR` data and the Task15 solution (`gefcom2014_experiment.py:38`–`:49`); GEFCom2012 uses `windpowermeasurements.csv` (`gefcom_protocol_v2.py:110`–`:112`, `:131`). `assemble` indexes observed history and future observed targets directly, preserving target validity separately from history/NWP eligibility (`gefcom_protocol_v2.py:74`–`:97`). P0 saves those observations (`route_a/p0_baseline.py:255`); P1 loads them (`route_a/p1_run.py:166`) and scores against `d['y']` (`:84`–`:110`). The reviewer verified the source-CSV chain numerically for both farm-01 datasets (`route_a/reviewer_verify_p1.py:101`–`:118`).

Historical predictions are used only as numerical reproduction controls: native raw-quantile witness (`route_a/p1_run.py:173`–`:177`), P0 CQR endpoints (`:203`–`:208`), and the old independent-tail implementation (`:211`–`:214`). Agreement with those outputs is not counted as forecast accuracy.

No current/future target selects a forecast, state, or pool. Forecast dates follow P0 input eligibility; complete-day target validity is a scoring/arrived-calibration mask. `arrived` restricts to seed/prior issues with `target_end <= origin` and the past 84 calendar days (`route_a/p1_distribution.py:60`–`:66`). States use raw forecast median changes, predicted near-zero mass, and the last observed history (`route_a/p1_run.py:23`–`:25`). Truth passed to `distribution_stats` affects CRPS only, not quantiles (`route_a/p1_distribution.py:20`–`:27`). Precomputed future PIT values exist offline but are never selected into an earlier pool. Logical arrival is still conditional on the disclosed zero reporting/release-delay convention; it is not measured live availability.

## B. Score definitions and normalization — PASS

Coverage, width and interval score use their direct definitions (`route_a/p0_baseline.py:95`–`:98`). WIS uses the method-specific median and three alpha/2-weighted interval scores divided by the fixed **3.5** weight convention; the common-median sensitivity and pointwise identity with twice mean pinball are explicit (`route_a/p1_run.py:87`–`:101`). No score is divided by the model's own maximum, mean, or variance. Macro metrics are means over farm-level means, not fabricated normalized scores (`:230`–`:237`).

CRPS is exact for the represented finite discrete law (`route_a/p1_distribution.py:20`–`:27`), independently checked by `E|X-y| - 0.5 E|X-X'|`. The raw ECDF and two PIT arms alone receive CRPS (`route_a/p1_run.py:110`). Brier scores use the real event `y <= .01` (`route_a/p1_analyze.py:72`–`:79`). Probability-weight normalization is legitimate normalization of a predictive law, not score inflation. Capacities are already represented on the dataset's unit scale.

## C. Result existence and claim fidelity — PASS

Both run and analysis completion files say complete (`RUN/completion.json:2`, `RUN/analysis_completion.json:2`). Required result tables and hourly artifacts exist. The string-based `evidence_precheck.json` verifies eight values' existence only; it is not treated as semantic or numerical proof. Independent recomputation established the reported numbers and gates.

The report correctly says no candidate passes the development or evaluation joint gate (`RUN/REPORT.md:7`–`:11`). Exact source rows for the highlighted comparisons are `RUN/joint_gates.csv:15` and `:28`. Selection is CQR with no eligible passing candidates and no evaluation dependency (`RUN/selection.json:2`–`:6`; `route_a/p1_analyze.py:37`–`:41`).

A minor broken method-document link was reported during review and corrected by the executor. I reread the final generator at `route_a/p1_report.py:36`, `:88` and final report at `RUN/REPORT.md:19`–`:32`, `:143`. The link now resolves to `docs/P1_METHODS.md`, and raw/reference arms are labeled as controls instead of candidate failures. No numerical claim changed. Final hashes:

- `route_a/p1_report.py`: `aa8a721c23a63de28d52f0db81fdb2b878990c8220a639ceb5147695f4fba1da`
- `RUN/REPORT.md`: `d2431abeefdcd902c8858b301566845a1510ac69a1b438001a2366f6cdc93689`

## D. Dead code and execution path — PASS

All claimed metric/distribution routines have an active path into saved results: forest weights (`route_a/p1_run.py:178`–`:182`), PIT/raw-law statistics (`:185`–`:198`), stage calibration (`:209`), scoring (`:216`), tables and paired blocks (`:230`–`:243`), and analysis gates/boundary checks (`route_a/p1_analyze.py:108`). Legacy routines outside the P1 call path are not presented as executed P1 methods. The report generator uses saved tables and explicitly constrains its negative narrative to the selected result (`route_a/p1_report.py:14`–`:20`).

## E. Scope and inference — WARN

The actual scope is two datasets, 17 farms, seven arms, one existing seed-0 forest per farm, one fixed randomized-PIT realization per farm, and retrospective dates. No research forest is fitted by P1. This cannot establish multi-seed robustness, new-site/date generalization, arbitrary-drift coverage, or practical real-time service. The report explicitly acknowledges these limits (`RUN/REPORT.md:3`, `:136`, `:144`–`:146`). There is no inflated comprehensive/robust claim requiring deletion.

The bootstrap is correctly synchronous over complete calendars: 184 dates in 2014 and 363 in 2012. Missing-input dates remain zero-count calendar positions; all farms/methods/groups share one draw matrix per dataset (`route_a/p1_run.py:125`–`:141`). Each draw divides within farm then averages farms; all macro groups in this run have all 10 or 7 farms. The 14-day circular, 1,000-draw percentile intervals are conditional descriptive summaries, without multiplicity correction or validation under drift. Their numerical correctness does not turn them into unconditional confidence guarantees. The recorded protocol and selection code establish the rule used, not externally timestamped prospective preregistration after these dates had already been inspected.

## F. Evaluation type — PASS: real_gt

Empirical forecast metrics are **real_gt**. Historical-output comparisons are explicitly numerical reproduction checks, not proxy ground truth. `route_a/check_p1.py` is **simulation_only / synthetic_unit_tests_only**, isolated from empirical effectiveness claims (`:1`, `:14`–`:21`, `:38`–`:41`). No synthetic unit score is substituted into wind evaluation tables.

## Distribution and calibration method findings

The original serialized forests and P0 control arrays remain fixed. Full response reconstruction uses every bootstrap occurrence in each selected leaf, inverse leaf size, and equal tree weights (`route_a/p1_distribution.py:31`–`:57`); it does not approximate each leaf by a small sample. ECDF inverse quantiles and the package's weighted linear interpolation are distinct representations and have separate raw controls.

The randomized transform uses the strict-left and point masses at the observed response (`route_a/p1_run.py:185`–`:195`). Pushing the calibration-PIT empirical law through the current inverse ECDF produces a normalized nonnegative finite predictive law; exact PIT 0/1 map to physical 0/1 (`route_a/p1_distribution.py:11`–`:17`; `route_a/p1_run.py:49`–`:56`). Its quantiles therefore nest. Boundary mass is part of this algorithm, not evidence that physical zero generation has been identified. This extension was not independently ablated.

Both tails use signed scores and `ceil((n+1)(1-alpha/2))`; crossed intervals collapse to the original interval midpoint before physical clipping (`route_a/p1_run.py:57`–`:76`). The state variant uses the same predicted-state/six-hour-band pools as PIT, at least 20 distinct days and 40 cells, and `min(.8, days/(days+20))` shrinkage (`route_a/p1_distribution.py:69`–`:83`). Its mixed empirical-rank rule is a heuristic, not an inherited conditional conformal theorem. Pool cells are temporally dependent. The report correctly publishes interval-only nonnesting and scarce-state fallback support (`RUN/REPORT.md:75`–`:103`).

The Jonkers source supports the documented scope deviation: Algorithm 2 uses OOB calibration (`artifacts/literature/jonkers2024.txt:815`–`:834`); Algorithm 3 fits a random forest to training residuals and populates leaves with calibration scores (`:859`–`:875`). SCDRF's coverage proof is itself deferred (`:847`–`:853`). P1 is neither of these algorithms and correctly claims only an empirical-CDF recalibration adaptation (`docs/P1_METHODS.md:5`–`:7`, `:18`–`:28`; `RUN/REPORT.md:143`). No novelty or exact CPS theorem follows from this implementation.

## Agent-follows-document witness

After reading `docs/COMPUTE_ENV.md`, I ran its P1 unit command verbatim from the project root:

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p1.py
```

Exit code **0**. It returned true for exact discrete CRPS, weighted inverse-rank ties, frozen bootstrap leaf weights, current/future-label invariance of all seven output arms, and full-CDF nesting. The three-tree/four-observation synthetic fit is explicitly disclosed and authorized (`docs/COMPUTE_ENV.md:34`–`:45`). **No doc-versus-reality divergence found.** This is a small supplied-input unit witness, not a full empirical invariance proof; actual arrival membership and label/state separation were independently checked above. All new reviewer scripts ran only through the project interpreter and guard. No research training, installation, publication, or writes to the original project/shared data occurred.

## Result-to-claim judgment

- **C1 — Fixed forest and P0 controls: yes.** Hash continuity, unchanged inputs, native quantile witnesses, full bootstrap reconstruction, and complete raw/CQR endpoint matching support the stated fixed-forest replay claim. They do not imply new independent training or new effectiveness evidence.
- **C2 — Any candidate achieves the joint coverage/width/score objective: no.** All four eligible candidates fail at least one of the nine criteria on 2014 validation and both evaluation periods. The report's negative conclusion is supported; retaining CQR follows the recorded selection rule.
- **C3 — Descriptive gains/tradeoffs: partial overall, yes for the report's explicitly narrow numerical statements.** In 2014, conditional DualTail gains **1.533356** and **3.607920 percentage points** of up-ramp/near-zero coverage but increases width **2.770025%**, and overall IS/WIS worsen. In 2012, conditional PIT improves WIS **3.141396%** relative to CQR and CRPS **5.02%** relative to raw ECDF, but near-zero coverage falls **3.041195 percentage points** and near-zero IS worsens. Those are dataset/arm/metric-specific retrospective observations. They do not isolate state-conditioning effects, tie randomization, physical boundary extension, a physical zero mechanism, novel algorithm superiority, or deployment validity.

Suggested claim: **On the same 17 fixed historical QRF models and P0-eligible samples, full-ECDF PIT and two-tail recalibration show dataset-specific tradeoffs; none meets the predefined joint target. The retained choice is rolling CQR.** Confidence is high in this scoped numerical conclusion; causal and broad-generalization claims remain unsupported.

## Action items, missing evidence, and next experiments

There is no unresolved computational repair required by this audit. Keep all current retrospective, fixed-model, single-PIT-realization, coverage-theorem and deployment qualifiers; do not reinterpret the failed joint goal as success or relax its thresholds after seeing these evaluation results. The broken link/control labeling items are resolved.

Before stronger claims, prespecify a limited follow-up on unused dates or sites. Measure PIT-seed sensitivity and isolate boundary extension and conditional pooling using matched controls; report state support and common-median WIS. If inference is central, examine block-length sensitivity and an explicit multiplicity policy. If operational claims are central, model actual observation/NWP availability and missing-input service, with latency evidence. Retraining/multiple-seed experiments are needed only for claims extending beyond these frozen forests. No such follow-up was run or authorized by this review.

Routing: retain CQR, archive P1 as a qualified negative/descriptive result, and plan any stronger evidence prospectively. The semantic verdict remains **same-family / provisional**.
