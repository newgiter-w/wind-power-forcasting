# P0 result-to-claim judgment

Reviewer: fresh GPT-6-Astra ultra; same-family / provisional. Integrity status: WARN. No new-method or submission-readiness acceptance.

## C1_baseline_reproduction

claim_supported: **yes**; confidence: high.

**what_results_support**: Local 17-forest q05/q95 replay and 90% rolling84 CQR endpoint reproduction, two fixed farm-1 seven-quantile seed-0 training witnesses, real-label scoring, and successful documented environment witnesses.

**what_results_dont_support**: 17 from-scratch refits, multiple seeds, official competition equivalence, new-method effectiveness, fresh-holdout or deployed performance, or full-distribution calibration.

**missing_evidence**: None required for the narrow local replay/two-witness claim. Broader training/environment/generalization assertions would require additional refits, clean environments, seeds and uninspected dates/sites.

**suggested_claim_revision**: 在指定本地环境及既有历史划分上，17 个冻结 QRF 的 q05/q95 与 90% rolling84 CQR 完全复现；两套数据各 1 个预先指定风场的 seed=0 从头重训也完全复现。真实标签评分与独立复算一致。

**next_experiments_needed**:

- No further full training is necessary merely to establish the narrow P0 reproduction claim; retain the frozen baseline as a versioned control.

## C2_diagnostic_attribution

claim_supported: **partial**; confidence: medium.

**what_results_support**: Structured service gaps, conditional tail failures despite adequate macro coverage, near-zero losses associated with negative signed corrections, nonnesting, descriptive 2012 IS improvement with wider intervals, and no demonstrated 2014 IS improvement.

**what_results_dont_support**: Causal decomposition among training data, optimization, model capacity and NWP information; sufficient training; a calibration-independent width floor; fault labels inferred from near-zero power; deployable subgroup selection using future labels; novel method effectiveness.

**missing_evidence**: Controlled same-date/input interventions, validation-only optimization or learning-curve controls, same-information model comparisons, latency sensitivity, service-expanded assessment, and an uninspected confirmation split.

**suggested_claim_revision**: P0 定位了可用性、条件覆盖、尾部误差及区间一致性的具体问题，但尚未识别这些误差由训练、表示或 NWP 信息分别造成的比例；后续方法仅作为待检验假设。

**next_experiments_needed**:

- Compare coherent full-distribution calibration with frozen static/rolling controls under identical label-arrival rules; evaluate CDF/nesting validity, boundary semantics, coverage, width and proper scores jointly.
- Include a fixed-budget asymmetric-tail control using only issue-time conditioning information.
- Evaluate a historical-power or strictly causal NWP fallback over the complete calendar, reporting service rate and both matched-subset and expanded-service performance.
- For causal attribution, conduct validation-only learning-curve/model/input interventions and preserve an uninspected date/site split.

The targeted final wording review confirms both wording findings resolved without changing predictions or metrics. See [review follow-up](review_trace/005-wording-followup.response.md).
