# Targeted wording follow-up

Date: 2026-11-07. Reviewer: GPT-6-Astra, ultra. `review_independence: same-family`; `acceptance_status: provisional`.

**Findings 1 and 2 are resolved. Overall verdict remains WARN; integrity_status remains warn. Baseline reproduction remains yes; diagnostic attribution remains partial.** The remaining WARN expresses the study's retained scope/availability/attribution limitations, not an outstanding numerical defect or these corrected wording issues.

- **Finding 1 / W1 — resolved.** `experiments/route_a/20261107_p0_v020/REPORT.md:7` now explicitly separates successful implementation/replay from adequate training optimization and states that training, model and NWP contributions cannot be separated. The source template agrees at `route_a/p0_report.py:22`. The execution-scope qualifier at `REPORT.md:125` / `route_a/p0_report.py:106` consistently says numerical reproduction does not establish sufficient training.
- **Finding 2 / W2 — resolved.** `REPORT.md:92` limits the absence of narrowing to the tested static/rolling CQR arms, requires a paired comparison for other calibrators, and explicitly denies that the observed width ratio is a performance lower bound. The source template agrees at `route_a/p0_report.py:83`.
- **Finding 3 / W3 — remains resolved by disclosure.** `REPORT.md:125` and `protocol_deviations.json:5–8` accurately distinguish reconstructing/saving the full calendar from checking historical complete-calendar cache equality. The original protocol is preserved; the authoritative calendar-bootstrap record remains identified at `protocol_deviations.json:11–14`.

I directly reread only the targeted report/template passages, the existing execution-scope qualifier and `protocol_deviations.json`. No broad numerical check, model training or environment rebuild was rerun. The prior independent numerical findings remain in `001-review.response.md` and `001-review.verdict.json`; this follow-up does not replace those historical records or enlarge their assurance. Single-seed retrospective scope, only two refits, assumed reporting/release availability and absence of causal attribution or new-method effectiveness evidence remain unchanged limitations.

SHA-256 values distinguish the historical review snapshot from the final wording snapshot checked here:

| Artifact | Historical 001-review hash | Final wording hash |
|---|---|---|
| `experiments/route_a/20261107_p0_v020/REPORT.md` | `6c0cf5d1270731478656d7769f3d9eddc092de0ad5301e4b8df8f6883452bd5b` | `035ebb7abcd738fd764d28b72ea6570eaa40fb2f9cca72479d51cb4e9d99e2b2` |
| `route_a/p0_report.py` | `b84adbca850502553466d535770108d337827ce3f7dd53473a13bbc0af341566` | `d5fee0f4d20d49d75441706d55f918d3bcdbfcf50ca6b6344fadd17c60d48615` |
| `experiments/route_a/20261107_p0_v020/protocol_deviations.json` | `e5dc362a59bd38ffcfdb19417bfac973f9f59ca103d0dae41ce68f9c78f42b05` | `e5dc362a59bd38ffcfdb19417bfac973f9f59ca103d0dae41ce68f9c78f42b05` |

No further wording correction is required for findings 1–3. This is still a same-family provisional review, with no publication or submission acceptance implied.
