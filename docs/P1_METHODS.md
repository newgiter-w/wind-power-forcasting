# P1 method contract

The original forest stays fixed. Sources checked on 2026-11-07:

- Jonkers et al., Applied Energy 2024, DOI [10.1016/j.apenergy.2024.122900](https://doi.org/10.1016/j.apenergy.2024.122900), section 4.3.3 / Algorithm 3. The full paper was read from the [authors' institutional repository](https://biblio.ugent.be/publication/01HQ8SA9C23K1EA9A2QJRD96N7). SCDRF trains a forest on residual targets and fills its leaves using calibration residuals. Implementing it faithfully would change the existing forest. **It is not implemented or claimed reproduced in this fixed-QRF phase.**
- Kuleshov, Fenner and Ermon, ICML 2018, [Algorithm 1 / section 3.2](https://proceedings.mlr.press/v80/kuleshov18a.html): recalibrate forecast CDFs with an auxiliary monotone probability map. Our empirical-PIT map is an adaptation of that idea, not a reproduction of its isotonic-regression experiment.
- Romano, Patterson and Candès, NeurIPS 2019, [Conformalized Quantile Regression](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html). The separate-tail control assigns half the total error budget to each one-sided score, matching the existing independent-tail baseline semantics.

The complete empirical leaf-response law gives each tree equal mass and each
bootstrap response within that tree's selected leaf inverse-leaf-size weight.
This law is not the original package's linearly interpolated quantile kernel.
Both raw representations are scored separately, so a representation change is
not silently attributed to calibration.

For the discrete response law, calibration PIT uses
`F(y-) + U * (F(y)-F(y-))`, with pre-fixed reproducible uniform variates. At an
issue time, only completed past label days enter the 84-calendar-day pool.
The empirical law of these PIT values is pushed through the current complete
ECDF inverse, using physical bounds for PIT exactly zero or one. This directly
defines a proper finite discrete predictive law, including possible boundary
mass. It is not claimed to be an exact conformal predictive system, to identify
physical atoms from rounded observations, or to guarantee conditional coverage
under temporal dependence/drift. The pseudo-random seed is fixed, not selected
by performance. Any claimed effect is conditional on this PIT realization.

State-conditioned variants use the same issue-time states and shrinkage pools
for both PIT and dual-tail arms. States use predicted near-zero probability or
predicted median rise, never observed future ramp/near-zero labels. See
`route_a/p1_protocol.json` for the fixed thresholds, support checks, all seven
arms, and the joint coverage/width/score gate. All failures remain reported.

PIT arms yield a complete discrete law and exact CRPS. CQR and dual-tail arms
only yield specified intervals and a retained raw median: no CRPS is invented
for them. Common comparisons use WIS and interval scores; a common-median WIS
sensitivity isolates changes in the PIT-calibrated median.
