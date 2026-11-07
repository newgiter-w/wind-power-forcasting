# Route A CPU environment

This document supersedes copied historical `.aris/compute/*` launch paths for Route A.
The historical ledgers refer to the original project and must not be executed here.

- Project: `D:/Study/wind_probability_route_a`.
- Interpreter: `.venv/Scripts/python.exe`, an independent venv without system site-packages.
- Dependency specification: `requirements-route-a.txt`; exact installed versions: `docs/environment.lock.txt`.
- Workload: CPU only, 2 tree / numerical-library threads, one forest at a time. No GPU or remote service required.
- Write guard: `tools/run_route_a.py`; all new outputs under `experiments/route_a/`.
- Baseline: seed 0, 500 trees, minimum leaf size 10, complete leaf responses; no hyperparameter search.
- Evidence: 17 frozen-model replays; independent from-scratch fits only for farm 1 in each dataset.

From the project root in PowerShell:

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script tools/verify_isolation.py
& .venv/Scripts/python.exe tools/run_route_a.py --script qrf_witness.py
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p0.py
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p0_baseline.py
```

The last command creates a new timestamped output directory and runs real data/model
reproduction. It requires local, read-only dataset CSVs and the copied baseline caches;
those assets are not distributed on GitHub. Import checks alone are not evidence of
reproducibility. The first three commands verify isolation, the actual compiled QRF
kernel, hand-calculated interval scores, and future-label invariance of calibration.

The Python audit hook is an accidental-write guard, not an operating-system sandbox.
No dependency installation or environment rebuild is required for P0.

## P1 fixed-forest recalibration

The same interpreter/dependency specification is warm-reused; no packages are
installed. CPU2 threads, one forest at a time. P1 loads the existing 17 seed0
forests and reconstructs full bootstrap leaf-response weights; it never fits a
new research forest. The synthetic unit witness below fits only a three-tree
four-observation fixture, clearly separate from empirical forecasting results.

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p1.py
```

Expected: exact discrete CRPS, weighted inverse-rank ties, frozen bootstrap
leaf weights, current/future-label invariance for all seven arms, and nested
full-CDF quantiles pass. The experiment command is:

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p1_run.py
```

This creates a new output directory. Run `route_a/p1_analyze.py --run <directory>`
through the same launcher afterward. Full distribution weights and hourly
arrays remain local. P1 uses retrospective dates and one fixed randomized-PIT
realization; no new training or finite-sample temporal-coverage theorem is implied.
