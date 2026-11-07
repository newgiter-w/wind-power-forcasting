# P2 fresh-agent documentation witness

Observed: 2026-11-07 09:22:13 UTC.

Result: both documented witness commands passed (exit 0). This verifies the environment with synthetic unit fixtures only; it is not experiment integrity or effectiveness acceptance.

Project root: D:/Study/wind_probability_route_a.

## Exact invocations and outputs

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script tools/verify_isolation.py
```

Exit status: 0.

```text
{"inside_write": true, "outside_write_blocked": true, "source_read_allowed": true, "source_and_dataset_writes_rejected": true, "guard_loaded_from": "D:\\Study\\wind_probability_route_a\\tools\\runtime_guard\\sitecustomize.py"}
```

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p2.py
```

Exit status: 0.

```text
{"fixture_type": "synthetic_unit_tests_only", "torch": "2.8.0+cpu", "cuda": false, "CPU_threads": 2, "causal_block_future_token_invariance": true, "checks": [{"model": "MLP", "initial_loss": 0.025960678234696388, "final_loss": 0.010088375769555569, "optimizer_steps": 30, "seconds": 0.08660560008138418, "parameters": 3911, "seed_repeat_exact": true}, {"model": "TCN", "initial_loss": 0.026206430047750473, "final_loss": 0.0054374136961996555, "optimizer_steps": 30, "seconds": 0.2542407000437379, "parameters": 13095, "seed_repeat_exact": true}, {"model": "GRU", "initial_loss": 0.02644922584295273, "final_loss": 0.006563213653862476, "optimizer_steps": 30, "seconds": 0.36822369997389615, "parameters": 12439, "seed_repeat_exact": true}]}
```

## Observed versions and baseline pins

Python: 3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)]

Interpreter: D:\Study\wind_probability_route_a\.venv\Scripts\python.exe

All 11 original baseline pins in requirements-route-a.txt match installed versions:

| Package | Expected | Observed | Match |
|---|---|---|---|
| numpy | 2.1.3 | 2.1.3 | yes |
| pandas | 2.2.3 | 2.2.3 | yes |
| scipy | 1.14.1 | 1.14.1 | yes |
| scikit-learn | 1.5.2 | 1.5.2 | yes |
| joblib | 1.4.2 | 1.4.2 | yes |
| threadpoolctl | 3.5.0 | 3.5.0 | yes |
| quantile-forest | 1.4.2 | 1.4.2 | yes |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 | yes |
| pytz | 2024.2 | 2024.2 | yes |
| tzdata | 2024.2 | 2024.2 | yes |
| six | 1.17.0 | 1.17.0 | yes |

Full observed package metadata:

```json
{
  "Jinja2": "3.1.6",
  "MarkupSafe": "3.0.3",
  "filelock": "4.2.4",
  "fsspec": "2026.9.0",
  "joblib": "1.4.2",
  "mpmath": "1.3.0",
  "networkx": "3.7",
  "numpy": "2.1.3",
  "pandas": "2.2.3",
  "pip": "24.2",
  "python-dateutil": "2.9.0.post0",
  "pytz": "2024.2",
  "quantile-forest": "1.4.2",
  "scikit-learn": "1.5.2",
  "scipy": "1.14.1",
  "setuptools": "78.1.0",
  "six": "1.17.0",
  "sympy": "1.14.0",
  "threadpoolctl": "3.5.0",
  "torch": "2.8.0+cpu",
  "typing_extensions": "4.16.0",
  "tzdata": "2024.2"
}
```

## Documentation versus reality

The two P2 commands execute verbatim. Local write, outside-write rejection, protected-source read access, raw quantile constraints, 30 optimizer steps per model, exact seed repeats, and causal TCN future-token invariance all pass. Torch is CPU 2.8.0+cpu, CUDA unavailable, two threads.

- docs/P2_COMPUTE_ENV.md references environment-temporal.lock.txt, which was absent when inspected.
- No .aris/compute/local-windows-cpu.md ledger exists. Existing .aris/compute/local.md contains legacy original-project runtime and invocation; that invocation was not followed because it targets the protected original project.

No environment was rebuilt. No wind data was trained on, packages installed, or source/docs edited. The launcher write guard is an accidental-write guard, not an OS sandbox.


## Targeted documentation conformance follow-up

Observed: 2026-11-07 09:23:58 UTC.

Result: PASS. Initial path/ledger ambiguities above are resolved. Updated P2_COMPUTE_ENV.md explicitly names docs/environment-temporal.lock.txt relative to project root and links the adjacent lock. All 21 lock pins match installed versions. Active .aris/compute/local-windows-cpu.md supersedes historical local.md commands and points to the project venv and temporal spec. Its spec and docs/P2_COMPUTE_SPEC.json have identical SHA256 hashes. No remaining observed documentation divergence.

Both witness commands were re-executed verbatim from project root:

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script tools/verify_isolation.py
```

Exit status: 0.

```text
{"inside_write": true, "outside_write_blocked": true, "source_read_allowed": true, "source_and_dataset_writes_rejected": true, "guard_loaded_from": "D:\\Study\\wind_probability_route_a\\tools\\runtime_guard\\sitecustomize.py"}
```

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p2.py
```

Exit status: 0.

```text
{"fixture_type": "synthetic_unit_tests_only", "torch": "2.8.0+cpu", "cuda": false, "CPU_threads": 2, "causal_block_future_token_invariance": true, "checks": [{"model": "MLP", "initial_loss": 0.025960678234696388, "final_loss": 0.010088375769555569, "optimizer_steps": 30, "seconds": 0.08156089996919036, "parameters": 3911, "seed_repeat_exact": true}, {"model": "TCN", "initial_loss": 0.026206430047750473, "final_loss": 0.0054374136961996555, "optimizer_steps": 30, "seconds": 0.25567109999246895, "parameters": 13095, "seed_repeat_exact": true}, {"model": "GRU", "initial_loss": 0.02644922584295273, "final_loss": 0.006563213653862476, "optimizer_steps": 30, "seconds": 0.34718949999660254, "parameters": 12439, "seed_repeat_exact": true}]}
```

This follow-up remains environment verification with tiny synthetic fixtures only. No wind-data training, installations, or source/document edits were performed by this witness.


