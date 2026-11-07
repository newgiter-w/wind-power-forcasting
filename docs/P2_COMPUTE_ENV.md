# P2 isolated temporal-model runtime

Project: `D:/Study/wind_probability_route_a`. Work from this project root only.
Interpreter: `.venv/Scripts/python.exe`. The CPU-only PyTorch extension is
declared in `requirements-route-a-temporal.txt`; all original baseline pins stay
unchanged. Actual installed versions are in `docs/environment-temporal.lock.txt`
(relative to the project root; [lock file](environment-temporal.lock.txt) beside this document).
No system/original-project Python environment is modified.

Hardware inventory: Intel Core Ultra 9 185H, integrated Intel Arc graphics;
no NVIDIA CUDA device. P2 runs eager CPU float32, two compute threads, deterministic
algorithms, one model at a time. No compiler/GPU/cloud service is required.

Run these witness commands verbatim from the project root:

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script tools/verify_isolation.py
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p2.py
```

Expected: project-local writes succeed and outside writes are rejected; all three
synthetic MLP/TCN/GRU fixtures perform 30 real optimizer steps, reduce their loss,
keep raw quantiles noncrossing/bounded, and reproduce exactly at the same seed.
The TCN residual block also passes a future-lead-token invariance check. These
fixtures are environment/unit tests, not empirical wind results.

The original root has a `statistics.py` module. The temporal-model module loads
the standard-library `statistics` module before importing Torch to avoid name
shadowing; it does not edit that inherited source file. The launcher places
temporary/cache files under this project and enables the accidental-write guard.
The guard is not an OS-level sandbox.

P2 training uses the protocol in `route_a/p2_protocol.json`, fixed before real
training: 17 farms, MLP/TCN/GRU, three seeds, internal chronological epoch
selection followed by full proper-training refit. All calibration/evaluation
labels remain outside network fitting and stopping. Predictions/checkpoints go
to a new directory under `experiments/route_a/`. Unit import success alone is
not an environment readiness claim; the optimizer witnesses above are required.
