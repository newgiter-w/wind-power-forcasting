# Route A: isolated probabilistic wind-power forecasting

本会话的唯一可写项目：`D:/Study/wind_probability_route_a`。

## Mandatory isolation
- Never write, run Git mutations, train into, or patch `D:/Study/wind_retrieval_interval`. It belongs to the other chat.
- `D:/Study/wind_datasets` and `D:/Study/wind_power_review` are shared READ-ONLY inputs. Copy material needed for edits into this project.
- This project is a physical copy with an independent `.git` directory, not a worktree, hardlink, symlink, or junction to the original.
- Always set the command workdir to this project. New implementations belong in `route_a/`; new experiment outputs belong in `experiments/route_a/<run-id>/`.
- Treat copied `out*` files as historical evidence. Never rerun legacy scripts using their default output paths; introduce a new route-A output argument first.
- Use `tools/run_route_a.py` for Python experiments. It enables a Python audit-hook write guard: writes outside this project are rejected. It is an accidental-write guard, NOT an OS sandbox and does not constrain arbitrary native child processes.
- Use this project's `.venv/Scripts/python.exe` and install dependencies only through this interpreter. Never change the original project's or system Python packages. `requirements-route-a.txt` pins the initial CPU baseline dependencies; temporal-network dependencies are a later explicit change.
- Do not copy future changes back to the original project. Cross-route comparison uses versioned exported results and reviewed commits only.

## Route and evidence
- Plan: `docs/ROUTE_A_PLAN.md`; releases: `CHANGELOG.md`; snapshot record: `docs/ISOLATION.md`.
- First reproduce QRF + a full-distribution calibration baseline; then consider a small non-crossing multi-quantile temporal model. Target-site adaptation and ensemble NWP are later, separate comparisons.
- Repeatedly viewed GEFCom/Kelmarsh/Hill-of-Towie dates are development or retrospective evidence. Do not rename them fresh holdouts.
- A ramp-alarm improvement is not an interval improvement. Report raw proper scores and calibrated coverage/width separately.
- No new experiment effectiveness claims are established by the isolation release v0.1.0.
- Raw data, model weights, local auth/config files and third-party PDFs stay local. GitHub publication uses an explicit allowlist.
