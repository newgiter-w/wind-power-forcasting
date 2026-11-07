# Wind Power Probabilistic Forecasting — Route A

本仓库是从现有风电项目中物理复制的独立路线。另一个会话继续原路线，本路线只在独立副本修改。

**当前版本：v0.3.0；状态：P1 固定 QRF 的完整分布与双尾校准比较完成，保留原 CQR 基线。**

本轮固定 17 个风场的原森林，比较七组预测。CDF/PIT 方案保持区间嵌套并在 2012 改善部分分布评分；双尾方案改善部分爬坡覆盖。但没有候选同时满足预先声明的上爬坡、近零覆盖、宽度与评分联合门槛，不替换默认方法。

[P1 完整比较报告](docs/results/v0.3.0/REPORT.md) · [联合门槛](docs/results/v0.3.0/joint_gates.csv) · [方法定义与文献区别](docs/P1_METHODS.md)

本轮复现两套 GEFCom 数据共 17 个风场的冻结 QRF 与 CQR；预先指定的两场从头重训也完全一致。结果揭示总体覆盖与条件覆盖的差异、2012 年 NWP 的周期性缺失，以及分置信水平校准后的区间嵌套问题。全部日期均已被历史研究查看，属于回顾性证据。

[P0 结果与诊断报告](docs/results/v0.2.0/REPORT.md) · [逐场与分组结果](docs/results/v0.2.0/metrics_per_farm.csv) · [审计记录](docs/results/v0.2.0/EXPERIMENT_AUDIT.md)

| 项目 | 位置 / 说明 |
|---|---|
| 本路线 | `D:/Study/wind_probability_route_a` |
| 原路线（禁止本会话写入） | `D:/Study/wind_retrieval_interval` |
| 分支 | `route-a/probabilistic-forecasting` |
| 本地完整备份 | 含已有代码、未提交修改、历史结果、模型和独立Git元数据 |
| 原始数据 | `D:/Study/wind_datasets`，共享只读；不上传GitHub |
| 新输出 | `experiments/route_a/<run-id>/` |

研究路线按以下顺序开展：

1. QRF完整预测分布与SCDRF/保形预测系统的复现、比较。
2. 小型TCN/GRU与非交叉多分位功率预测，先匹配QRF输入信息。
3. 目标场概率/轻量输出适配，区分排序和校准收益。
4. 数据条件满足后再考虑集合NWP与天气—功率两源不确定性。

[详细技术方案](docs/ROUTE_A_PLAN.md) · [隔离与快照说明](docs/ISOLATION.md) · [版本记录](CHANGELOG.md)

## 安全入口

```powershell
Set-Location -LiteralPath 'D:/Study/wind_probability_route_a'
& '.venv/Scripts/python.exe' tools/run_route_a.py --check
```

新脚本应使用`route_a.paths.new_run_dir()`创建独立结果目录，并通过`tools/run_route_a.py --script <相对脚本路径>`启动。历史`out*`缓存只读复用；不在原项目启动本路线实验。

GitHub版本包含方案、源代码、汇总结果和隔离工具；17GB级历史权重、原始数据和逐时预测保存在本地副本中。文献中的方法属于借鉴来源，当前版本不声称新方法的创新性或预测效果提升。

## P0 复现

本地输出：`experiments/route_a/20261107_p0_v020`；运行方式与资源约束见 [CPU 环境说明](docs/COMPUTE_ENV.md)。P0 使用冻结模型作全场回放、两场从头重训、独立因果 CQR 实现及同步日历时间块分析。运行需要本地原始 CSV 和冻结模型缓存，GitHub 不附这些资产，详见结果目录中的 `input_manifest.json`。

P1 已完成上述固定森林比较。未来候选需要在验证期预定有限改动并另留确认数据；时序网络、缺 NWP 时的 fallback 均为后续独立实验，本版本没有执行。

## P1 复现

```powershell
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p1.py
& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p1_run.py
```

执行依赖本地 P0 数组及冻结模型；结果目录自动新建，再通过同一入口调用 `route_a/p1_analyze.py --run <目录>`。正式输出在 `experiments/route_a/20261107_p1_v030`。本轮没有新拟合研究森林。完整分布方法是借鉴 CDF/PIT 重校准的改编，**不是**需要另训残差森林的 SCDRF 原算法复现。
