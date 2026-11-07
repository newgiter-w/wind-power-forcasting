# Wind Power Probabilistic Forecasting — Route A

本仓库是从现有风电项目中物理复制的独立路线。另一个会话继续原路线，本路线只在独立副本修改。

**当前版本：v0.1.1；状态：项目隔离、环境核验与研究方案完成，尚未实现/验证新方法。**

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

[详细技术方案](docs/ROUTE_A_PLAN.md) · [隔离与快照说明](docs/ISOLATION.md) · [版本记录](CHANGELOG.md) · [历史README](README_LEGACY.md)

## 安全入口

```powershell
Set-Location -LiteralPath 'D:/Study/wind_probability_route_a'
& '.venv/Scripts/python.exe' tools/run_route_a.py --check
```

新脚本应使用`route_a.paths.new_run_dir()`创建独立结果目录，并通过`tools/run_route_a.py --script <相对脚本路径>`启动。历史`out*`缓存只读复用；不在原项目启动本路线实验。

GitHub版本仅包含方案、源代码与隔离工具；17GB级历史权重和结果保存在本地副本中。文献中的方法属于借鉴来源，当前版本不声称创新性或预测效果提升。

## 代码快照

既有155个Python源文件完整保存在[源码压缩包](archives/route-a-v0.1.1-legacy-python.zip)。新路线代码位于`route_a/`，隔离入口位于`tools/`。`RELEASE_MANIFEST.json`记录发布文件和本地版本对应关系；原始数据与模型不在包内。
