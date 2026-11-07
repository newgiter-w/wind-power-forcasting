"""Generate the P1 report strictly from saved tables; no model or threshold search."""
import argparse,json
import pandas as pd
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from route_a.p1_run import ARMS,REFERENCE


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=require_local_write(ROOT/args.run)
    m=pd.read_csv(run/'metrics_macro.csv');s=pd.read_csv(run/'scores_macro.csv');g=pd.read_csv(run/'joint_gates.csv')
    b=pd.read_csv(run/'paired_calendar_blocks.csv');bm=pd.read_csv(run/'boundary_metrics.csv');state=pd.read_csv(run/'state_support.csv')
    selected=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    # These narrative conclusions belong to this frozen P1 result, not arbitrary runs.
    assert selected['selected']==REFERENCE and g[g.stage.eq('evaluation')&g.eligible&g.joint_pass].empty
    def interval(year,method,group='all',stage='evaluation'):
        return m[m.year.eq(year)&m.stage.eq(stage)&m.method.eq(method)&m.confidence.eq(.9)&m.dimension.eq('regime')&m.group.eq(group)].iloc[0]
    def scores(year,method,group='all',stage='evaluation'):
        return s[s.year.eq(year)&s.stage.eq(stage)&s.method.eq(method)&s.group.eq(group)].iloc[0]
    a14=g[g.year.eq(2014)&g.stage.eq('evaluation')&g.method.eq('DualTail84_state')].iloc[0]
    a12=g[g.year.eq(2012)&g.stage.eq('evaluation')&g.method.eq('PIT84_state')].iloc[0]
    crps12=scores(2012,'PIT84_state').CRPS/scores(2012,'QRF_ECDF_raw').CRPS-1
    lines=['# 路线 A：固定 QRF 的完整分布与双尾校准（v0.3.0）','',
        '日期：2026-11-07。P1 范围：两套 GEFCom 共 17 个风场、原 seed0/500 树/leaf10 的森林。没有拟合新森林，也没有改变样本或天气输入。全部日期已在历史研究中被查看，属于开发/回顾性比较，不是全新确认试验。','',
        '## 执行结果与判定','',
        f'2014 验证期按预先固定的联合门槛得到：{selected["eligible_passing"] or "没有候选通过"}。冻结选择为 **{selected["selected"]}**；选择函数不使用两套评价期。该选择也不能恢复这些历史日期的独立性。','',
        f'评价期也没有候选在任一数据集通过全部联合门槛。2014 的条件双尾把上爬坡/近零覆盖分别提高 {a14.up_coverage_gain*100:.2f}/{a14.zero_coverage_gain*100:.2f} 个百分点，但宽度增加 {(a14.width_ratio-1)*100:.2f}%，总体 IS/WIS 也未改善。2012 的条件 PIT 相对 CQR 改善 WIS {(1-a12.WIS_ratio)*100:.2f}%，相对原 ECDF 改善 CRPS {-crps12*100:.2f}%，但近零覆盖下降 {-a12.zero_coverage_gain*100:.2f} 个百分点。因此保留原基线，不宣称完成“两个工况同时改善且宽度受控”的目标。','',
        '本轮判定不是“覆盖率越高越好”：90% 总体覆盖至少 90%，相对 CQR 下降不超过 0.5 个百分点；宽度不超过原 CQR 的 1.01 倍；总体 IS/WIS 不变差；上爬坡和近零覆盖均至少增加 1 个百分点，且两个工况 IS 不变差。任一失败就不能宣称达成联合目标。1% 是事先声明的宽度容许量，不代表宽度完全不增加。','',
        '## 90% 区间与共同评分','',
        '先逐场计算再等权平均。功率、宽度和 IS 按容量单位计。WIS 使用各方法自己的中位数；另提供固定原 q50 的敏感性分析。','',
        '| 数据 | 方法 | 覆盖率 | 宽度 | IS | WIS | 上爬坡覆盖 | 近零覆盖 | 联合门槛 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for year in (2014,2012):
        for method in ARMS:
            a=interval(year,method);ss=scores(year,method);up=interval(year,method,'up_ramp');zero=interval(year,method,'near_zero')
            passed=bool(g[g.year.eq(year)&g.stage.eq('evaluation')&g.method.eq(method)].iloc[0].joint_pass)
            lines.append(f'| {year} | {method} | {a.coverage:.2%} | {a.width:.6f} | {a.IS:.6f} | {ss.WIS:.6f} | {up.coverage:.2%} | {zero.coverage:.2%} | {"参考/原始对照" if method in ("CQR_signed84","QRF_linear_raw","QRF_ECDF_raw") else ("通过" if passed else "未通过")} |')
    lines+=['',
        '全部 80%/90%/95% 结果、上下尾漏率、逐提前量/月份/季节/功率分组及样本数见 [逐场指标](metrics_per_farm.csv)、[宏指标](metrics_macro.csv)。所有方法共用 P0 的发报和评分样本，没有删掉失败风场或扩展服务范围。','',
        '## 相对原 rolling84 CQR 的取舍','']
    for year in (2014,2012):
        lines += [f'### GEFCom{year}','', '| 方法 | 宽度变化 | IS 变化 | WIS 变化 | 上爬坡覆盖变化（百分点） | 近零覆盖变化（百分点） | 失败门槛 |',
            '|---|---:|---:|---:|---:|---:|---|']
        for method in ('DualTail84','PIT84','PIT84_state','DualTail84_state'):
            a=g[g.year.eq(year)&g.stage.eq('evaluation')&g.method.eq(method)].iloc[0]
            failed=[col.replace('pass_','') for col in g.columns if col.startswith('pass_') and not bool(a[col])]
            lines.append(f'| {method} | {(a.width_ratio-1)*100:+.2f}% | {(a.IS_ratio-1)*100:+.2f}% | {(a.WIS_ratio-1)*100:+.2f}% | {a.up_coverage_gain*100:+.2f} | {a.zero_coverage_gain*100:+.2f} | {", ".join(failed) or "无"} |')
        lines+=['']
    lines += ['数字改善只适用于对应方法、数据和指标。没有任何一个覆盖率数字可以替代全部门槛；表中的“通过”也是本次回顾性点估计门槛，不是独立确认或置信保证。','',
        '## 完整分布的 proper score 与边界诊断','',
        '仅三个完整离散分布组报告精确 CRPS；不为区间型 CQR/双尾方案构造额外假想分布。CRPS 越小越好。QRF_ECDF_raw 与 QRF_linear_raw 表示不同，PIT 的 CRPS 应首先与 ECDF 原始组比较。','',
        '| 数据 | 分布方法 | CRPS | 预测近零 Brier | 平均预测近零概率 | 真实近零频率 | 平均零点质量 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for year in (2014,2012):
        for method in ('QRF_ECDF_raw','PIT84','PIT84_state'):
            ss=scores(year,method);a=bm[bm.year.eq(year)&bm.stage.eq('evaluation')&bm.method.eq(method)].mean(numeric_only=True)
            lines.append(f'| {year} | {method} | {ss.CRPS:.6f} | {a.near_zero_Brier:.6f} | {a.mean_predicted_near_zero:.2%} | {a.observed_near_zero:.2%} | {a.mean_zero_mass:.2%} |')
    lines += ['',
        '近零事件定义为真实功率 <=0.01，概率来自完整分布，不是从区间端点反推。预测为零的质量是模型分布的质量；不能据此识别物理零功率机制，测量舍入也可能产生零。PIT=0/1 的物理边界扩展是本方案的一部分，未单独消融，不能把全部差异归因于单一校准机制。','',
        '## 区间一致性与条件池支持','',
        '| 数据 | 方法 | 80/90 嵌套违反小时数 | 90/95 嵌套违反小时数 |', '|---|---|---:|---:|']
    for year in (2014,2012):
        for method in ARMS:
            ss=scores(year,method)
            lines.append(f'| {year} | {method} | {int(ss.nested_80_90)} | {int(ss.nested_90_95)} |')
    lines += ['',
        'CDF 组由同一离散分布取所有分位数，因此保持嵌套。双尾分置信水平校准没有这项自动保证；没有通过静默排序掩盖问题。原 CQR 与双尾保留原始 q50，可能不与独立修正的区间构成一个一致的完整分布。','',
        '| 数据 | 预测状态 | 该状态小时数 | 条件池回退比例（逐场平均） | 平均混合权重 |',
        '|---|---|---:|---:|---:|']
    for year in (2014,2012):
        for label,part in state[state.year.eq(year)&state.stage.eq('evaluation')].groupby('state'):
            lines.append(f'| {year} | {label} | {int(part.cells.sum())} | {part.local_fallback_fraction.mean():.2%} | {part.mean_lambda.mean():.3f} |')
    lines += ['',
        '预测状态在发报时即可计算：近零概率 >=0.5；否则 raw 中位数小时增量 >0.1；否则其他。真实上爬坡/近零只用于事后评价，不能用作发报选择器。条件池以相同预测状态和相同六小时提前量段取样，支持不足回退全局。两个 state 组使用相同条件信息和支持规则；没有按评价表现调整阈值或回退规则。池内小时相关，不能把细胞数量当独立样本数。','',
        '## 配对时间块结果','',
        '完整日历上同步循环抽取 14 天块，1000 次；同一数据集全部风场/方法共享抽样索引，缺 NWP 日期作为零计数保留。差值为方法减参考，IS/WIS/CRPS 负值较好。下表为上爬坡、近零及总体 IS 的 95% 描述性块区间；其余宽度、覆盖和评分结果见 [完整配对表](paired_calendar_blocks.csv)。','',
        '| 数据 | 方法 | 组 | ΔIS | 95% 块区间 |', '|---|---|---|---:|---|']
    for year in (2014,2012):
        for method in ('DualTail84','PIT84','PIT84_state','DualTail84_state'):
            for group in ('all','up_ramp','near_zero'):
                a=b[b.year.eq(year)&b.method.eq(method)&b.reference.eq(REFERENCE)&b.group.eq(group)&b.metric.eq('IS')].iloc[0]
                lines.append(f'| {year} | {method} | {group} | {a.delta:+.6f} | [{a.ci_low:+.6f}, {a.ci_high:+.6f}] |')
    lines += ['',
        '这些区间条件于固定森林、已选择场站与单次预固定随机 PIT。未考虑重训、场站总体、随机 PIT 多种子或多重比较不确定性；时间漂移下也不是严格置信保证。不能把“区间跨零”写成方法等效，也不能把组内区间不跨零写成对所有方法/组都成立的结论。','',
        '## 实现与边界','',
        '- 森林和原输入固定，17 份模型 SHA-256 与 P0 一致，执行前后也一致；没有调用森林 fit。原始线性 QRF/CQR 宏指标与 P0 逐项匹配；双尾 90% 端点与既有独立尾实现一致。',
        '- 完整响应 ECDF 保留各树等权、叶内 bootstrap 响应逆叶大小权重，不以少量抽样响应近似分布；它与 package 的加权线性插值分位核不同，两个原始对照都保留。',
        '- 离散 PIT 使用 F(y-) + U[F(y)-F(y-)]，随机数种子固定。PIT 经验分布通过当前预测逆 CDF 映射成完整预测质量；PIT 恰为 0/1 时映射至物理边界。所有标定样本均来自完整标签已到达的过去 84 个日历日。',
        '- 本实现借鉴 CDF 重校准思路，不是原论文的 SCDRF 复现，也不是已证明的共形预测系统。SCDRF Algorithm 3 需要另训残差森林，与“固定当前 QRF”不兼容。源文献核对与区别见 [方法说明](../../../docs/P1_METHODS.md)。',
        '- 双尾按 alpha/2 分别做有限样本阶统计量；state 版本是条件经验混合修正，没有继承任意漂移/分组下的有限样本保证。',
        '- 不处理 P0 中未发报的 NWP 缺失日；不插补未来标签；不使用事后真实爬坡状态。发报率和评分范围与 P0 相同。',
        '- GEFCom2014 的 NWP 可用性依赖竞赛协议推定；观测上报和天气发布延迟仍按 P0 的零延迟假设，不能冒称已验证实盘部署。','',
        '## 后续执行建议','',
        '以 [联合门槛](joint_gates.csv) 和 [冻结选择](selection.json) 为准保留/拒绝候选；没有通过就保留基线，不追溯放宽宽度或条件覆盖门槛。局部改善只能作为下一轮假设。若继续研究，应先预定验证上的有限改动与未使用日期/场站，再比较条件池支持、边界扩展或同输入时序表示；不继续用本评价期无限调参。','',
        '## 复现与证据','',
        '```powershell',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p1.py',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p1_run.py',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p1_analyze.py --run experiments/route_a/<新运行目录>',
        '```','',
        '运行依赖本地 P0 数组与冻结模型；源码与汇总结果上传 GitHub，原始数据、完整响应权重、逐时预测和模型留在路线 A 本地。输入/代码散列见 input_manifest.json、code_manifest.json；额外分析代码散列见 analysis_completion.json。完整本地目录：`experiments/route_a/20261107_p1_v030`。','',
        '独立审查与结论边界见 [审查报告](EXPERIMENT_AUDIT.md)、[结果到结论判断](CLAIMS_FROM_RESULTS.md)。同模型家族的 fresh-agent 审查按约定标记为 provisional。','']
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('P1_REPORT_WRITTEN',run/'REPORT.md')


if __name__=='__main__':main()
