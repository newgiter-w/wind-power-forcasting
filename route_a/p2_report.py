"""Render P2 evidence without assuming a positive or negative model outcome."""
import argparse,json
import pandas as pd
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=require_local_write(ROOT/args.run)
    m=pd.read_csv(run/'metrics_macro.csv');s=pd.read_csv(run/'scores_macro.csv');g=pd.read_csv(run/'joint_gates.csv')
    gs=pd.read_csv(run/'joint_gates_by_seed.csv');b=pd.read_csv(run/'paired_calendar_blocks.csv')
    train=pd.read_csv(run/'training_diagnostics.csv');fits=pd.read_csv(run/'training_summary.csv')
    selected=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    def row(year,method,group='all'):
        return m[m.year.eq(year)&m.stage.eq('evaluation')&m.method.eq(method)&m.confidence.eq(.9)&m.dimension.eq('regime')&m.group.eq(group)].iloc[0]
    def score(year,method,group='all'):
        return s[s.year.eq(year)&s.stage.eq('evaluation')&s.method.eq(method)&s.group.eq(group)].iloc[0]
    def spread(value,std,digits=6):
        return f'{value:.{digits}f}' if pd.isna(std) else f'{value:.{digits}f} ± {std:.{digits}f}'
    ep=g[g.stage.eq('evaluation')&g.joint_pass]
    tc=g[g.year.eq(2014)&g.stage.eq('evaluation')&g.method.eq('TCN_CQR')].iloc[0]
    gr=g[g.year.eq(2012)&g.stage.eq('evaluation')&g.method.eq('GRU_CQR')].iloc[0]
    near_IS_worse=bool((g[g.stage.eq('evaluation')].zero_IS_ratio>1).all())
    near_statement='本轮所有神经候选的近零 IS 均值都更差。' if near_IS_worse else '不同候选的近零 IS 表现不一致。'
    lines=['# 路线 A：同输入非交叉 MLP/TCN/GRU 比较（v0.4.0）','',
        '日期：2026-11-07。范围：GEFCom2014 10 场、GEFCom2012 7 场；每个网络三个种子，153 个最终网络模型。QRF 固定为 P0 的 seed0。所有日期已在历史研究中查看，属于回顾性开发评价，不能称为新鲜测试或实盘认证。','',
        '## 执行结论','',
        f'按预先固定的 2014 验证期联合门槛，冻结选择为 **{selected["selected"]}**；通过候选为 {selected["eligible_passing"] or "无"}。没有挑选最好种子或做三模型集成。',
        f'评价期点估计通过联合门槛的组合：{ep[["year","method"]].to_dict("records") if len(ep) else "无"}。评价期结果不参与架构选择；门槛通过也不等于独立确认。','',
        f'关键取舍：2014 TCN_CQR 的宽度变化 {(tc.width_ratio-1)*100:+.2f}%，上爬坡覆盖变化 {tc.up_coverage_gain*100:+.2f} 个百分点、近零覆盖变化 {tc.zero_coverage_gain*100:+.2f} 个百分点，但 IS 变化 {(tc.IS_ratio-1)*100:+.2f}%。2012 GRU_CQR 的对应变化为宽度 {(gr.width_ratio-1)*100:+.2f}%、上爬坡 {gr.up_coverage_gain*100:+.2f} 个百分点、近零 {gr.zero_coverage_gain*100:+.2f} 个百分点、IS {(gr.IS_ratio-1)*100:+.2f}%。这些都是种子均值描述，不是独立确认。','',
        '近零覆盖不能替代近零区间质量：'+near_statement+'覆盖频率和漏点惩罚反映不同信息，须结合下表和上下尾漏率判断。','',
        '联合目标：90% 总体覆盖至少 90%，相对 QRF_CQR 的下降不超过 0.5 个百分点；宽度至多增加 1%；总体 IS/WIS 不变差；上爬坡和近零覆盖均至少增加 1 个百分点，且两个工况 IS 不变差。只改善其中一项不构成目标达成，所有失败组合均保留。','',
        '## 同输入、训练与校准契约','',
        '- 原始输入、样本日期、历史 24h、未来 24h NWP、容量归一与有效性掩码均沿用 P0；没有增加历史长度、气象变量、目标场数据或检索邻居。',
        '- 最终填补器与 QRF 的训练期中位数/缺失指示器一致，填补后的值逐元素等于 QRF 缓存。网络标准化只在完整 proper-training 数据上拟合，保持相同信息预算。',
        '- 每个起报时刻组织成 24 个预报时距 token，包含与 QRF 相同的列；历史和完整天气曲线列在 token 间重复。TCN/GRU 编码的是预报时距序列，历史功率仍以 QRF 的原 24h 特征作上下文；不是另加历史窗口或读取未来观测功率。',
        '- 早停仅用 proper-training 内最后 20% 日期，优化用前 80%；内部填补器/标准化也只在优化前缀拟合。选出轮数后重新初始化，再用全部 proper-training 日期训练该轮数，与 QRF 的最终训练标签预算一致。校准/外部验证/评价标签不参与网络拟合或早停。',
        '- MLP 共享每时距 96/32 隐层；TCN 为四层因果卷积、膨胀率 1/2/4/8；GRU 为单层 hidden48。输入全天气曲线在发报时已知，所以卷积的 lead 因果性不是禁止利用已知未来天气。GRU 每个 origin 都从零隐藏状态开始。',
        '- 三族用相同 pinball 目标、AdamW、批大小和预算；参数量近似匹配但不完全相等，网络的归一化等细节也有区别。与 MLP 的对照能检验这组实现的时序收益，不能单独证明收益只由时间结构造成。',
        '- 七分位头用无约束基值加正增量并裁到 [0,1]，允许真实边界值和相等分位数，原始预测按构造非交叉。没有事后按真实工况选择输出。',
        '- 所有模型共用 P0 的逐提前量 signed rolling84 CQR、有限样本阶统计量、完整日标签到达时间、相同初始校准日期、中点回退和物理裁剪。CQR 独立修正各水平，可能再次破坏跨水平嵌套；未额外排序修复，以保持校准规则一致。','',
        '## 原始预测：先看模型本身','',
        '先逐场计算，再对场站等权平均，最后对三个网络种子平均。± 为三个固定训练种子的标准差；QRF 仅一份冻结 seed0，不把缺少重复记为零方差。','',
        '| 数据 | raw 方法 | WIS（均值 ± 种子标准差） | 中位数 MAE | 90% 宽度 | 上爬坡覆盖 | 近零覆盖 |',
        '|---|---|---|---:|---:|---:|---:|']
    for year in (2014,2012):
        for family in ('QRF','MLP','TCN','GRU'):
            name=family+'_raw';a=row(year,name);ss=score(year,name)
            lines.append(f'| {year} | {name} | {spread(ss.WIS,ss.WIS_seed_std)} | {ss.median_MAE:.6f} | {a.width:.6f} | {row(year,name,"up_ramp").coverage:.2%} | {row(year,name,"near_zero").coverage:.2%} |')
    lines += ['',
        '## 相同 CQR 后的核心结果','',
        '| 数据 | 校准方法 | 覆盖率 | 宽度 | IS | WIS（均值 ± 种子标准差） | 上爬坡覆盖 | 近零覆盖 |',
        '|---|---|---:|---:|---:|---|---:|---:|']
    for year in (2014,2012):
        for family in ('QRF','MLP','TCN','GRU'):
            name=family+'_CQR';a=row(year,name);ss=score(year,name)
            lines.append(f'| {year} | {name} | {a.coverage:.2%} | {a.width:.6f} | {a.IS:.6f} | {spread(ss.WIS,ss.WIS_seed_std)} | {row(year,name,"up_ramp").coverage:.2%} | {row(year,name,"near_zero").coverage:.2%} |')
    lines += ['',
        'WIS 是三个中央区间加固定该模型 q50 的加权区间评分；这里恰等于七分位平均 pinball 的两倍。只预测七个分位数，没有构造完整 CDF，所以不报告或补造精确 CRPS。校准不改 q50，raw/CQR 的中位数 MAE 相同。','',
        '## 联合门槛与失败原因','',
        '| 数据 | 方法 | 宽度变化 | IS 变化 | WIS 变化 | 上爬坡覆盖变化（百分点） | 近零覆盖变化（百分点） | 通过种子数/3 | 均值失败项 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for year in (2014,2012):
        for family in ('MLP','TCN','GRU'):
            name=family+'_CQR';a=g[g.year.eq(year)&g.stage.eq('evaluation')&g.method.eq(name)].iloc[0]
            count=int(gs[gs.year.eq(year)&gs.stage.eq('evaluation')&gs.method.eq(name)].joint_pass.sum())
            failed=[c.replace('pass_','') for c in g if c.startswith('pass_') and not bool(a[c])]
            lines.append(f'| {year} | {name} | {(a.width_ratio-1)*100:+.2f}% | {(a.IS_ratio-1)*100:+.2f}% | {(a.WIS_ratio-1)*100:+.2f}% | {a.up_coverage_gain*100:+.2f} | {a.zero_coverage_gain*100:+.2f} | {count} | {", ".join(failed) or "无"} |')
    lines += ['',
        '全部 80/90/95% 区间、月份/季节/功率/逐时距与上下尾漏率见 [宏指标](metrics_macro.csv) 和 [逐场逐种子指标](metrics_per_farm.csv)。逐种子汇总见 metrics_seed_macro.csv、scores_seed_macro.csv；近零的边界值、下尾漏点诊断见 boundary_diagnostics.csv。真实上爬坡和近零标签只作事后分组，不是可部署的门控变量。','',
        '## 网络训练是否正常','',
        '| 数据 | 模型 | 最终模型数 | 参数范围 | 所选轮数均值/最小/最大 | 选择过程触顶次数 | 初始/最终训练 pinball（均值） |',
        '|---|---|---:|---|---|---:|---|']
    for a in train.itertuples():
        lines.append(f'| {a.year} | {a.model} | {a.models} | {int(a.parameters_min)}–{int(a.parameters_max)} | {a.selected_epochs_mean:.1f}/{a.selected_epochs_min}/{a.selected_epochs_max} | {a.hit_epoch_cap} | {a.initial_train_pinball:.6f}/{a.final_train_pinball:.6f} |')
    not_improved=int((fits.final_full_train_pinball>=fits.initial_full_train_pinball).sum())
    lines += ['',
        f'共有 {not_improved}/153 个最终网络的完整训练期损失未低于初始化；这些结果未被删除。153 次内部轮数选择后，另有 153 次完整训练期重新拟合；因此“153 个最终模型”不是仅运行了 153 次优化过程。训练损失下降只能说明执行/学习信号，不证明配置已经充分优化或泛化良好。预算触顶、早停和 seed 差异均保留，不据此宣称所有 TCN/GRU 都优于或劣于 QRF。','',
        '## 非交叉与校准后嵌套','',
        '| 数据 | 方法 | 相邻分位对交叉次数（每种子平均） | 80/90 嵌套违反小时数（每种子平均） | 90/95 嵌套违反小时数（每种子平均） |',
        '|---|---|---:|---:|---:|']
    for year in (2014,2012):
        for family in ('QRF','MLP','TCN','GRU'):
            for suffix in ('raw','CQR'):
                name=family+'_'+suffix;a=score(year,name)
                lines.append(f'| {year} | {name} | {a.adjacent_crossings:.1f} | {a.nested_80_90:.1f} | {a.nested_90_95:.1f} |')
    lines += ['',
        '相邻分位对的交叉计数允许同一小时贡献多个违反分位对；两个嵌套违反列分别是受影响小时数。非交叉指模型原始七分位输出；同一独立水平 CQR 可能破坏该性质。两个 raw/CQR 结果均保留，不把后处理造成的交叉隐去，也不把逐点 90% 覆盖冒充整条 24h 轨迹联合覆盖。','',
        '## 配对统计：与 QRF 及 MLP 比较','',
        '同步循环抽取完整日历 14 天块、1000 次；同数据集所有场、模型与种子共享日期抽样。先算每场每种子，再平均种子；不是把三个网络预测集成为新模型。区间条件于这些已拟合模型，不包含模型族选择/新场站/全部重训变异，且未做多重比较校正。','',
        '| 数据 | 方法 − 参考 | 组 | ΔIS | 95% 时间块区间 |', '|---|---|---|---:|---|']
    for year in (2014,2012):
        for reference in ('QRF_CQR','MLP_CQR'):
            for family in ('TCN','GRU'):
                for group in ('all','up_ramp','near_zero'):
                    a=b[b.year.eq(year)&b.reference.eq(reference)&b.method.eq(family+'_CQR')&b.group.eq(group)&b.metric.eq('IS')].iloc[0]
                    lines.append(f'| {year} | {family}_CQR − {reference} | {group} | {a.delta:+.6f} | [{a.ci_low:+.6f}, {a.ci_high:+.6f}] |')
    lines += ['',
        '原始模型 WIS、覆盖/宽度/上下尾漏率与点误差的全部比较见 [配对表](paired_calendar_blocks.csv)。即使某组区间不跨零，也只支持对应历史模型/指标/组的描述性比较；不跨零不是目标同时达成的替代标准。种子标准差与这些时间块区间刻画不同来源的不确定性，不能混用。','',
        '## 研究边界与下一步','',
        '本轮同时控制了原始信息、训练标签预算和校准规则；没有做架构/学习率的大规模搜索。若目标未同时达成，保留原基线，只将局部得分改善作为后续假设；若某候选通过，也要以预定未使用日期/场站确认。MLP 对照与相近参数量有助于判断时序归纳偏置，但不能从单一配置严格分离容量、归一化、优化和时间结构的因果贡献。','',
        '服务日期、完整目标日评分口径、零观测上报延迟和 GEFCom2014 推定天气发报时钟均与 P0 相同；不填补被排除的 NWP 日，也没有用未来真实功率或场站标签改变发报选择。','',
        '## 复现与证据','',
        '```powershell',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p2.py',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p2_train.py',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p2_evaluate.py --run experiments/route_a/<新运行目录>',
        '& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/p2_analyze.py --run experiments/route_a/<新运行目录>',
        '```','',
        '运行环境与依赖见 [P2 环境说明](../../../docs/P2_COMPUTE_ENV.md)。完整本地目录 `experiments/route_a/20261107_p2_v040`；原始数据、逐时数组、预处理器和检查点不上传 GitHub。源码、训练汇总、汇总指标、输入/代码散列、环境核验及审查记录随版本发布。','',
        '独立审查见 [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)，结果到结论范围见 [CLAIMS_FROM_RESULTS.md](CLAIMS_FROM_RESULTS.md)。fresh-agent 同模型家族审查标记为 provisional；数值核对不等于跨模型独立认证或新方法成功。','']
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('P2_REPORT_WRITTEN',run/'REPORT.md')


if __name__=='__main__':main()
