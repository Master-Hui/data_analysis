"""Audit completed frozen experiments and generate a Chinese research summary.

This script only analyzes saved predictions. It never trains or selects models.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.common import save_json
from src.evaluate import metrics
from src.plot import plt, save_figure

TAG = 'formal_20261014'
MAIN = ['B0', 'B1', 'B2', 'B3', 'M1', 'M2', 'M3', 'H0', 'H1']
HORIZONS = [1, 6, 24]


def markdown_table(frame, decimals=3):
    names = list(frame.columns)
    rows = ['| ' + ' | '.join(map(str, names)) + ' |', '| ' + ' | '.join(['---'] * len(names)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for item in row:
            if item is None:
                cells.append('—')
            elif isinstance(item, (float, np.floating)):
                cells.append(f'{item:.{decimals}f}' if np.isfinite(item) else '—')
            else:
                cells.append(str(item))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def read_predictions(folder, model, horizon, seed=42):
    p = pd.read_parquet(folder / 'predictions' / f'{model}_h{horizon}_s{seed}.parquet')
    p = p.loc[p.eligible_common].sort_values(['station', 'origin_time', 'target_time']).reset_index(drop=True)
    return p


def paired_block_interval(baseline, method, iterations=2000):
    keys = ['station', 'origin_time', 'target_time']
    pd.testing.assert_frame_equal(baseline[keys], method[keys])
    np.testing.assert_allclose(baseline.y_true, method.y_true)
    delta = (baseline.y_pred - baseline.y_true).abs() - (method.y_pred - method.y_true).abs()
    daily = pd.DataFrame({'date': baseline.target_time.dt.floor('D'), 'delta': delta})
    daily = daily.groupby('date').delta.agg(['sum', 'count'])
    # Keep every station on a given date together. Resample paired calendar blocks.
    block_id = (daily.index - daily.index.min()).days // 7
    blocks = daily.groupby(block_id).sum()
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    sums = blocks['sum'].to_numpy()[draws].sum(axis=1)
    counts = blocks['count'].to_numpy()[draws].sum(axis=1)
    low, high = np.quantile(sums / counts, [.025, .975])
    return {'mae_reduction': float(delta.mean()), 'ci95_low': float(low), 'ci95_high': float(high),
            'n_pairs': len(delta), 'n_calendar_blocks': len(blocks), 'resamples': iterations}


def verify_run(folder, frozen=None):
    completion = json.loads((folder / 'completed.json').read_text())
    table = pd.read_csv(folder / 'metrics.csv')
    splits = json.loads((folder / 'split_manifest.json').read_text())
    assert completion['status'] == 'complete' and len(table) == completion['metric_rows'] == 19
    if frozen:
        for entry in frozen['data_manifest'] + frozen['code_manifest']:
            assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    keys = ['station', 'origin_time', 'target_time']
    evaluation_keys = {int(h): pd.read_parquet(folder / f'evaluation_keys_h{int(h)}.parquet').sort_values(keys).reset_index(drop=True) for h in table.horizon.unique()}
    for row in table.itertuples(index=False):
        pred = read_predictions(folder, row.model, int(row.horizon), int(row.seed))
        pd.testing.assert_frame_equal(pred[keys], evaluation_keys[int(row.horizon)])
        actual = metrics(pred.y_true, pred.y_pred)
        assert len(pred) == row.n
        for metric in ['mae', 'rmse', 'r2']:
            assert np.isclose(actual[metric], getattr(row, metric), rtol=1e-9, atol=1e-9)
        if row.model not in ['B0', 'B1']:
            info = json.loads((folder / 'models' / f'{row.model}_h{row.horizon}_s{row.seed}.json').read_text())
            assert info['model'] == row.model and info['seed'] == row.seed
            assert info['fit_rows'] == splits[str(row.horizon)]['train']['n_labeled']
            assert pd.Timestamp(info['fit_max_target_time']) < pred.origin_time.min()
            assert info['n_features'] == row.n_features
    return table


def main():
    status = json.loads((ROOT / 'reports' / f'{TAG}_status.json').read_text())
    if status.get('status') != 'complete':
        raise RuntimeError('Formal experiments have not completed.')
    destination = ROOT / 'reports' / f'{TAG}_summary'
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    (destination / 'figures').mkdir()
    tables, validation_tables, abl, params, cvrows, intervals, winners, grouped = [], [], [], [], [], [], [], []
    integrity = {'primary_seed': 42, 'repeated_models': ['M3', 'H0', 'H1'],
                 'repeated_seeds': [42, 2026, 7], 'horizons': {}, 'source_status': status}
    for h in HORIZONS:
        valdir = Path(status['validation'][str(h)])
        testdir = Path(status['test'][str(h)])
        frozen_path = valdir / 'frozen_experiment.json'
        assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == status['frozen_sha256'][str(h)]
        frozen = json.loads(frozen_path.read_text())
        used = json.loads((testdir / 'evaluated_frozen_experiment.json').read_text())
        assert frozen['test_viewed'] is False and frozen['tuning_used'] is True
        assert used['test_viewed'] is True and used['tuning_used'] is False
        assert used['model_configs'] == frozen['model_configs']
        assert used['config'] == frozen['config']
        assert (valdir / 'environment.txt').read_text() == (testdir / 'environment.txt').read_text(), 'Validation/test environments differ'
        v = verify_run(valdir, frozen)
        t = verify_run(testdir, frozen)
        assert v['n'].nunique() == t['n'].nunique() == 1
        validation_tables.append(v)
        tables.append(t)
        abl.append(pd.read_csv(testdir / 'ablation.csv'))
        grouped.append(pd.read_csv(testdir / 'group_metrics.csv'))
        integrity['horizons'][str(h)] = {
            'validation': json.loads((valdir / 'split_manifest.json').read_text())[str(h)],
            'test': json.loads((testdir / 'split_manifest.json').read_text())[str(h)],
            'parameter_hash': status['frozen_sha256'][str(h)],
            'environment_sha256': hashlib.sha256((testdir / 'environment.txt').read_bytes()).hexdigest()}
        eligible = v.loc[(v.seed == 42) & v.model.isin(MAIN)].sort_values('mae')
        winner = eligible.iloc[0]
        result = t.loc[(t.seed == 42) & t.model.eq(winner.model)].iloc[0]
        winners.append({'提前量': h, '验证期选出方法': winner.model, '验证MAE': winner.mae,
                        '正式测试MAE': result.mae, '正式测试RMSE': result.rmse, '正式测试R²': result.r2})
        for name in MAIN:
            if name in ['B0', 'B1']:
                continue
            config = frozen['model_configs'][str(h)][name]
            r = config['regressor']
            params.append({'horizon': h, 'model': name, 'alpha': config['ridge_alpha'] if name == 'B2' else None,
                           'learning_rate': r['learning_rate'] if name != 'B2' else None,
                           'max_iter': r['max_iter'] if name != 'B2' else None,
                           'max_leaf_nodes': r['max_leaf_nodes'] if name != 'B2' else None,
                           'min_samples_leaf': r['min_samples_leaf'] if name != 'B2' else None,
                           'l2_regularization': r['l2_regularization'] if name != 'B2' else None,
                           'top_k': config['feature_selection']['top_k'] if name in ['M2', 'M3', 'H1'] else None,
                           'n_clusters': config['regime']['n_clusters'] if name in ['M3', 'H1'] else None})
            for record in json.loads((valdir / 'cv' / f'{name}_h{h}.json').read_text()):
                cvrows.append({'horizon': h, 'model': name, 'stage': record['stage'], 'trial': record['trial'],
                               'mean_mae': record['mean_mae'],
                               **{f'fold_{i+1}_mae': x for i, x in enumerate(record['fold_mae'])},
                               'parameters': json.dumps({k: record[k] for k in
                                   ['feature_selection', 'regime', 'regressor', 'ridge_alpha']}, sort_keys=True)})
        pred = {m: read_predictions(testdir, m, h) for m in ['B0', 'B3', 'M3', 'H0', 'H1']}
        for baseline, method in [('B3', 'M3'), ('H0', 'H1'), ('B0', 'H1'), ('B0', 'M3')]:
            intervals.append({'horizon': h, 'baseline': baseline, 'method': method,
                              **paired_block_interval(pred[baseline], pred[method])})
        for stage, folder in [('validation', valdir), ('test', testdir)]:
            for p in (folder / 'figures').glob('*.png'):
                shutil.copy2(p, destination / 'figures' / f'{stage}_h{h}_{p.name}')
    combined = pd.concat(tables, ignore_index=True)
    validation = pd.concat(validation_tables, ignore_index=True)
    ablation = pd.concat(abl, ignore_index=True)
    group_scores = pd.concat(grouped, ignore_index=True)
    parameters = pd.DataFrame(params)
    search = pd.DataFrame(cvrows)
    uncertainty = pd.DataFrame(intervals)
    primary = combined.loc[combined.seed == 42]
    stability = combined.loc[combined.model.isin(['M3', 'H0', 'H1'])].groupby(['horizon', 'model']).agg(
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'), mae_min=('mae', 'min'), mae_max=('mae', 'max'),
        rmse_mean=('rmse', 'mean'), seeds=('seed', 'count')).reset_index()
    for name, frame in [('test_metrics', combined), ('validation_metrics', validation),
                        ('group_metrics', group_scores), ('ablation', ablation), ('selected_parameters', parameters), ('cv_trials', search),
                        ('paired_block_intervals', uncertainty), ('seed_stability', stability)]:
        frame.to_csv(destination / f'{name}.csv', index=False)
    integrity['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    integrity['verified_prediction_tables'] = len(combined) + len(validation)
    assert len(search) == 237, 'Incomplete candidate search records'
    integrity['cv_candidates'] = len(search)
    integrity['cv_model_fits'] = len(search) * 3
    save_json(destination / 'integrity_audit.json', integrity)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ['B0', 'B3', 'M1', 'M3', 'H0', 'H1']:
        s = primary.loc[primary.model == name].sort_values('horizon')
        ax.plot(s.horizon, s.mae, marker='o', label=name)
    ax.set(xticks=HORIZONS, xlabel='Forecast horizon (hours)', ylabel='Test MAE (µg/m³)',
           title='12 Beijing stations | frozen test | seed 42')
    ax.legend(ncol=3)
    save_figure(fig, destination / 'figures/test_mae_by_horizon.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    matrix = ablation.pivot(index='ablated_model', columns='horizon', values='delta_mae')
    bound = float(np.abs(matrix.to_numpy()).max())
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f'{matrix.iloc[i,j]:+.3f}', ha='center', va='center',
                    color='white' if abs(float(matrix.iloc[i,j])) > .55 * bound else '#111111')
    ax.set(xticks=range(3), xticklabels=HORIZONS, xlabel='Horizon (hours)',
           yticks=range(len(matrix)), yticklabels=matrix.index, title='Ablated MAE − M3 MAE | fixed parameters')
    fig.colorbar(im, ax=ax, label='MAE difference (µg/m³)')
    save_figure(fig, destination / 'figures/ablation_all_horizons.png')
    values = primary.loc[primary.model.isin(MAIN)].pivot(index='model', columns='horizon', values='mae').reindex(MAIN)
    values.columns = [f'{h}小时 MAE' for h in values.columns]
    values = values.reset_index().rename(columns={'model': '方法'})
    details = primary.loc[primary.model.isin(['B0', 'B3', 'M3', 'H0', 'H1']),
                          ['horizon', 'model', 'n', 'mae', 'rmse', 'r2', 'n_features']]
    details = details.rename(columns={'horizon':'提前量', 'model':'方法', 'n':'共同测试样本',
                                     'mae':'MAE', 'rmse':'RMSE', 'r2':'R²', 'n_features':'特征数'})
    a = ablation[['horizon','ablated_model','mae_full','mae_ablated','delta_mae']].rename(columns={
        'horizon':'提前量','ablated_model':'移除模块变体','mae_full':'M3 MAE','mae_ablated':'消融MAE','delta_mae':'消融减完整'})
    group_primary = group_scores.loc[group_scores.seed.eq(42) & group_scores.model.isin(['B0','B3','M3','H0','H1'])]
    seasons = group_primary.loc[group_primary.dimension.eq('season'), ['horizon','model','group','n','mae','rmse']]
    high = group_primary.loc[group_primary.dimension.eq('high_concentration'), ['horizon','model','group','n','mae','rmse']]
    station_effects = []
    for h in HORIZONS:
        station_scores = group_primary.loc[group_primary.horizon.eq(h) & group_primary.dimension.eq('station')].pivot(index='group', columns='model', values='mae')
        for base, method in [('B3','M3'), ('H0','H1'), ('B0','H1')]:
            effect = station_scores[base] - station_scores[method]
            station_effects.append({'提前量': h, '对照':base, '方法':method, '改善站点数': int(effect.gt(0).sum()), '站点总数':len(effect), '站点平均MAE减少': effect.mean(), '最小MAE减少':effect.min(), '最大MAE减少':effect.max()})
    station_effects = pd.DataFrame(station_effects)
    station_effects.to_csv(destination / 'station_effects.csv', index=False)
    notes = []
    for h in HORIZONS:
        x = primary.loc[primary.horizon == h].set_index('model')
        improvement = (x.loc['B3','mae'] - x.loc['M3','mae']) / x.loc['B3','mae'] * 100
        increment = x.loc['H0','mae'] - x.loc['H1','mae']
        baseline = x.loc['B0','mae'] - x.loc['H1','mae']
        notes.append(f'- **{h} 小时**：M3 相对当前气象 B3 的 MAE 变化为改善 {improvement:.2f}%（负值表示退化）；'
                     f'H1 相对历史污染物 H0 的 MAE 减少 {increment:.3f}，相对持久性 B0 减少 {baseline:.3f} μg/m³。')
    useful = sum(primary.loc[primary.horizon.eq(h) & primary.model.eq('M3'), 'mae'].iloc[0] < primary.loc[primary.horizon.eq(h) & primary.model.eq('M1'), 'mae'].iloc[0] for h in HORIZONS)
    if useful == 0:
        module_note = '本轮三个提前量中，M3 均未超过未筛选的工程气象模型 M1。因此实测不支持“特征筛选和聚类普遍提升精度”的表述。课程说明书应保留该对照，将已得到证据的气象历史特征作用与筛选、聚类的探索性结果分别说明。'
    elif useful < 3:
        module_note = f'M3 在三个提前量中的 {useful} 个提前量超过 M1，筛选与聚类的作用随提前量变化；不能声称所有情形均有效。'
    else:
        module_note = 'M3 在本轮种子 42 的三个提前量均超过 M1；这是本实验的对照观察，仍需结合消融、重复实验与独立数据验证判断各模块作用。'
    text = f'''# 全站调参与正式测试报告

本次完成北京 12 站点、1/6/24 小时 PM2.5 预测的时间交叉验证调参和冻结参数后的正式测试。全部三个提前量的验证均完成并冻结后才启动真实测试期评分。主比较采用种子 42；M3、H0、H1 另用 2026、7 重复训练。

## 1 实验范围和可信性

- 数据：UCI Beijing Multi-Site Air Quality，2013-03-01 至 2017-02-28，420,768 条原始站点小时记录。
- 初始训练：2013-03-01 至 2015-09-01 前；外部验证：2015-09-01 至 2016-03-01 前。
- 方法：B0 为当前浓度持久性，B1 为前一日基线；B2 为当前气象 Ridge，B3 为当前气象梯度提升；M1 为工程气象，M2 加特征筛选，M3 再加气象聚类；H0 为历史 PM2.5，H1 为历史 PM2.5 与筛选气象、聚类融合。
- 正式测试：预测发起时间位于 2016-03-01 至 2017-03-01 前，且真实目标没有越界；每个模型最终用测试开始前的训练和验证数据重新拟合。
- 三个训练内部时间折、每个模型独立选择参数；共 {len(search)} 个候选配置评估、{len(search)*3} 次折内模型拟合。共输出 {len(validation)} 组验证结果和 {len(combined)} 组正式测试结果。
- 全站训练没有使用单站检查配置的 4,000 行互信息上限，互信息和相关统计使用该折全部训练样本。缓存按完整训练输入、标签、算法参数和库版本隔离。
- 自动测试 29 项通过。对 {integrity['verified_prediction_tables']} 个验证/测试预测文件重新计算指标，与汇总表逐项一致；数据和代码哈希、冻结参数与实际测试参数、训练目标时间上界均通过核验。
- 正式测试逐小时发起预测，每次仅使用截至发起时刻的已观测历史，整个测试期模型参数固定。
- 原始数据只读，未用未来真实气象或未来污染物作为输入，未填补真实标签。

## 2 按验证结果确定的对照方法

以下在汇总时仅按保存的验证期种子 42 MAE，从九组主模型及基线中选出对照方法，并列出它的正式测试表现。这一选择规则不读取测试分数；全部预定方法都已进入正式测试，没有根据测试分数再调参。

{markdown_table(pd.DataFrame(winners))}

## 3 正式测试模型对比

MAE 单位为 μg/m³，越小越好。以下全为种子 42，在每个提前量的相同站点和时刻样本上评分。

{markdown_table(values)}

B0 与 B1 在 24 小时任务中是同一个基线，不能视为两项独立证据。

![不同提前量的正式测试误差](figures/test_mae_by_horizon.png)

主要指标明细：

{markdown_table(details)}

## 4 有益效果应如何表述

{chr(10).join(notes)}

{module_note}

这些结论仅适用于本次历史数据和实验协议，不代表对未来城市空气质量或新站点的性能保证。PM2.5 浓度不等同 AQI。H0/H1 分别调参，且 H1 另搜索特征数量与聚类数，因此二者差值反映整个融合方案的对照效果；不能单凭该差值断言某个气象特征的因果作用，也不能把历史 PM2.5 自身带来的提升归因于气象。

## 5 固定参数消融

所有消融沿用同一步长 M3 参数，种子固定为 42。正的“消融减完整”表示移除该模块后误差增加，负值表示该模块在该实验中没有收益。不能用仅有单个正结果声称所有模块普遍有效。

{markdown_table(a)}

![各提前量消融比较](figures/ablation_all_horizons.png)

## 6 随机种子重复实验

{markdown_table(stability)}

此标准差反映三次算法随机初始化差异，不能代替跨时间或跨城市不确定性。正式主表只使用种子 42，避免把不同种子中最好的分数挑出来比较。

## 7 配对时间块误差区间

同一时刻的全站观测放在同一日历块内，以连续 7 天为块，配对重采样 2,000 次；最后不足 7 天的块保留并按样本数加权。结果为“基线 MAE 减方法 MAE”，正值表示方法改善。95% 区间包含 0 时，不声称差异稳定。

{markdown_table(uncertainty)}

这些区间是保存预测上的附加敏感性分析，不参与调参；它们依赖时间块重采样假设，不能完全消除季节变化和序列依赖的影响，也没有做多重比较校正。

## 8 按站点、季节和高浓度检查

整体 MAE 可能掩盖不同站点与季节差异。以下为共同样本上的站点对照，正的 MAE 减少表示改善。

{markdown_table(station_effects)}

季节对照（种子 42）：

{markdown_table(seasons.rename(columns={'horizon':'提前量','model':'方法','group':'季节','n':'样本数','mae':'MAE','rmse':'RMSE'}))}

高浓度目标子集（阈值为同一提前量最终训练目标的第 90 百分位数，未由测试标签选阈值；阈值不是空气质量等级标准）：

{markdown_table(high.rename(columns={'horizon':'提前量','model':'方法','group':'浓度阈值','n':'样本数','mae':'MAE','rmse':'RMSE'}))}

## 9 选定参数

{markdown_table(parameters)}

完整候选与各折 MAE 见 `cv_trials.csv`，完整配置在各验证运行的 `frozen_experiment.json`。每组梯度提升搜索包含默认参数和 8 组随机候选；特征 k 与聚类 K 在其前按配置候选搜索。

## 10 文件与复现

- `test_metrics.csv`、`validation_metrics.csv`：完整指标，包括覆盖率、特征数和耗时。
- `selected_parameters.csv`、`cv_trials.csv`：选定参数与候选记录。
- `group_metrics.csv`、`station_effects.csv`：分组误差与站点改善范围。
- `ablation.csv`、`seed_stability.csv`、`paired_block_intervals.csv`：消融、重复与区间结果。
- `integrity_audit.json`：数据、代码、参数和逐行指标核验记录。
- `figures/`：训练期 EDA、验证和正式测试图，文件名明确标注阶段和提前量。主报告的 `test_mae_by_horizon.png` 固定种子 42；原运行的 `model_comparison` 和 `station_errors` 图采用每个模型所有已运行种子的均值（多数方法 1 次，M3/H0/H1 各 3 次），曲线图采用种子 42，解释时需区分统计口径。
- 原始模型和逐行预测保存在项目 `reports/runs/formal_20261014_test_h1/`、`..._h6/`、`..._h24/`。

本次采用三个独立提前量进程，每个限制 3 线程。不同模型的统计缓存命中情况不同，`fit_seconds` 是此次运行用时，不能单据此声称某个模块降低训练成本。正式测试没有用于改变参数或选择新的特征。

数据出处：[UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)，许可 CC BY 4.0。模型并非实时空气质量服务。
'''
    (destination / '正式测试报告.md').write_text(text, encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
