"""Create publication-exportable PNG figures from saved experiment outputs."""
import argparse
import json
import os
from pathlib import Path

from .common import ROOT

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 180, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_training_eda(base, train_end, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train = base.loc[base.timestamp.lt(pd.Timestamp(train_end))]
    names = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    fig, ax = plt.subplots(figsize=(8, 4))
    rates = train[[n + '_raw' for n in names]].isna().mean() * 100
    ax.bar(names, rates, color='#347e95')
    ax.set(ylabel='Missing observations (%)', title='Training period: missing data')
    save_figure(fig, directory / 'eda_missing.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    series = train.groupby(train.timestamp.dt.month)['PM2.5_raw'].mean()
    ax.plot(series.index, series.values, marker='o', color='#347e95')
    ax.set(xlabel='Calendar month', ylabel='Mean PM2.5 (µg/m³)',
           title='Training period: seasonal PM2.5 pattern', xticks=range(1, 13))
    save_figure(fig, directory / 'eda_monthly.png')
    matrix = train[[n + '_raw' for n in names]].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
           yticklabels=names, title='Training period: Spearman correlations')
    fig.colorbar(im, ax=ax, label='Spearman correlation')
    save_figure(fig, directory / 'eda_correlation.png')


def plot_run(directory):
    directory = Path(directory)
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(directory / 'metrics.csv')
    for horizon, frame in summary.groupby('horizon'):
        aggregate = frame.groupby('model', sort=False).agg(mae=('mae', 'mean'),
                   std=('mae', 'std'), n_features=('n_features', 'mean'))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].bar(aggregate.index, aggregate.mae, yerr=aggregate['std'].fillna(0), color='#347e95', capsize=3)
        axes[0].set(ylabel='MAE (µg/m³)', title=f'{frame.split.iloc[0].title()} | h={horizon} | common samples')
        axes[1].bar(aggregate.index, aggregate.n_features, color='#c3934a')
        axes[1].set(ylabel='Model input features')
        axes[1].tick_params(axis='x', rotation=45)
        save_figure(fig, figures / f'model_comparison_h{horizon}.png')
    if (directory / 'ablation.csv').exists():
        ablation = pd.read_csv(directory / 'ablation.csv')
        for horizon, frame in ablation.groupby('horizon'):
            data = frame.groupby('ablated_model').delta_mae.mean()
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(data.index, data.values, color=np.where(data >= 0, '#347e95', '#b65348'))
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(ylabel='Ablated MAE − M3 MAE (µg/m³)', title=f'Fixed-parameter ablations | h={horizon}')
            ax.tick_params(axis='x', rotation=15)
            save_figure(fig, figures / f'ablation_h{horizon}.png')
    groups = pd.read_csv(directory / 'group_metrics.csv')
    station = groups.loc[groups.dimension == 'station']
    for horizon, frame in station.groupby('horizon'):
        table = frame.pivot_table(index='group', columns='model', values='mae', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(11, max(3, len(table) * .35)))
        im = ax.imshow(table, aspect='auto', cmap='YlOrRd')
        ax.set(xticks=range(len(table.columns)), xticklabels=table.columns,
               yticks=range(len(table)), yticklabels=table.index,
               title=f'MAE by station | h={horizon}')
        ax.tick_params(axis='x', rotation=45)
        fig.colorbar(im, ax=ax, label='MAE (µg/m³)')
        save_figure(fig, figures / f'station_errors_h{horizon}.png')
    first_seed = int(summary.seed.iloc[0])
    for model_id in ['M3', 'H1']:
        paths = sorted((directory / 'predictions').glob(f'{model_id}_h*_s{first_seed}.parquet'))
        if not paths:
            continue
        path = next((p for p in paths if '_h1_' in p.name), paths[0])
        pred = pd.read_parquet(path)
        station_name = 'Aotizhongxin' if 'Aotizhongxin' in pred.station.values else sorted(pred.station.unique())[0]
        pred = pred.loc[(pred.station == station_name) & pred.eligible_common].sort_values('target_time')
        pred = pred.set_index('target_time')
        first_month = pred.index.min().to_period('M')
        first = pred.loc[pred.index.to_period('M') == first_month]
        daily = (pred.y_true - pred.y_pred).abs().resample('D').agg(['sum', 'count'])
        totals = daily.rolling(7, min_periods=7).sum()
        daily_score = (totals['sum'] / totals['count']).dropna()
        windows = [('first_month', first)]
        if len(daily_score):
            end = daily_score.idxmax() + pd.Timedelta(days=1)
            windows.append(('worst_7_days', pred.loc[(pred.index >= end - pd.Timedelta(days=7)) & (pred.index < end)]))
        for label, subset in windows:
            subset = subset[['y_true', 'y_pred']].asfreq('h')
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(subset.index, subset.y_true, color='#252b37', linewidth=1, label='Observed')
            ax.plot(subset.index, subset.y_pred, color='#d1773e', linewidth=1, label=model_id)
            ax.set(ylabel='PM2.5 (µg/m³)', title=f'{station_name} | {path.stem} | {label.replace("_", " ")}')
            ax.legend()
            fig.autofmt_xdate()
            save_figure(fig, figures / f'{path.stem}_{label}.png')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pred.y_pred - pred.y_true, bins=50, color='#347e95')
        ax.set(xlabel='Prediction − observation (µg/m³)', ylabel='Samples', title=f'{path.stem}: residuals')
        save_figure(fig, figures / f'{path.stem}_residuals.png')
        model_report = directory / 'models' / f'{path.stem}.json'
        info = json.loads(model_report.read_text())['regime']
        if info:
            fig, ax = plt.subplots(figsize=(10, 4))
            matrix = np.asarray(info['centers_standardized'])
            im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
            ax.set(xticks=range(matrix.shape[1]), xticklabels=info['input_columns'],
                   yticks=range(matrix.shape[0]),
                   yticklabels=[f'Cluster {i} (n={n})' for i, n in enumerate(info['training_counts'])],
                   title='Training meteorological regime centers')
            ax.tick_params(axis='x', rotation=35)
            fig.colorbar(im, ax=ax, label='Standardized value')
            save_figure(fig, figures / f'{path.stem}_regimes.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    plot_run(parser.parse_args().run)
