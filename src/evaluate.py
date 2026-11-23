"""Common-sample scoring, grouped errors and fixed-parameter ablations."""
import numpy as np
import pandas as pd

from .models import ABLATIONS


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Scoring requires nonempty finite targets and predictions.')
    residual = prediction - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {'n': len(y), 'mae': float(np.mean(np.abs(residual))),
            'rmse': float(np.sqrt(np.mean(residual ** 2))),
            'r2': float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None}


def prediction_table(frame, predictions, model, horizon, seed, split, common, clusters=None):
    result = frame[['station', 'origin_time', 'target_time', 'input_missing_fraction']].copy()
    result['y_true'], result['y_pred'] = frame.target.to_numpy(), predictions
    result['model'], result['horizon'], result['seed'], result['split'] = model, horizon, seed, split
    result['eligible_common'] = np.asarray(common)
    result['regime'] = clusters if clusters is not None else -1
    return result


def group_metrics(predictions, high_threshold):
    common = predictions.loc[predictions.eligible_common].copy()
    common['month'] = common.target_time.dt.strftime('%Y-%m')
    common['season'] = common.target_time.dt.month.map(
        {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
         6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'})
    rows = []
    identity = common.iloc[0][['model', 'horizon', 'seed', 'split']].to_dict()
    for dimension in ['station', 'month', 'season', 'regime']:
        for value, part in common.groupby(dimension):
            if dimension == 'regime' and value == -1:
                continue
            rows.append(dict(identity, dimension=dimension, group=str(value), **metrics(part.y_true, part.y_pred)))
    high = common.loc[common.y_true >= high_threshold]
    if len(high):
        rows.append(dict(identity, dimension='high_concentration', group=f'>={high_threshold:.4f}',
                         **metrics(high.y_true, high.y_pred)))
    return rows


def ablation_table(summary):
    rows = []
    for (horizon, seed, split), group in summary.groupby(['horizon', 'seed', 'split']):
        indexed = group.set_index('model')
        if 'M3' not in indexed.index:
            continue
        reference = indexed.loc['M3']
        for model, component in ABLATIONS.items():
            if model in indexed.index:
                other = indexed.loc[model]
                if reference['n'] != other['n']:
                    raise ValueError('Ablation sample counts differ.')
                rows.append(dict(horizon=horizon, seed=seed, split=split, reference_model='M3',
                                 ablated_model=model, removed_component=component, n=int(reference['n']),
                                 mae_full=reference.mae, mae_ablated=other.mae,
                                 delta_mae=other.mae - reference.mae,
                                 feature_count_full=int(reference.n_features),
                                 feature_count_ablated=int(other.n_features)))
    return pd.DataFrame(rows)
