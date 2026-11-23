"""Run validation experiments, optional temporal CV, or frozen final testing.

Examples:
    python -m src.train --config configs/smoke.yaml --output reports/runs/smoke
    python -m src.train --config configs/base.yaml --tune --output reports/runs/validation
    python -m src.train --stage final --frozen reports/runs/validation/frozen_experiment.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import time
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .baselines import baseline_predict
from .common import ROOT, file_manifest, merge_dict, read_config, save_environment, save_json
from .evaluate import ablation_table, group_metrics, metrics, prediction_table
from .features import build_features
from .load_data import load_data
from .models import ABLATIONS, SPECS, ForecastModel
from .preprocess import clean_hourly
from .split import common_mask, partition, split_summary, time_folds


def cross_validate(frame, config, metadata, model_id, cache=None):
    scores = []
    for fold, (train, valid) in enumerate(time_folds(frame, config)):
        # Cache only within this frame/search invocation and exact fold/representation.
        signature = json.dumps({'seed': config['seed'], 'selection': config['feature_selection'],
                                'regime': config['regime'], 'train_start': config['train_start'],
                                'fold_boundaries': config['cv_folds'][fold]}, sort_keys=True)
        key = (id(frame), model_id, fold, signature)
        if cache is not None and key in cache:
            model, x_train, y_train, x_valid, y_valid = cache.pop(key)
            cache[key] = (model, x_train, y_train, x_valid, y_valid)
        else:
            model = ForecastModel(model_id, config, metadata, config['seed'])
            x_train = model.fit_preprocessor(train)
            y_train = train.target.to_numpy()
            mask = common_mask(valid)
            x_valid = model.design_matrix(valid.loc[mask])
            y_valid = valid.loc[mask, 'target'].to_numpy()
            if cache is not None:
                cache[key] = (model, x_train, y_train, x_valid, y_valid)
                while len(cache) > len(config['cv_folds']):
                    cache.popitem(last=False)
        model.config = config
        model.fit_regressor(x_train, y_train)
        pred = np.maximum(0, model.estimator.predict(x_valid))
        scores.append(metrics(y_valid, pred)['mae'])
    if not scores:
        raise ValueError('No temporal CV folds configured.')
    return scores


def choose_config(frame, config, metadata, model_id, logfile):
    """Two-stage search; every candidate refits all transformations per fold."""
    selected, trials = config, []
    prepared_folds = OrderedDict()
    spec, search = SPECS[model_id], config['search']

    def compare(candidates, stage):
        candidates = list(candidates)
        best, best_score = candidates[0], float('inf')
        for i, candidate in enumerate(candidates, 1):
            scores = cross_validate(frame, candidate, metadata, model_id, prepared_folds)
            score = float(np.mean(scores))
            trials.append({'stage': stage, 'trial': i, 'fold_mae': scores, 'mean_mae': score,
                           'feature_selection': candidate['feature_selection'],
                           'regime': candidate['regime'], 'regressor': candidate['regressor'],
                           'ridge_alpha': candidate['ridge_alpha']})
            save_json(logfile, trials)
            print(f'  CV {model_id} {stage} {i}/{len(candidates)} MAE={score:.3f}', flush=True)
            if score < best_score:
                best, best_score = candidate, score
        return best

    if spec.get('ridge'):
        return compare([merge_dict(config, {'ridge_alpha': alpha}) for alpha in search['ridge_alpha']], 'ridge')
    if spec.get('select') or spec.get('regime'):
        ks = search['top_k'] if spec.get('select') else [config['feature_selection']['top_k']]
        clusters = search['clusters'] if spec.get('regime') else [config['regime']['n_clusters']]
        selected = compare([merge_dict(config, {'feature_selection': {'top_k': k},
                                                'regime': {'n_clusters': c}})
                            for k, c in itertools.product(ks, clusters)], 'representation')
    keys = list(search['regressor'])
    combinations = list(itertools.product(*(search['regressor'][k] for k in keys)))
    rng = np.random.default_rng(config['seed'])
    selected_indices = rng.choice(len(combinations), size=min(search['n_trials'], len(combinations)), replace=False)
    candidates = [selected] + [merge_dict(selected, {'regressor': dict(zip(keys, combinations[i]))})
                               for i in selected_indices]
    return compare(candidates, 'regression')


def run(config, output, stage='validate', tune=False, frozen=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    # Validate IDs and dependencies before loading data or creating outputs.
    models = list(dict.fromkeys(config['models']))
    if not models or not set(models).issubset(set(SPECS) | {'B0', 'B1'}):
        raise ValueError('Configuration contains an empty or unknown model list.')
    if config.get('ablations'):
        if 'M3' not in models:
            raise ValueError('Ablations require M3 in the experiment.')
        models.extend(m for m in ABLATIONS if m not in models)
    if any(m in ABLATIONS for m in models) and 'M3' not in models:
        raise ValueError('Fixed-parameter ablations require M3.')
    # M3 must be configured before its fixed-parameter variants.
    models = [m for m in models if m not in ABLATIONS] + [m for m in models if m in ABLATIONS]
    raw, paths = load_data(ROOT / config['raw_directory'], config['stations'])
    source = file_manifest(paths)
    code = file_manifest((ROOT / 'src').glob('*.py'))
    if stage == 'final':
        if frozen is None or frozen['data_manifest'] != source or frozen['code_manifest'] != code:
            raise ValueError('Frozen experiment is missing or source/data hashes have changed. Revalidate first.')
        if tune:
            raise ValueError('Hyperparameter search is not allowed during final testing.')
    # Validation mode never generates features or plots from the test period.
    cutoff = config['test_start'] if stage == 'validate' else config['test_end_exclusive']
    raw = raw.loc[raw.timestamp.lt(pd.Timestamp(cutoff))].copy()
    base, quality = clean_hourly(raw, config['forward_fill_limit_hours'])
    output.mkdir(parents=True)
    for subdir in ['models', 'predictions', 'cv', 'figures']:
        (output / subdir).mkdir()
    save_json(output / 'data_manifest.json', source)
    save_json(output / 'quality.json', quality)
    (output / 'config_snapshot.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    save_environment(output / 'environment.txt')
    from .plot import plot_training_eda
    plot_training_eda(base.loc[base.timestamp.ge(pd.Timestamp(config['train_start']))],
                      config['validation_start'], output / 'figures')
    rows, grouped, partitions, chosen = [], [], {}, {}
    seeds = config.get('seeds', [config['seed']])
    for horizon in config['horizons']:
        print(f'Building h={horizon} features ({len(base):,} station-hours)...', flush=True)
        frame, metadata = build_features(base, horizon, config)
        save_json(output / f'feature_dictionary_h{horizon}.json', metadata)
        train_end = config['validation_start'] if stage == 'validate' else config['test_start']
        end = config['test_start'] if stage == 'validate' else config['test_end_exclusive']
        train = partition(frame, config['train_start'], train_end)
        valid = partition(frame, train_end, end)
        mask = common_mask(valid)
        if not mask.any():
            raise ValueError('No common evaluation samples.')
        assert train.target_time.max() < valid.origin_time.min()
        split_name = 'validation' if stage == 'validate' else 'test'
        partitions[str(horizon)] = {'train': split_summary(train), split_name: split_summary(valid)}
        valid.loc[mask, ['station', 'origin_time', 'target_time']].to_parquet(
            output / f'evaluation_keys_h{horizon}.parquet', index=False)
        chosen[str(horizon)] = {}
        for model_id in models:
            if stage == 'final':
                model_config = frozen['model_configs'][str(horizon)][model_id]
            elif model_id in ABLATIONS:
                model_config = chosen[str(horizon)]['M3']
            elif tune and model_id in SPECS:
                model_config = choose_config(frame, config, metadata, model_id,
                                             output / 'cv' / f'{model_id}_h{horizon}.json')
            else:
                model_config = config
            chosen[str(horizon)][model_id] = model_config
            for seed in config.get('model_seeds', {}).get(model_id, seeds):
                key = f'{model_id}_h{horizon}_s{seed}'
                print(f'Fitting {key}: train={len(train):,}, common evaluation={int(mask.sum()):,}', flush=True)
                start = time.perf_counter()
                model = None
                if model_id in SPECS:
                    model = ForecastModel(model_id, model_config, metadata, seed).fit(train)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = model.predict(valid) if model else baseline_predict(valid, model_id)
                predict_seconds = time.perf_counter() - start
                threshold = float(train.target.quantile(.9))
                clusters = model.regime.labels(valid) if model and model.regime else None
                table = prediction_table(valid, prediction, model_id, horizon, seed, split_name, mask, clusters)
                table.to_parquet(output / 'predictions' / f'{key}.parquet', index=False)
                observed = np.isfinite(prediction)
                if not observed[np.asarray(mask)].all():
                    raise ValueError('Missing model prediction on the common evaluation set.')
                score = metrics(valid.loc[mask, 'target'], prediction[np.asarray(mask)])
                station_maes = table.loc[mask].assign(error=lambda x: abs(x.y_true - x.y_pred)).groupby('station').error.mean()
                rows.append(dict(model=model_id, horizon=horizon, seed=seed, split=split_name,
                                 **score, station_macro_mae=float(station_maes.mean()),
                                 n_features=len(model.feature_names) if model else 1,
                                 fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                                 coverage=float(observed.mean()), common_fraction=float(mask.mean()),
                                 n_labeled=len(valid)))
                grouped.extend(group_metrics(table, threshold))
                if model:
                    package = {'model': model, 'config': model_config, 'horizon': horizon,
                               'metadata': metadata, 'stage': stage, 'seed': seed}
                    model_path = output / 'models' / f'{key}.joblib'
                    joblib.dump(package, model_path, compress=3)
                    save_json(output / 'models' / f'{key}.json', model.report())
                    # Verify serialization on a small deterministic subset.
                    reloaded = joblib.load(model_path)['model']
                    # Use the same batch shape for the serialization check;
                    # BLAS can accumulate float32 batches in different orders.
                    np.testing.assert_allclose(reloaded.predict(valid.iloc[:8]), model.predict(valid.iloc[:8]))
                pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
                pd.DataFrame(grouped).to_csv(output / 'group_metrics.csv', index=False)
                print(f'  {key}: MAE={score["mae"]:.3f}, RMSE={score["rmse"]:.3f}', flush=True)
        del frame, train, valid
    summary = pd.DataFrame(rows)
    ablation = ablation_table(summary)
    if len(ablation):
        ablation.to_csv(output / 'ablation.csv', index=False)
    summary.groupby(['model', 'horizon', 'split'])[['mae', 'rmse', 'r2']].agg(
        ['mean', 'std', 'count']).to_csv(output / 'seed_summary.csv')
    save_json(output / 'split_manifest.json', partitions)
    frozen_result = {'schema_version': 1, 'config': config, 'model_configs': chosen,
                     'data_manifest': source, 'code_manifest': code,
                     'tuning_used': bool(tune), 'selection_metric': 'mean temporal-CV MAE' if tune else 'preconfigured parameters',
                     'test_viewed': stage == 'final', 'evaluation': 'common observed B0/B1 and target, >=24h history',
                     'prediction_rule': 'fixed model, observed history updated causally; clip predictions at zero'}
    from .plot import plot_run
    plot_run(output)
    if stage == 'validate':
        save_json(output / 'frozen_experiment.json', frozen_result)
    else:
        save_json(output / 'evaluated_frozen_experiment.json', frozen_result)
    save_json(output / 'completed.json', {'stage': stage, 'metric_rows': len(summary),
                                         'tuning_used': bool(tune), 'status': 'complete'})
    print(f'Completed {stage}: {output}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--stage', choices=['validate', 'final'], default='validate')
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frozen = None
    if args.stage == 'final':
        if not args.frozen or args.config or args.tune:
            parser.error('Final testing requires --frozen and disallows --config/--tune.')
        frozen = json.loads(args.frozen.read_text())
        if frozen.get('test_viewed'):
            parser.error('Use the validation-stage frozen experiment.')
        config = frozen['config']
    else:
        if args.frozen:
            parser.error('--frozen is only used for final testing.')
        config = read_config(args.config or ROOT / 'configs/base.yaml')
    output = args.output or ROOT / 'reports/runs' / f'{args.stage}_{datetime.now():%Y%m%d_%H%M%S}'
    with threadpool_limits(limits=config.get('threads', 4)):
        run(config, output, args.stage, args.tune, frozen)


if __name__ == '__main__':
    main()
