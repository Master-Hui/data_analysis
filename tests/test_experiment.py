import json
import numpy as np
import pandas as pd
import pytest

from src.common import merge_dict
from src.evaluate import metrics
from src.train import run


@pytest.mark.parametrize('model_id', ['B2', 'M3'])
def test_temporal_hyperparameter_search(raw, config, tmp_path, model_id):
    from src.features import build_features
    from src.load_data import regularize
    from src.preprocess import clean_hourly
    from src.train import choose_config
    cfg = merge_dict(config, {
        'train_start': '2015-01-01', 'validation_start': '2015-01-08',
        'cv_folds': [['2015-01-04', '2015-01-06'], ['2015-01-06', '2015-01-08']],
        'search': {'n_trials': 1, 'top_k': [8], 'clusters': [3], 'ridge_alpha': [1.0],
                   'regressor': {'max_iter': [5], 'max_leaf_nodes': [7],
                                 'min_samples_leaf': [5], 'l2_regularization': [1.0]}}})
    base, _ = clean_hourly(regularize(raw))
    frame, metadata = build_features(base, 1, cfg)
    logfile = tmp_path / 'cv.json'
    result = choose_config(frame, cfg, metadata, model_id, logfile)
    records = json.loads(logfile.read_text())
    assert all(len(r['fold_mae']) == 2 for r in records)
    assert all(np.isfinite(r['mean_mae']) for r in records)
    assert result['test_start'] == config['test_start']


def test_metrics_zero_variance_and_invalid_inputs():
    assert metrics([1, 2], [2, 4])['mae'] == 1.5
    assert metrics([3, 3], [3, 3])['r2'] is None
    with pytest.raises(ValueError):
        metrics([], [])
    with pytest.raises(ValueError):
        metrics([1], [np.nan])


def test_validation_then_frozen_final_run(raw, config, tmp_path, monkeypatch):
    import src.train as train_module
    monkeypatch.setattr(train_module, 'ROOT', tmp_path)
    raw_path = tmp_path / 'raw'
    raw_path.mkdir()
    for name, group in raw.groupby('station'):
        group.to_csv(raw_path / f'PRSA_Data_{name}_synthetic.csv', index=False)
    cfg = merge_dict(config, {
        'stations': 'all', 'raw_directory': 'raw', 'models': ['B0', 'B1', 'M3', 'H1'],
        'ablations': True, 'train_start': '2015-01-01', 'validation_start': '2015-01-07',
        'test_start': '2015-01-10', 'test_end_exclusive': '2015-01-14',
        'cv_folds': [['2015-01-04', '2015-01-05'], ['2015-01-05', '2015-01-07']]})
    validation = tmp_path / 'validation'
    summary = run(cfg, validation)
    assert summary['n'].nunique() == 1
    assert summary.split.eq('validation').all()
    frozen = json.loads((validation / 'frozen_experiment.json').read_text())
    assert frozen['test_viewed'] is False
    assert frozen['model_configs']['1']['A_NO_CLUSTER'] == frozen['model_configs']['1']['M3']
    final = tmp_path / 'final'
    test_metrics = run(cfg, final, stage='final', frozen=frozen)
    assert test_metrics.split.eq('test').all()
    pred = pd.read_parquet(final / 'predictions/M3_h1_s42.parquet')
    assert pred.origin_time.min() >= pd.Timestamp('2015-01-10')
    assert (final / 'figures/model_comparison_h1.png').exists()
    assert (final / 'ablation.csv').exists()
    with pytest.raises(FileExistsError):
        run(cfg, final, stage='final', frozen=frozen)
    (raw_path / 'PRSA_Data_Aotizhongxin_synthetic.csv').write_text('changed')
    # Provenance checks also reject corrupted input before proceeding.
    with pytest.raises((ValueError, KeyError)):
        run(cfg, tmp_path / 'changed', stage='final', frozen=frozen)
