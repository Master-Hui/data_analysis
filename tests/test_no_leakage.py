import numpy as np
import pandas as pd
import pytest
import joblib

from src.feature_selection import NumericBlock
from src.features import build_features
from src.load_data import NUMERIC, regularize
from src.models import SPECS, ForecastModel
from src.predict import predict_history
from src.preprocess import clean_hourly
from src.split import partition


def test_changing_future_observations_cannot_change_past_features(raw, config):
    cutoff = pd.Timestamp('2015-01-07')
    base, _ = clean_hourly(regularize(raw))
    before, metadata = build_features(base, 6, config)
    changed = raw.copy()
    changed.loc[changed.timestamp > cutoff, NUMERIC] = 9999.
    changed.loc[changed.timestamp > cutoff, 'wd'] = 'S'
    base, _ = clean_hourly(regularize(changed))
    after, _ = build_features(base, 6, config)
    keys = list(metadata) + ['station', 'wd_category', 'target_weekday']
    pd.testing.assert_frame_equal(before.loc[before.origin_time <= cutoff, keys],
                                  after.loc[after.origin_time <= cutoff, keys])


def test_missing_imputation_fits_training_only_and_preserves_order():
    train = pd.DataFrame({'a': [1., 3., np.nan], 'empty': [np.nan]*3, 'b': [7., 7., 7.]})
    block = NumericBlock().fit(train, ['b', 'a', 'empty'])
    result = block.transform(pd.DataFrame({'a': [np.nan, 1000], 'b': [np.nan, 8], 'empty': [3, 4]}))
    assert list(result) == ['b', 'a']
    assert result.iloc[0].tolist() == [7., 2.]
    assert block.medians['a'] == 2.
    with pytest.raises(ValueError, match='Missing'):
        block.transform(pd.DataFrame({'a': [1]}))


@pytest.mark.parametrize('model_id', list(SPECS))
def test_models_serialization_and_no_target_inputs(raw, config, model_id, tmp_path):
    base, _ = clean_hourly(regularize(raw))
    frame, metadata = build_features(base, 1, config)
    train = partition(frame, '2015-01-01', '2015-01-08')
    valid = partition(frame, '2015-01-08', '2015-01-13')
    model = ForecastModel(model_id, config, metadata).fit(train)
    predicted = model.predict(valid)
    assert np.isfinite(predicted).all() and (predicted >= 0).all()
    assert all(name not in model.feature_names for name in ['target', 'B0', 'B1', 'No'])
    if not model.spec.get('history'):
        assert not any('pm25' in name for name in model.feature_names)
    if model_id == 'H0':
        assert not any(name.startswith(('met_', 'regime_')) for name in model.feature_names)
    if model_id == 'A_NO_HISTORY':
        assert not any(name.startswith(('lag_', 'roll_', 'diff_')) for name in model.feature_names)
    if model_id == 'A_NO_WIND':
        assert not any('wind_sin' in name or 'wind_cos' in name for name in model.feature_names)
        assert 'met_u' not in model.regime.names
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(valid), predicted)
    changed = valid.copy()
    changed[['target', 'B0', 'B1']] = 1.e6
    np.testing.assert_allclose(model.predict(changed), predicted)
    assert pd.Timestamp(model.fit_max_target_time) < valid.origin_time.min()


def test_predict_future_from_history_without_future_labels(raw, config):
    base, _ = clean_hourly(regularize(raw))
    frame, metadata = build_features(base, 6, config)
    model = ForecastModel('M3', config, metadata).fit(partition(frame, '2015-01-01', '2015-01-08'))
    observed = raw.loc[raw.timestamp <= pd.Timestamp('2015-01-10')].drop(columns=['PM2.5'])
    package = {'model': model, 'config': config, 'horizon': 6}
    result = predict_history(package, observed)
    assert len(result) == 2
    assert result.target_time.eq(pd.Timestamp('2015-01-10 06:00')).all()
    assert result.predicted_pm25.ge(0).all()
    with pytest.raises(ValueError, match='25 hourly'):
        predict_history(package, observed.groupby('station').tail(12))
