import numpy as np
import pandas as pd
import pytest

from src.baselines import baseline_predict
from src.features import build_features
from src.load_data import regularize
from src.preprocess import clean_hourly
from src.split import common_mask, partition


def test_missing_hour_is_reindexed_and_not_used_as_next_hour(raw, config):
    raw = raw.drop(index=50)
    base, _ = clean_hourly(regularize(raw))
    frame, _ = build_features(base, 1, config)
    row = frame.loc[(frame.station == 'Aotizhongxin') &
                    (frame.origin_time == pd.Timestamp('2015-01-03 01:00'))].iloc[0]
    assert np.isnan(row.target)  # The next hour is missing, not the next recorded row.
    assert len(frame) == 600


def test_forward_fill_limit_and_station_boundary(raw):
    raw.loc[2:6, 'TEMP'] = np.nan
    raw.loc[300, 'TEMP'] = np.nan
    base, _ = clean_hourly(regularize(raw), 3)
    a = base.loc[base.station == 'Aotizhongxin'].reset_index(drop=True)
    b = base.loc[base.station == 'Changping'].reset_index(drop=True)
    assert a.TEMP_input.iloc[2:5].eq(a.TEMP_raw.iloc[1]).all()
    assert a.TEMP_input.iloc[5:7].isna().all()
    assert np.isnan(b.TEMP_input.iloc[0])
    assert a.TEMP_age.iloc[6] == 5


def test_targets_remain_missing_and_baselines_use_real_observations(raw, config):
    raw.loc[70, 'PM2.5'] = np.nan
    base, _ = clean_hourly(regularize(raw))
    frame, _ = build_features(base, 1, config)
    a = frame.loc[frame.station == 'Aotizhongxin'].reset_index(drop=True)
    assert np.isnan(a.target.iloc[69])
    assert np.isnan(a.B0.iloc[70])
    assert not common_mask(a).iloc[70]
    assert np.isfinite(a.hist_pm25.iloc[70])  # Filled input is permitted; filled label is not.


@pytest.mark.parametrize('horizon', [1, 6, 24])
def test_targets_baselines_and_split_purge(raw, config, horizon):
    base, _ = clean_hourly(regularize(raw))
    frame, _ = build_features(base, horizon, config)
    a = frame.loc[frame.station == 'Aotizhongxin'].reset_index(drop=True)
    assert a.target.iloc[48] == pytest.approx(raw['PM2.5'].iloc[48+horizon])
    assert baseline_predict(a, 'B1')[48] == pytest.approx(raw['PM2.5'].iloc[48+horizon-24])
    train = partition(frame, '2015-01-01', '2015-01-06')
    assert train.target_time.max() < pd.Timestamp('2015-01-06')
    assert train.origin_time.max() < pd.Timestamp('2015-01-06') - pd.Timedelta(hours=horizon)


def test_rain_sum_uses_observations_and_minimum_coverage(raw, config):
    raw.loc[30:35, 'RAIN'] = [1, np.nan, 2, np.nan, 3, 4]
    base, _ = clean_hourly(regularize(raw))
    frame, _ = build_features(base, 1, config)
    a = frame.loc[frame.station == 'Aotizhongxin'].reset_index(drop=True)
    assert np.isnan(a.roll_RAIN_sum_6.iloc[35])  # Only 4/6 observed; threshold is 5.
    assert a.q_RAIN_coverage_6.iloc[35] == pytest.approx(4/6)


def test_circular_wind_vector_sign_and_invalid_direction(raw, config):
    raw.loc[30:32, 'wd'] = ['N', 'E', 'bad']
    raw.loc[30:32, 'WSPM'] = 2.
    base, _ = clean_hourly(regularize(raw), fill_limit=0)
    frame, _ = build_features(base, 1, config)
    a = frame.loc[frame.station == 'Aotizhongxin'].reset_index(drop=True)
    assert a.met_u.iloc[30] == pytest.approx(0, abs=1e-6)
    assert a.met_v_N.iloc[30] == pytest.approx(-2)
    assert a.met_u.iloc[31] == pytest.approx(-2)
    assert np.isnan(a.met_wind_cos.iloc[32])
    assert a.q_wd_missing.iloc[32] == 1


def test_conflicting_duplicates_rejected_and_exact_duplicates_removed(raw):
    assert len(regularize(pd.concat([raw, raw.iloc[[0]]]))) == len(raw)
    conflict = raw.iloc[[0]].copy()
    conflict['TEMP'] += 1
    with pytest.raises(ValueError, match='Conflicting'):
        regularize(pd.concat([raw, conflict]))
