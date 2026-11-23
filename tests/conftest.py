import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.common import ROOT, merge_dict, read_config


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):
        yield


@pytest.fixture
def config():
    return merge_dict(read_config(ROOT / 'configs/base.yaml'), {
        'horizons': [1], 'regressor': {'max_iter': 8, 'min_samples_leaf': 5},
        'feature_selection': {'top_k': 8, 'mi_max_samples': 150},
        'regime': {'n_clusters': 3, 'n_init': 2, 'batch_size': 64}, 'threads': 2})


@pytest.fixture
def raw():
    rng = np.random.default_rng(13)
    parts = []
    for station, offset in [('Aotizhongxin', 0), ('Changping', 100)]:
        n = 300
        index = np.arange(n)
        parts.append(pd.DataFrame({
            'station': station, 'timestamp': pd.date_range('2015-01-01', periods=n, freq='h'),
            'TEMP': 5 + 8*np.sin(index/24) + offset,
            'PRES': 1010 + rng.normal(0, 3, n), 'DEWP': -4 + rng.normal(0, 2, n),
            'RAIN': rng.choice([0., 0., 0., 1.], n), 'WSPM': rng.uniform(.2, 6, n),
            'wd': np.resize(['N', 'NE', 'E', 'S', 'W'], n),
            'PM2.5': 30 + 10*np.sin(index/12) + rng.normal(0, 2, n) + offset,
            'PM10': rng.uniform(30, 60, n), 'SO2': rng.uniform(1, 5, n),
            'NO2': rng.uniform(10, 30, n), 'CO': rng.uniform(100, 500, n),
            'O3': rng.uniform(20, 80, n)}))
    return pd.concat(parts, ignore_index=True)
