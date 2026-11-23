from collections import OrderedDict
import numpy as np

from src.common import merge_dict
from src.features import build_features
from src.feature_selection import MeteorologySelector
from src.load_data import regularize
from src.preprocess import clean_hourly
from src.train import cross_validate


def test_selector_cache_is_identical_and_training_target_sensitive(tmp_path):
    import pandas as pd
    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(size=(100, 5)), columns=list('abcde'))
    y = x.a.to_numpy() + rng.normal(size=100) * .01
    uncached = MeteorologySelector(top_k=3).fit(x, y)
    cached = MeteorologySelector(top_k=3, cache_directory=str(tmp_path)).fit(x, y)
    assert uncached.report() == cached.report()
    again = MeteorologySelector(top_k=3, cache_directory=str(tmp_path)).fit(x, y)
    assert again.report() == cached.report()
    changed = MeteorologySelector(top_k=3, cache_directory=str(tmp_path)).fit(x, x.e.to_numpy())
    assert changed.scores != cached.scores


def test_fold_reuse_preserves_scores_and_respects_representation(raw, config):
    cfg = merge_dict(config, {'train_start': '2015-01-01', 'validation_start': '2015-01-08',
        'cv_folds': [['2015-01-04', '2015-01-06'], ['2015-01-06', '2015-01-08']]})
    base, _ = clean_hourly(regularize(raw))
    frame, metadata = build_features(base, 1, cfg)
    cache = OrderedDict()
    original = cross_validate(frame, cfg, metadata, 'M3')
    first = cross_validate(frame, cfg, metadata, 'M3', cache)
    repeated = cross_validate(frame, cfg, metadata, 'M3', cache)
    np.testing.assert_allclose(original, first)
    np.testing.assert_allclose(first, repeated)
    changed = merge_dict(cfg, {'regressor': {'max_iter': 12}})
    np.testing.assert_allclose(cross_validate(frame, changed, metadata, 'M3', cache),
                               cross_validate(frame, changed, metadata, 'M3'))
    changed = merge_dict(cfg, {'feature_selection': {'top_k': 4}})
    np.testing.assert_allclose(cross_validate(frame, changed, metadata, 'M3', cache),
                               cross_validate(frame, changed, metadata, 'M3'))
    assert len(cache) == 2
