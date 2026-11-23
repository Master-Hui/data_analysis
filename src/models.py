"""Serializable training/inference bundle and explicit experiment variants."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_selection import MeteorologySelector, NumericBlock
from .regime import RegimeEncoder
from .common import ROOT

SPECS = {
    'B2': dict(weather='current', ridge=True),
    'B3': dict(weather='current'),
    'M1': dict(weather='engineered'),
    'M2': dict(weather='engineered', select=True),
    'M3': dict(weather='engineered', select=True, regime=True),
    'H0': dict(history=True),
    'H1': dict(weather='engineered', select=True, regime=True, history=True),
    'A_NO_CLUSTER': dict(weather='engineered', select=True),
    'A_NO_SELECT': dict(weather='engineered', regime=True),
    'A_NO_HISTORY': dict(weather='engineered', select=True, regime=True, no_windows=True),
    'A_NO_WIND': dict(weather='engineered', select=True, regime=True, no_wind=True),
}
ABLATIONS = {'A_NO_CLUSTER': 'regime distances', 'A_NO_SELECT': 'feature selection',
             'A_NO_HISTORY': 'meteorological history', 'A_NO_WIND': 'wind vector/cyclic package'}


class ForecastModel:
    def __init__(self, model_id, config, metadata, seed=42):
        if model_id not in SPECS:
            raise ValueError(f'Unknown model: {model_id}')
        self.model_id, self.config, self.metadata, self.seed = model_id, config, metadata, seed
        self.spec = SPECS[model_id]

    def _columns(self):
        s = self.spec
        continuous, fixed = [], []
        for name, info in self.metadata.items():
            family = info['family']
            if family == 'C' or (family == 'H' and s.get('history')):
                fixed.append(name)
            elif family == 'W' and s.get('weather'):
                if s['weather'] == 'current':
                    current = info['source'] in ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd']
                    if current and not info['history'] and name not in ['met_wind_sin', 'met_wind_cos']:
                        fixed.append(name)
                else:
                    if (s.get('no_windows') and info['history']) or (s.get('no_wind') and info['wind']):
                        continue
                    (fixed if info['kind'] == 'quality' else continuous).append(name)
        categorical = ['station', 'target_weekday']
        if s.get('weather') == 'current' or s.get('no_wind'):
            categorical.append('wd_category')
        return continuous, fixed, categorical

    def fit(self, frame):
        matrix = self.fit_preprocessor(frame)
        return self.fit_regressor(matrix, frame.target.to_numpy())

    def fit_preprocessor(self, frame):
        """Fit only training transforms; separable for exact CV reuse."""
        if frame.target.isna().any() or len(frame) < 3:
            raise ValueError('Training requires at least three observed targets.')
        continuous, fixed, self.categorical = self._columns()
        self.weather_block = NumericBlock().fit(frame, continuous)
        self.fixed_block = NumericBlock().fit(frame, fixed)
        weather = self.weather_block.transform(frame)
        self.selector = None
        if self.spec.get('select'):
            options = self.config['feature_selection']
            self.selector = MeteorologySelector(
                top_k=options['top_k'], threshold=options['spearman_threshold'],
                neighbors=options['mutual_info_neighbors'], seed=self.seed,
                max_samples=options.get('mi_max_samples'),
                cache_directory=str(ROOT / self.config['cache_directory'])
                if self.config.get('cache_directory') else None).fit(weather, frame.target)
        self.categories = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        self.categories.fit(frame[self.categorical])
        self.regime = None
        if self.spec.get('regime'):
            self.regime = RegimeEncoder(self.config['regime'], self.seed,
                                       wind_vectors=not self.spec.get('no_wind')).fit(frame)
        self.feature_names = ((self.selector.selected if self.selector else self.weather_block.columns)
                              + self.fixed_block.columns
                              + self.categories.get_feature_names_out(self.categorical).tolist())
        if self.regime:
            self.feature_names += [f'regime_distance_{i}' for i in range(self.config['regime']['n_clusters'])]
        self.scale = None
        matrix = self.transform(frame)
        if self.spec.get('ridge'):
            self.scale = StandardScaler().fit(matrix)
            matrix = self.scale.transform(matrix).astype('float64')
        self.fit_max_target_time = str(frame.target_time.max())
        self.fit_rows = len(frame)
        self.high_concentration_threshold = float(frame.target.quantile(.9))
        return matrix

    def fit_regressor(self, matrix, y):
        if self.spec.get('ridge'):
            self.estimator = Ridge(alpha=self.config.get('ridge_alpha', 1.0))
        else:
            params = {k: v for k, v in self.config['regressor'].items() if k != 'type'}
            params['random_state'], params['early_stopping'] = self.seed, False
            self.estimator = HistGradientBoostingRegressor(**params)
        self.estimator.fit(matrix, y)
        return self

    def transform(self, frame):
        weather = self.weather_block.transform(frame)
        if self.selector:
            weather = self.selector.transform(weather)
        matrices = [weather.to_numpy(), self.fixed_block.transform(frame).to_numpy(),
                    self.categories.transform(frame[self.categorical])]
        if self.regime:
            matrices.append(self.regime.transform(frame))
        result = np.column_stack(matrices).astype('float32')
        if not np.isfinite(result).all():
            raise ValueError('Non-finite model features after preprocessing.')
        return result

    def design_matrix(self, frame):
        matrix = self.transform(frame)
        if self.scale is not None:
            matrix = self.scale.transform(matrix).astype('float64')
        return matrix

    def predict(self, frame):
        result = np.maximum(0, self.estimator.predict(self.design_matrix(frame)))
        if not np.isfinite(result).all():
            raise ValueError('Model produced non-finite predictions.')
        return result

    def report(self):
        return {'model': self.model_id, 'specification': self.spec, 'seed': self.seed,
                'fit_rows': self.fit_rows, 'fit_max_target_time': self.fit_max_target_time,
                'high_concentration_threshold': self.high_concentration_threshold,
                'n_features': len(self.feature_names), 'feature_names': self.feature_names,
                'weather_medians': self.weather_block.medians.to_dict(),
                'fixed_medians': self.fixed_block.medians.to_dict(),
                'dropped_all_missing': self.weather_block.dropped_empty + self.fixed_block.dropped_empty,
                'selection': self.selector.report() if self.selector else None,
                'regime': self.regime.report() if self.regime else None}
