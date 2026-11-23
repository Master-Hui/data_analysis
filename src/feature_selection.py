"""Training-only imputation and mutual-information/redundancy selection."""
import numpy as np
import pandas as pd
from joblib import Memory
import sklearn
from sklearn.feature_selection import mutual_info_regression


def _statistics(frame, y, neighbors, seed, max_samples, versions):
    """Pure training-data computation; all inputs participate in the cache key."""
    variable = [c for c in frame if frame[c].nunique() > 1]
    reasons = {c: 'constant' for c in frame if c not in variable}
    if not variable:
        return variable, reasons, {}, pd.DataFrame(), len(frame)
    sample = np.arange(len(frame))
    if max_samples and len(sample) > max_samples:
        sample = np.sort(np.random.default_rng(seed).choice(sample, size=max_samples, replace=False))
    scores = mutual_info_regression(frame[variable].iloc[sample], np.asarray(y)[sample],
                                   n_neighbors=min(neighbors, len(sample) - 1), random_state=seed)
    scores = {c: float(s) for c, s in zip(variable, scores)}
    correlation = frame[variable].corr(method='spearman').abs().fillna(0)
    return variable, reasons, scores, correlation, len(sample)


class NumericBlock:
    """Keep column order and training medians, including constant indicators."""
    def fit(self, frame, columns):
        self.requested = list(columns)
        self.columns = [c for c in columns if frame[c].notna().any()]
        self.dropped_empty = sorted(set(columns) - set(self.columns))
        self.medians = frame[self.columns].median()
        return self

    def transform(self, frame):
        missing = set(self.requested) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing prediction features: {sorted(missing)}')
        return frame[self.columns].fillna(self.medians).astype('float32')


class MeteorologySelector:
    def __init__(self, top_k=40, threshold=.95, neighbors=5, seed=42, max_samples=None, cache_directory=None):
        self.top_k, self.threshold, self.neighbors = top_k, threshold, neighbors
        self.seed, self.max_samples = seed, max_samples
        self.cache_directory = cache_directory

    def fit(self, frame, y):
        compute = Memory(self.cache_directory, verbose=0).cache(_statistics) if self.cache_directory else _statistics
        variable, self.reasons, self.scores, correlation, self.fit_rows = compute(
            frame, np.asarray(y), self.neighbors, self.seed, self.max_samples,
            (sklearn.__version__, pd.__version__, np.__version__))
        if not variable:
            self.selected = []
            return self
        ranking = sorted(variable, key=lambda c: (-self.scores[c], c))
        self.selected = [c for c in ['met_wind_sin', 'met_wind_cos'] if c in variable]
        for name in ranking:
            if name in self.selected:
                continue
            redundant = next((c for c in self.selected if correlation.loc[name, c] > self.threshold), None)
            if redundant:
                self.reasons[name] = f'correlated_with:{redundant}'
            elif len(self.selected) >= self.top_k:
                self.reasons[name] = 'top_k_limit'
            else:
                self.selected.append(name)
        return self

    def transform(self, frame):
        return frame[self.selected]

    def report(self):
        return {'selected': self.selected, 'mutual_information': self.scores,
                'removed': self.reasons, 'mi_fit_rows': self.fit_rows,
                'mi_sampling_limit': self.max_samples}
