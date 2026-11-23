"""Training-fitted meteorological regimes, with continuous distance features."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_selection import NumericBlock


class RegimeEncoder:
    def __init__(self, config, seed=42, wind_vectors=True):
        self.config, self.seed, self.wind_vectors = config, seed, wind_vectors

    def fit(self, frame):
        columns = ['met_' + c for c in ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']]
        if self.wind_vectors:
            columns += ['met_u', 'met_v_N']
        self.numeric = NumericBlock().fit(frame, columns)
        values = self.numeric.transform(frame).to_numpy()
        self.names = list(self.numeric.columns)
        self.wind_encoder = None
        if not self.wind_vectors:
            self.wind_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
            encoded = self.wind_encoder.fit_transform(frame[['wd_category']])
            values = np.column_stack([values, encoded])
            self.names += self.wind_encoder.get_feature_names_out(['wd_category']).tolist()
        if not values.shape[1] or len(values) < self.config['n_clusters']:
            raise ValueError('Insufficient meteorological observations for clustering.')
        self.scaler = StandardScaler().fit(values)
        self.kmeans = MiniBatchKMeans(n_clusters=self.config['n_clusters'],
                                     batch_size=self.config['batch_size'], n_init=self.config['n_init'],
                                     random_state=self.seed).fit(self.scaler.transform(values))
        assignments = self.kmeans.predict(self.scaler.transform(values))
        self.counts = np.bincount(assignments, minlength=self.config['n_clusters']).tolist()
        return self

    def _values(self, frame):
        values = self.numeric.transform(frame).to_numpy()
        if self.wind_encoder is not None:
            values = np.column_stack([values, self.wind_encoder.transform(frame[['wd_category']])])
        return self.scaler.transform(values)

    def transform(self, frame):
        return self.kmeans.transform(self._values(frame)).astype('float32')

    def labels(self, frame):
        return self.kmeans.predict(self._values(frame))

    def report(self):
        return {'input_columns': self.names, 'training_counts': self.counts,
                'centers_original_units': self.scaler.inverse_transform(self.kmeans.cluster_centers_).tolist(),
                'centers_standardized': self.kmeans.cluster_centers_.tolist()}
