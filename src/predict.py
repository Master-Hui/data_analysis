"""Apply a saved model to observed history, without requiring future labels."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .features import build_features
from .load_data import NUMERIC, regularize
from .preprocess import clean_hourly


def predict_history(package, observations, latest=True):
    observations = observations.copy()
    # Pure meteorological models can accept absent pollutant measurements.
    model = package['model']
    required = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd', 'station']
    if model.spec.get('history'):
        required.append('PM2.5')
    missing = set(required) - set(observations.columns)
    if missing:
        raise ValueError(f'Missing required history columns: {sorted(missing)}')
    for name in NUMERIC:
        if name not in observations:
            observations[name] = float('nan')
    hourly = regularize(observations)
    base, _ = clean_hourly(hourly, package['config']['forward_fill_limit_hours'])
    frame, _ = build_features(base, package['horizon'], package['config'])
    frame = frame.loc[frame.history_hours.ge(24)]
    missing_history = set(hourly.station.unique()) - set(frame.station.unique())
    if missing_history:
        raise ValueError(f'At least 25 hourly timestamps (24h span) required per station: {sorted(missing_history)}')
    if latest:
        frame = frame.groupby('station', sort=False).tail(1)
    result = frame[['station', 'origin_time', 'target_time', 'input_missing_fraction']].copy()
    result['predicted_pm25'] = model.predict(frame)
    result['horizon'] = package['horizon']
    result['unit'] = 'ug/m3'
    known = set(model.categories.categories_[model.categorical.index('station')])
    result['unseen_station'] = ~result.station.isin(known)
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Load only your own trusted joblib model files.')
    parser.add_argument('--input', type=Path, required=True, help='CSV of observed station history.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-origins', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new path.')
    package = joblib.load(args.model)
    with threadpool_limits(limits=package['config'].get('threads', 4)):
        result = predict_history(package, pd.read_csv(args.input, na_values=['NA']), not args.all_origins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
