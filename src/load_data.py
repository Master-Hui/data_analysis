"""Load canonical UCI station files; conflicting duplicates are errors."""
from pathlib import Path

import pandas as pd

MET = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
NUMERIC = MET + POLLUTANTS
WIND = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def raw_files(directory, stations='all'):
    paths = sorted(Path(directory).glob('PRSA_Data_*.csv'))
    if stations != 'all':
        if not isinstance(stations, list) or not stations:
            raise ValueError('stations must be all or a nonempty list.')
        requested = set(stations)
        paths = [p for p in paths if p.name.split('_')[2] in requested]
        found = {p.name.split('_')[2] for p in paths}
        if requested != found:
            raise ValueError(f'Missing stations: {sorted(requested - found)}')
    if not paths:
        raise FileNotFoundError(f'No canonical PRSA station CSVs in {directory}')
    return paths


def load_data(directory, stations='all'):
    paths = raw_files(directory, stations)
    frames = [pd.read_csv(path, na_values=['NA']) for path in paths]
    return regularize(pd.concat(frames, ignore_index=True)), paths


def regularize(frame):
    """Reindex each station before any shifts; never bridge a missing hour."""
    frame = frame.copy()
    required = NUMERIC + ['wd', 'station']
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Missing fields: {sorted(set(required) - set(frame.columns))}')
    if 'timestamp' not in frame:
        frame['timestamp'] = pd.to_datetime(frame[['year', 'month', 'day', 'hour']])
    else:
        frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    if frame.station.isna().any() or frame.timestamp.isna().any():
        raise ValueError('Missing station or timestamp.')
    if not frame.timestamp.eq(frame.timestamp.dt.floor('h')).all():
        raise ValueError('Observations must be aligned to exact hours.')
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    frame = frame[['station', 'timestamp'] + NUMERIC + ['wd']].drop_duplicates()
    if frame.duplicated(['station', 'timestamp']).any():
        conflicts = frame.loc[frame.duplicated(['station', 'timestamp'], keep=False)]
        raise ValueError(f'Conflicting station/time records: {conflicts.head().to_dict("records")}')
    if frame.empty:
        raise ValueError('Input is empty.')
    parts = []
    for station, group in frame.groupby('station', sort=True):
        group = group.set_index('timestamp').sort_index()
        grid = pd.date_range(group.index.min(), group.index.max(), freq='h', name='timestamp')
        group = group.reindex(grid)
        group['station'] = station
        parts.append(group.reset_index())
    return pd.concat(parts, ignore_index=True)
