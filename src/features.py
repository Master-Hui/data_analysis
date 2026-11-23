"""Feature definitions shared by training and inference.

Metadata is explicit: target, baseline and quality-report columns never enter
the model unless listed in its selected feature groups.
"""
import math

import numpy as np
import pandas as pd

from .load_data import MET, WIND


def build_features(base, horizon, config):
    if not 1 <= horizon <= 24:
        raise ValueError('Supported forecast horizons are 1 to 24 hours.')
    parts, metadata = [], {}
    for station, group in base.groupby('station', sort=False):
        group = group.sort_values('timestamp').reset_index(drop=True).copy()
        out = {'station': group.station, 'origin_time': group.timestamp,
               'target_time': group.timestamp + pd.Timedelta(hours=horizon),
               'target': group['PM2.5_raw'].shift(-horizon),
               'B0': group['PM2.5_raw'],
               'B1': group['PM2.5_raw'].shift(24 - horizon),
               'history_hours': group.history_hours}

        def add(name, values, family, source, kind='value', history=False, wind=False):
            out[name] = values
            units = {'TEMP': 'degC', 'DEWP': 'degC', 'TEMP-DEWP': 'degC',
                     'PRES': 'hPa', 'RAIN': 'mm', 'WSPM': 'm/s', 'u': 'm/s',
                     'v_N': 'm/s', 'PM2.5': 'ug/m3'}
            unit = units.get(source, 'dimensionless')
            if kind == 'quality':
                unit = 'hours' if name.endswith('_age') else 'dimensionless'
            metadata[name] = dict(family=family, source=source, kind=kind,
                                  history=history, wind=wind, unit=unit)

        target_time = out['target_time']
        for label, value, period in [('hour', target_time.dt.hour, 24),
                                      ('month', target_time.dt.month - 1, 12)]:
            for func, fn in [('sin', np.sin), ('cos', np.cos)]:
                add(f'time_{label}_{func}', fn(2 * np.pi * value / period), 'C', 'target_calendar')
        out['target_weekday'] = target_time.dt.dayofweek.astype(str)
        out['wd_category'] = group.wd_input.fillna('unknown')

        direction = {name: i * np.pi / 8 for i, name in enumerate(WIND)}
        for suffix in ['raw', 'input']:
            theta = group['wd_' + suffix].map(direction).astype(float)
            group['u_' + suffix] = -group['WSPM_' + suffix] * np.sin(theta)
            group['v_N_' + suffix] = -group['WSPM_' + suffix] * np.cos(theta)
            if suffix == 'input':
                add('met_wind_sin', np.sin(theta), 'W', 'wd', wind=True)
                add('met_wind_cos', np.cos(theta), 'W', 'wd', wind=True)
        add('met_temp_dew_gap', group.TEMP_input - group.DEWP_input, 'W', 'TEMP-DEWP')
        add('q_dewpoint_above_temperature', group.dewpoint_above_temperature,
            'W', 'DEWP>TEMP', kind='quality')

        for variable in MET + ['u', 'v_N', 'PM2.5']:
            family = 'H' if variable == 'PM2.5' else 'W'
            prefix = 'pm25' if family == 'H' else variable
            wind = variable in ['u', 'v_N']
            raw, filled = group[variable + '_raw'], group[variable + '_input']
            add(f'{"hist" if family == "H" else "met"}_{prefix}', filled,
                family, variable, wind=wind)
            if not wind:
                add(f'q_{prefix}_missing', group[variable + '_missing'], family, variable, kind='quality')
                add(f'q_{prefix}_age', group[variable + '_age'], family, variable, kind='quality')
            for lag in config['lags']:
                add(f'lag_{prefix}_{lag}', filled.shift(lag), family, variable, history=True, wind=wind)
            for window in config['windows']:
                minimum = math.ceil(window * config['minimum_observed_fraction'])
                rolling = raw.rolling(window, min_periods=minimum)
                if variable == 'RAIN':
                    add(f'roll_{prefix}_sum_{window}', rolling.sum(), family, variable, history=True)
                else:
                    add(f'roll_{prefix}_mean_{window}', rolling.mean(), family, variable, history=True, wind=wind)
                    add(f'roll_{prefix}_std_{window}', rolling.std(ddof=0), family, variable, history=True, wind=wind)
                add(f'q_{prefix}_coverage_{window}', raw.notna().astype(float).rolling(
                    window, min_periods=1).sum() / window, family, variable,
                    kind='quality', history=True, wind=wind)
            for lag in [3, 24]:
                add(f'diff_{prefix}_{lag}', filled - filled.shift(lag), family, variable, history=True, wind=wind)
        add('q_wd_missing', group.wd_missing, 'W', 'wd', kind='quality', wind=True)
        add('q_wd_age', group.wd_age, 'W', 'wd', kind='quality', wind=True)
        out['input_missing_fraction'] = group[[n + '_missing' for n in MET + ['wd']]].mean(axis=1)
        part = pd.DataFrame(out)
        part[list(metadata)] = part[list(metadata)].astype('float32')
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    return result, metadata
