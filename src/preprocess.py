"""Causal, station-local cleaning. No fitted statistics or target imputation."""
import numpy as np
import pandas as pd

from .load_data import MET, NUMERIC, POLLUTANTS, WIND


def clean_hourly(frame, fill_limit=3):
    frame = frame.sort_values(['station', 'timestamp']).reset_index(drop=True)
    groups, report = [], {}
    for station, part in frame.groupby('station', sort=False):
        part = part.copy()
        station_report = {}
        for name in NUMERIC:
            raw = part[name].astype(float).replace([np.inf, -np.inf], np.nan)
            invalid = raw.le(0) if name == 'PRES' else (
                raw.lt(0) if name in POLLUTANTS + ['RAIN', 'WSPM'] else pd.Series(False, index=part.index))
            part[name + '_observed'] = part[name]  # Preserve original values separately.
            raw = raw.mask(invalid)
            part[name + '_raw'] = raw
            part[name + '_input'] = raw.ffill(limit=fill_limit) if fill_limit else raw
            part[name + '_missing'] = raw.isna().astype('float32')
            last = part['timestamp'].where(raw.notna()).ffill()
            part[name + '_age'] = (part.timestamp - last).dt.total_seconds() / 3600
            station_report[name] = {'invalid': int(invalid.sum()), 'missing': int(raw.isna().sum())}
        wd = part.wd.where(part.wd.isin(WIND))
        part['wd_raw'] = wd
        part['wd_input'] = wd.ffill(limit=fill_limit) if fill_limit else wd
        part['wd_missing'] = wd.isna().astype('float32')
        last_wind = part.timestamp.where(wd.notna()).ffill()
        part['wd_age'] = (part.timestamp - last_wind).dt.total_seconds() / 3600
        part['dewpoint_above_temperature'] = (part.DEWP_raw > part.TEMP_raw).astype('float32')
        part['history_hours'] = (part.timestamp - part.timestamp.min()).dt.total_seconds() / 3600
        station_report['wd'] = {'missing_or_unknown': int(wd.isna().sum())}
        station_report['dewpoint_above_temperature'] = int(part.dewpoint_above_temperature.sum())
        groups.append(part)
        report[station] = station_report
    return pd.concat(groups, ignore_index=True), report
