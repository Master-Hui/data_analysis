"""Configuration, provenance and artifact helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (Path, pd.Timestamp)):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=default, allow_nan=False) + '\n', encoding='utf-8')


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping.')
    if config.get('extends'):
        parent = read_config(Path(path).parent / config.pop('extends'))
        config = merge_dict(parent, config)
    dates = [pd.Timestamp(config[k]) for k in
             ['train_start', 'validation_start', 'test_start', 'test_end_exclusive']]
    if not all(a < b for a, b in zip(dates, dates[1:])):
        raise ValueError('Split boundaries must be strictly increasing.')
    if not all(isinstance(h, int) and 1 <= h <= 24 for h in config['horizons']):
        raise ValueError('Horizons must be integers in [1, 24].')
    if config['forward_fill_limit_hours'] < 0:
        raise ValueError('Forward fill limit must be nonnegative.')
    if not 0 < config['minimum_observed_fraction'] <= 1:
        raise ValueError('Minimum observed fraction must be in (0, 1].')
    if config['feature_selection']['top_k'] < 2:
        raise ValueError('top_k must allow the protected wind pair.')
    if any(w <= 0 for w in config['windows']) or any(l <= 0 for l in config['lags']):
        raise ValueError('Windows and lags must be positive.')
    return config


def merge_dict(base, updates):
    result = dict(base)
    for key, value in updates.items():
        result[key] = merge_dict(result[key], value) if (
            isinstance(value, dict) and isinstance(result.get(key), dict)) else value
    return result


def file_manifest(paths, root=ROOT):
    result = []
    for path in sorted(map(Path, paths)):
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path)
        result.append({'path': name, 'bytes': path.stat().st_size,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def save_environment(path):
    proc = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                          capture_output=True, text=True, check=True)
    Path(path).write_text(f'# Python {sys.version}\n' + proc.stdout, encoding='utf-8')
