"""Baselines use original observed PM2.5, never filled labels."""
import numpy as np


def baseline_predict(frame, model_id):
    if model_id not in ['B0', 'B1']:
        raise ValueError(model_id)
    return frame[model_id].to_numpy(dtype=float)


def baseline_feature_count(model_id):
    return 1
