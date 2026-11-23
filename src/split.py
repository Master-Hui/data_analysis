"""Date-based station-synchronous splits with purged future labels."""
import pandas as pd


def partition(frame, start, end, minimum_history=24):
    selected = frame.loc[
        frame.origin_time.ge(pd.Timestamp(start))
        & frame.origin_time.lt(pd.Timestamp(end))
        & frame.target_time.lt(pd.Timestamp(end))
        & frame.target.notna()
        & frame.history_hours.ge(minimum_history)
    ].copy()
    if selected.empty:
        raise ValueError(f'No eligible labeled samples in [{start}, {end}).')
    return selected


def common_mask(frame):
    return frame.target.notna() & frame.B0.notna() & frame.B1.notna() & frame.history_hours.ge(24)


def time_folds(frame, config):
    for start, end in config['cv_folds']:
        if pd.Timestamp(end) > pd.Timestamp(config['validation_start']):
            raise ValueError('CV fold extends outside the training period.')
        train = partition(frame, config['train_start'], start)
        valid = partition(frame, start, end)
        assert train.target_time.max() < valid.origin_time.min()
        yield train, valid


def split_summary(frame):
    return {'n_labeled': len(frame), 'n_common': int(common_mask(frame).sum()),
            'stations': sorted(frame.station.unique().tolist()),
            'origin_start': str(frame.origin_time.min()), 'origin_end': str(frame.origin_time.max()),
            'target_end': str(frame.target_time.max())}
