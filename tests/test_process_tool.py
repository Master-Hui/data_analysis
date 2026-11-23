import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from src.process_tool import ProcessTool
from src.common import ROOT


def imported(raw, tmp_path):
    (tmp_path / 'configs').symlink_to(ROOT / 'configs')
    tool = ProcessTool(tmp_path)
    data = raw.loc[raw.station.eq('Aotizhongxin')].copy()
    data = pd.concat([data.assign(timestamp=data.timestamp + pd.Timedelta(hours=300*i)) for i in range(3)], ignore_index=True)
    options = dict(csv=data.to_csv(index=False), start='2015-01-01', end='2015-03-01')
    tool.ingest(options)
    return tool, data, options


def test_gaps_forward_fill_and_window(raw, tmp_path):
    tool, data, options = imported(raw, tmp_path)
    data.loc[30:34, 'TEMP'] = np.nan
    data = data.drop(index=40)
    options['csv'] = data.to_csv(index=False)
    result = tool.ingest(options)
    assert result['quality']['inserted_hours'] == 1
    assert tool.base.loc[30:32, 'TEMP_input'].notna().all()
    assert tool.base.loc[33:34, 'TEMP_input'].isna().all()
    before = tool.inspect(dict(variable='TEMP', window=6, position=35))
    assert before['observed'] == 1
    assert next(r for r in before['features'] if r['feature'] == 'roll_TEMP_mean_6')['value'] is None
    data.loc[data.timestamp > pd.Timestamp('2015-01-02 11:00'), 'TEMP'] = 900
    options['csv'] = data.to_csv(index=False)
    tool.ingest(options)
    assert tool.inspect(dict(variable='TEMP', window=6, position=35)) == before


def test_real_training_purge_export_and_reset(raw, tmp_path):
    tool, data, options = imported(raw, tmp_path)
    messages = []
    result = tool.train(dict(model='H1', horizon=24, top_k=20, clusters=3), messages.append)
    assert pd.Timestamp(result['max_train_target']) < pd.Timestamp(result['min_test_origin'])
    assert len(messages) == 5
    assert result['forecast']['value'] >= 0
    assert len(result['forecast']['distances']) == 3
    assert result['common_scores']['model']['n'] == result['common_scores']['persistence']['n']
    with zipfile.ZipFile(tool.export_path) as z:
        assert len(z.namelist()) == 9
        saved = json.loads(z.read('run.json'))
        assert saved['config']['regressor']['max_iter'] == 60
        assert b'target_time' in z.read('06_holdout_predictions.csv')
    tool.ingest(options)
    assert tool.model is None and tool.export_path is None


def test_validation_and_atomic_import(raw, tmp_path):
    tool, data, options = imported(raw, tmp_path)
    original = tool.raw.copy()
    with pytest.raises(ValueError):
        tool.ingest({**options, 'end': '2020-01-01'})
    pd.testing.assert_frame_equal(tool.raw, original)
    with pytest.raises(ValueError):
        tool.train({'horizon': 99})
    with pytest.raises(ValueError):
        tool.inspect({'position': 0})
    conflict = pd.concat([data, data.iloc[[0]].assign(TEMP=999)])
    with pytest.raises(ValueError, match='Conflicting'):
        tool.ingest({**options, 'csv': conflict.to_csv(index=False)})
