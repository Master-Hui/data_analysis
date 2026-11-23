"""Audit the original UCI station files without changing observations.

Usage: python scripts/audit_dataset.py
Requires pandas. Outputs metadata JSON and a Markdown quality report.
"""
from pathlib import Path
import hashlib
import json
import platform
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2',
            'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']
NUMERIC = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

def main():
    paths = sorted((ROOT / 'datasets/raw').glob('PRSA_Data_*.csv'))
    assert len(paths) == 12, f'Expected 12 station files, found {len(paths)}'
    frames, stations, manifest = [], [], []
    for path in paths:
        df = pd.read_csv(path, na_values=['NA'])
        assert list(df.columns) == EXPECTED, path.name
        df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.sort_values('timestamp').reset_index(drop=True)
        assert df.station.nunique() == 1
        bad_gaps = int(df.timestamp.diff().dropna().ne(pd.Timedelta(hours=1)).sum())
        stations.append({'station': df.station.iloc[0], 'rows': len(df),
                         'pm25_missing': int(df['PM2.5'].isna().sum()),
                         'start': str(df.timestamp.min()), 'end': str(df.timestamp.max()),
                         'non_hourly_intervals': bad_gaps})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    assert len(df) == 420768
    assert df.station.nunique() == 12
    duplicates = int(df.duplicated(['station', 'timestamp']).sum())
    assert duplicates == 0
    assert all(x['non_hourly_intervals'] == 0 for x in stations)
    for path in sorted((ROOT / 'datasets').rglob('*')):
        if path.is_file() and 'metadata' not in path.parts:
            manifest.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    missing = {c: {'count': int(df[c].isna().sum()), 'percent': float(df[c].isna().mean()*100)} for c in EXPECTED}
    nonnegative = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']
    splits = [('train', '2013-03-01', '2015-09-01'),
              ('validation', '2015-09-01', '2016-03-01'),
              ('test', '2016-03-01', '2017-03-01')]
    counts = []
    for horizon in [1, 6, 24]:
        y = df.groupby('station', sort=False)['PM2.5'].shift(-horizon)
        label_time = df.timestamp + pd.Timedelta(hours=horizon)
        for name, start, end in splits:
            origin = df.timestamp.ge(pd.Timestamp(start)) & df.timestamp.lt(pd.Timestamp(end))
            eligible = origin & label_time.lt(pd.Timestamp(end)) & y.notna()
            counts.append({'horizon_hours': horizon, 'split': name,
                           'raw_origin_rows': int(origin.sum()),
                           'observed_target_rows_before_feature_filtering': int(eligible.sum())})
    result = {'dataset': 'UCI Beijing Multi-Site Air Quality, id=501',
              'source_url': 'https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data',
              'download_url': 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip',
              'download_date': '2026-10-13', 'license': 'CC BY 4.0',
              'python': platform.python_version(), 'pandas': pd.__version__,
              'rows': len(df), 'original_columns': EXPECTED, 'stations': stations,
              'duplicate_station_timestamps': duplicates, 'missing': missing,
              'numeric_ranges': {c: {'min': float(df[c].min()), 'max': float(df[c].max())} for c in NUMERIC},
              'negative_values': {c: int(df[c].lt(0).sum()) for c in nonnegative},
              'wind_directions': sorted(df.wd.dropna().unique().tolist()),
              'dewpoint_above_temperature': int((df.DEWP > df.TEMP).sum()),
              'split_counts': counts, 'files': manifest}
    out = ROOT / 'datasets/metadata'
    out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in manifest), encoding='utf-8')
    lines = ['# 原始数据质量检查', '', '检查对象：从 UCI 官方压缩包提取的 12 个 PRSA 站点 CSV。下载日期：2026-10-13。', '',
             f'实测共 {len(df):,} 行、18 个原始字段；站点时间重复键 {duplicates} 个。各站点时间网格连续。', '',
             '这些是完整性核验结果，不是模型预测结果。以下缺失概况覆盖全数据；模型相关分布分析、阈值学习与特征筛选只使用训练部分。', '',
             '## 站点检查', '', '| 站点 | 行数 | PM2.5 缺失数 | 非一小时间隔数 |', '|---|---:|---:|---:|']
    for x in stations: lines.append(f"| {x['station']} | {x['rows']} | {x['pm25_missing']} | {x['non_hourly_intervals']} |")
    lines += ['', '各站点起止时间均为 2013-03-01 00:00:00 至 2017-02-28 23:00:00。按北京本地小时解释；原文件未附时区元数据。', '',
              '## 字段缺失', '', '| 字段 | 缺失数 | 缺失率 |', '|---|---:|---:|']
    for c, x in missing.items(): lines.append(f"| {c} | {x['count']} | {x['percent']:.4f}% |")
    lines += ['', '## 数值范围', '', '| 字段 | 最小值 | 最大值 |', '|---|---:|---:|']
    for c, x in result['numeric_ranges'].items(): lines.append(f"| {c} | {x['min']} | {x['max']} |")
    lines += ['', f"污染物、降水、风速负数记录：{sum(result['negative_values'].values())}；露点高于气温记录：{result['dewpoint_above_temperature']}。", '',
              '极端高污染值仅列出检查，不据此删除。缺失值仍保留为 NA，未插值、标准化、训练或改写原始文件。', '',
              '## 计划切分的标签可用量', '',
              '按预测发起时间切分，剔除目标时间越过本段右边界及真实目标缺失的样本；尚未剔除历史窗口不足、基线输入缺失等样本。', '',
              '| 预测步长 | 分区 | 原始发起时刻行数 | 有真实标签且未跨界 |', '|---:|---|---:|---:|']
    for x in counts: lines.append(f"| {x['horizon_hours']} h | {x['split']} | {x['raw_origin_rows']} | {x['observed_target_rows_before_feature_filtering']} |")
    lines += ['', '## 复核', '', '`python scripts/audit_dataset.py` 可重新核验并更新本报告、audit.json 与 SHA256SUMS.txt。',
              '原下载外层包包含的 data.csv/test.csv 不作为官方实验划分使用；本项目只读取 raw 下 12 个站点 CSV。', '']
    (out/'数据质量报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'rows': len(df), 'stations': len(stations), 'pm25_missing': missing['PM2.5'],
                      'negative_values': result['negative_values'], 'split_counts': counts}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
