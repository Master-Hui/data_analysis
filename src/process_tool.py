"""Interactive, inspectable local workflow; never writes formal experiment outputs."""
from pathlib import Path
from io import StringIO
import hashlib
import json
import math
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .common import ROOT, read_config
from .load_data import MET, NUMERIC, raw_files, regularize
from .preprocess import clean_hourly
from .features import build_features
from .models import ForecastModel
from .evaluate import metrics


def records(df):
    return json.loads(df.to_json(orient='records', date_format='iso'))


def safe_json(value):
    return json.loads(json.dumps(value, default=lambda v: v.item() if isinstance(v, np.generic) else str(v), allow_nan=False))


class ProcessTool:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.raw = self.base = self.frame = self.model = None
        self.export_path = None
        self.logs = []

    def catalog(self):
        files = raw_files(self.root / 'datasets/raw')
        return {'stations': [p.name.split('_')[2] for p in files], 'default_start': '2016-01-01', 'default_end': '2016-03-01'}

    def ingest(self, options):
        if options.get('csv'):
            if len(options['csv'].encode()) > 8_000_000:
                raise ValueError('CSV 文件不能超过 8 MB。')
            df = pd.read_csv(StringIO(options['csv']))
            source = '用户上传 CSV'
        else:
            station = options.get('station', 'Aotizhongxin')
            choices = {p.name.split('_')[2]: p for p in raw_files(self.root / 'datasets/raw')}
            if station not in choices:
                raise ValueError('请选择列表中的站点。')
            df = pd.read_csv(choices[station])
            source = choices[station].name
        if 'timestamp' not in df:
            if not set(['year', 'month', 'day', 'hour']).issubset(df):
                raise ValueError('需要 timestamp 或 year/month/day/hour 时间字段。')
            df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df['timestamp'] = pd.to_datetime(df.timestamp, errors='raise')
        if df.timestamp.dt.tz is not None:
            raise ValueError('请使用北京时间、不带时区的整点时间。')
        required = MET + ['wd', 'station', 'PM2.5']
        if not set(required).issubset(df):
            raise ValueError('缺少必需字段：' + ', '.join(sorted(set(required) - set(df))))
        start, end = pd.Timestamp(options['start']), pd.Timestamp(options['end'])
        if not start < end or end - start > pd.Timedelta(days=366):
            raise ValueError('结束日期须晚于开始日期，范围不得超过 366 天（结束日期不包含在内）。')
        df = df.loc[(df.timestamp >= start) & (df.timestamp < end)].copy()
        if df.empty or len(df) > 20000 or df.station.nunique() != 1:
            raise ValueError('每次导入一个站点，日期内需有 1–20000 行数据。')
        if df.timestamp.isna().any() or df.station.isna().any():
            raise ValueError('站点和时间不能为空。')
        for c in NUMERIC:
            if c not in df:
                df[c] = np.nan
        raw = regularize(df)
        quality = {"input_rows": len(df), "regularized_rows": len(raw), "inserted_hours": len(raw) - df.timestamp.nunique()}
        base, cleaning = clean_hourly(raw, fill_limit=3)
        # Replace session atomically only after validation succeeds.
        self.raw, self.base = raw, base
        self.frame = self.model = None
        self.export_path = None
        self.source = {'source': source, 'station': str(raw.station.iloc[0]), 'requested_start': str(start),
                       'requested_end_exclusive': str(end), 'sha256': hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()}
        self.logs = [f'导入 {len(df)} 行；补齐逐小时时间轴后 {len(raw)} 行。', '清洗：非法值置空；仅向前填充，最多连续 3 小时；保留原始缺失标记。']
        return {'rows': len(raw), 'start': str(raw.timestamp.min()), 'end': str(raw.timestamp.max()),
                'source': self.source, 'quality': safe_json(quality), 'cleaning': safe_json(cleaning),
                'missing': [{'name': c, 'count': int(base[c + '_raw'].isna().sum()),
                             'filled': int((base[c + '_raw'].isna() & base[c + '_input'].notna()).sum())} for c in MET + ['PM2.5']],
                'logs': self.logs}

    def inspect(self, options):
        if self.base is None:
            raise ValueError('请先导入数据。')
        variable, window = options.get('variable', 'TEMP'), int(options.get('window', 24))
        if variable not in MET + ['PM2.5'] or window not in [3, 6, 12, 24]:
            raise ValueError('变量或窗口无效。')
        position = int(options.get('position', len(self.base) - 1))
        if not 24 <= position < len(self.base):
            raise ValueError('请选第 25 个小时或更晚的预测发起时刻。')
        cfg = read_config(self.root / 'configs/base.yaml')
        frame, _ = build_features(self.base, 6, cfg)
        row = frame.iloc[position]
        segment = self.base.iloc[max(0, position - 47): position + 1]
        name = 'pm25' if variable == 'PM2.5' else variable
        columns = [c for c in frame.columns if c.startswith(('lag_' + name + '_', 'diff_' + name + '_')) or
                   c in [f'roll_{name}_mean_{window}', f'roll_{name}_std_{window}', f'roll_{name}_sum_{window}', f'q_{name}_coverage_{window}']]
        detail = segment[['timestamp', variable + '_raw', variable + '_input', variable + '_missing', variable + '_age']].copy()
        detail['in_window'] = detail.timestamp >= self.base.timestamp.iloc[position] - pd.Timedelta(hours=window - 1)
        original = self.raw.set_index('timestamp')[variable]
        detail['original'] = detail.timestamp.map(original)
        return {'variable': variable, 'window': window, 'origin': str(row.origin_time), 'minimum': math.ceil(.75 * window),
                'observed': int(self.base.iloc[position - window + 1:position + 1][variable + '_raw'].notna().sum()),
                'series': records(detail), 'features': records(pd.DataFrame({'feature': columns, 'value': [row[c] for c in columns]}))}

    def train(self, options, progress=lambda message: None):
        if self.base is None:
            raise ValueError('请先导入数据。')
        horizon, top_k, clusters = (int(options.get(k, default)) for k, default in [('horizon', 6), ('top_k', 40), ('clusters', 4)])
        model_id = options.get('model', 'H1')
        if horizon not in [1, 6, 24] or top_k not in [20, 40, 60] or clusters not in [3, 4, 5] or model_id not in ['M3', 'H1']:
            raise ValueError('请选择支持的模型和参数。')
        self.model = None
        self.export_path = None
        cfg = read_config(self.root / 'configs/base.yaml')
        cfg['cache_directory'] = None
        cfg['feature_selection'].update(top_k=top_k, mi_max_samples=3000)
        cfg['regime']['n_clusters'] = clusters
        cfg['regressor']['max_iter'] = 60
        progress('1/5 构建因果窗口、风向编码与缺失特征')
        frame, metadata = build_features(self.base, horizon, cfg)
        cutoff = self.base.timestamp.iloc[int(len(self.base) * .8)]
        eligible = frame.target.notna() & frame.history_hours.ge(24)
        train = frame.loc[eligible & frame.origin_time.lt(cutoff) & frame.target_time.lt(cutoff)]
        test = frame.loc[eligible & frame.origin_time.ge(cutoff)]
        if len(train) < 200 or len(test) < 30:
            raise ValueError('有效训练样本至少 200 行、留出样本至少 30 行；请扩大日期范围或补充 PM2.5 观测。')
        if model_id == 'H1' and train.hist_pm25.notna().sum() < 100:
            raise ValueError('H1 需要足够的历史 PM2.5 观测。')
        progress(f'2/5 按时间切分：训练 {len(train)} 行、留出 {len(test)} 行；剔除跨界标签')
        with threadpool_limits(limits=3):
            model = ForecastModel(model_id, cfg, metadata)
            progress('3/5 仅在训练段拟合中位数、互信息筛选、冗余筛除与气象聚类')
            matrix = model.fit_preprocessor(train)
            progress('4/5 训练 60 轮梯度提升树')
            model.fit_regressor(matrix, train.target.to_numpy())
            pred = model.predict(test)
            latest = frame.iloc[[-1]]
            future = float(model.predict(latest)[0])
            distances = model.regime.transform(latest)[0].tolist()
            labels = model.regime.labels(train)
        progress('5/5 计算留出预测；导出全部中间数据与运行参数')
        prediction = test[['origin_time', 'target_time', 'target', 'B0']].copy()
        prediction['prediction'] = pred
        common = prediction.B0.notna()
        common_scores = {'model': metrics(prediction.loc[common, 'target'], prediction.loc[common, 'prediction']),
                         'persistence': metrics(prediction.loc[common, 'target'], prediction.loc[common, 'B0'])} if common.any() else None
        report = model.report()
        selection = report['selection']
        selected = set(selection['selected'])
        ranking = [{'name': k, 'mi': v, 'selected': k in selected,
                    'reason': '保留' if k in selected else selection['removed'].get(k, '未保留')} for k, v in selection['mutual_information'].items()]
        ranking.sort(key=lambda r: (-r['mi'], r['name']))
        projection = train[['met_TEMP', 'met_DEWP']].copy()
        projection['cluster'] = labels
        result = {'mode': '交互探索，独立于正式测试', 'model': model_id, 'horizon': horizon, 'train_rows': len(train), 'test_rows': len(test),
                  'cutoff': str(cutoff), 'max_train_target': str(train.target_time.max()), 'min_test_origin': str(test.origin_time.min()),
                  'scores': metrics(test.target, pred), 'common_scores': common_scores,
                  'feature_count': len(model.feature_names), 'ranking': safe_json(ranking), 'selection': safe_json(selection),
                  'regime': safe_json(report['regime']), 'projection': records(projection.iloc[::max(1, len(projection)//600)]),
                  'predictions': records(prediction.iloc[-240:]),
                  'forecast': {'origin': str(latest.origin_time.iloc[0]), 'target_time': str(latest.target_time.iloc[0]), 'value': future,
                               'distances': distances, 'input_missing_fraction': float(latest.input_missing_fraction.iloc[0])},
                  'config': cfg, 'source': self.source}
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out = self.root / 'reports/tool_runs' / run_id
        out.mkdir(parents=True)
        self.raw.to_csv(out / '01_imported.csv', index=False)
        self.base.to_csv(out / '02_cleaned.csv', index=False)
        frame.to_csv(out / '03_features.csv', index=False)
        pd.DataFrame(ranking).to_csv(out / '04_selection.csv', index=False)
        projection.to_csv(out / '05_training_clusters.csv', index=False)
        prediction.to_csv(out / '06_holdout_predictions.csv', index=False)
        pd.DataFrame(model.transform(latest), columns=model.feature_names).to_csv(out / '07_latest_model_input.csv', index=False)
        (out / 'run.json').write_text(json.dumps(safe_json(result), ensure_ascii=False, indent=2))
        (out / 'model_process.json').write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2))
        archive = out / '过程记录.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                if p != archive:
                    z.write(p, p.name)
        self.model, self.frame, self.export_path = model, frame, archive
        result['download'] = '/api/download'
        return safe_json(result)
