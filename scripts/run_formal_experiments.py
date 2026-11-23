"""Run independent horizons concurrently; test only after all validations finish."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
RUN_TAG = 'formal_20261014'


def run_stage(stage, horizon):
    output = ROOT / 'reports/runs' / f'{RUN_TAG}_{stage}_h{horizon}'
    logfile = ROOT / 'reports/logs' / f'{RUN_TAG}_{stage}_h{horizon}.log'
    command = [sys.executable, '-m', 'src.train']
    if stage == 'validation':
        command += ['--config', f'configs/formal_h{horizon}.yaml', '--tune']
    else:
        frozen = ROOT / 'reports/runs' / f'{RUN_TAG}_validation_h{horizon}/frozen_experiment.json'
        command += ['--stage', 'final', '--frozen', str(frozen)]
    command += ['--output', str(output)]
    environment = dict(os.environ, LOKY_MAX_CPU_COUNT='3', PYTHONUNBUFFERED='1')
    print(f'{stage} h={horizon} started: {output}', flush=True)
    with logfile.open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    if not (output / 'completed.json').exists():
        raise RuntimeError(f'Missing completion marker: {output}')
    print(f'{stage} h={horizon} completed', flush=True)
    return str(output)


def main():
    status_path = ROOT / 'reports' / f'{RUN_TAG}_status.json'
    existing = [ROOT / 'reports/runs' / f'{RUN_TAG}_{stage}_h{h}'
                for stage in ['validation', 'test'] for h in [1, 6, 24]]
    if status_path.exists() or any(path.exists() for path in existing):
        raise FileExistsError('Experiment tag already exists; preserve it and use a new tag/project copy.')
    (ROOT / 'reports/logs').mkdir(parents=True, exist_ok=True)
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'horizons': [1, 6, 24],
              'workers': 3, 'threads_per_worker': 3, 'validation': {}, 'test': {}}
    status_path = ROOT / 'reports' / f'{RUN_TAG}_status.json'

    def save():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        for stage in ['validation', 'test']:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(run_stage, stage, h): h for h in status['horizons']}
                for future in as_completed(futures):
                    h = futures[future]
                    status[stage][str(h)] = future.result()
                    save()
            if stage == 'validation':
                status['all_parameters_frozen_utc'] = datetime.now(timezone.utc).isoformat()
                status['frozen_sha256'] = {}
                for h, folder in status['validation'].items():
                    status['frozen_sha256'][h] = hashlib.sha256(
                        (Path(folder) / 'frozen_experiment.json').read_bytes()).hexdigest()
                save()
        status['completed_utc'] = datetime.now(timezone.utc).isoformat()
        status['status'] = 'complete'
    except Exception as error:
        status['status'], status['error'] = 'failed', str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
