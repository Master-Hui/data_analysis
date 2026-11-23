#!/usr/bin/env python3
"""Loopback-only interactive tool; stdlib HTTP, no cloud services."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.process_tool import ProcessTool

engine = ProcessTool(ROOT)
lock = threading.Lock()
state = {'busy': False, 'logs': [], 'result': None, 'error': None}


def training(options):
    try:
        def progress(message):
            with lock:
                state['logs'].append(message)
        result = engine.train(options, progress)
        with lock:
            state['result'] = result
    except Exception as e:
        with lock:
            state['error'] = str(e)
    finally:
        with lock:
            state['busy'] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        host = self.headers.get('Host', '')
        expected = f'127.0.0.1:{self.server.server_port}'
        origin = self.headers.get('Origin')
        return host == expected and (origin is None or origin == 'http://' + expected)

    def do_GET(self):
        if not self.local_request():
            return self.reply({'error': '仅接受本地工具页面请求。'}, 403)
        if self.path == '/api/catalog':
            return self.reply(engine.catalog())
        if self.path == '/api/status':
            with lock:
                return self.reply(state)
        if self.path == '/api/download':
            with lock:
                path = engine.export_path if not state['busy'] else None
            if not path:
                return self.reply({'error': '请先完成一次训练。'}, 400)
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', 'attachment; filename="process_records.zip"')
        elif self.path == '/':
            data = (ROOT / 'tool_web/index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
        else:
            return self.reply({'error': '未找到页面。'}, 404)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.local_request() or self.headers.get('Content-Type') != 'application/json':
            return self.reply({'error': '请求来源或格式无效。'}, 403)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 9_000_000:
                raise ValueError('请求大小无效；CSV 最大 8 MB。')
            options = json.loads(self.rfile.read(size))
            with lock:
                if state['busy']:
                    return self.reply({'error': '训练进行中，请等待完成。'}, 409)
                if self.path == '/api/train':
                    state.update(busy=True, logs=[], result=None, error=None)
                    threading.Thread(target=training, args=(options,), daemon=True).start()
                    return self.reply({'started': True})
                if self.path == '/api/import':
                    result = engine.ingest(options)
                    state.update(logs=[], result=None, error=None)
                elif self.path == '/api/inspect':
                    result = engine.inspect(options)
                else:
                    return self.reply({'error': '未知操作。'}, 404)
            return self.reply(result)
        except Exception as e:
            return self.reply({'error': str(e)}, 400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'空气质量分析工具已启动：{url}\n关闭本窗口或按 Ctrl+C 停止。', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
