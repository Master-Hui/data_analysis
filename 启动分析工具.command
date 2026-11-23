#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo '未找到项目 Python 环境 .venv，请先按 README 安装依赖。'
  read 'reply?按回车退出'
  exit 1
fi
.venv/bin/python scripts/start_tool.py
