# 多维气象特征空气质量预测工具

本项目使用气象观测及历史 PM2.5 数据，预测未来 1、6、24 小时对应时刻的 PM2.5 浓度（μg/m³）。提供可交互的本地网页工具，逐步展示导入、清洗、历史窗口、特征筛选、气象聚类、模型训练和预测过程，并支持导出中间记录。

## 数据范围

仓库包含 UCI Beijing Multi-Site Air Quality 数据集的 12 个站点原始 CSV，共 420,768 条小时记录，位于 `datasets/raw/`。

**内置数据仅覆盖 2013-03-01 00:00 至 2017-02-28 23:00，不提供实时数据，也没有 2026 年数据。其他年份或站点的数据需要自行准备并通过工具导入 CSV。**

数据来源：[UCI 数据集主页](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)，[DOI: 10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G)。引用：Chen, S. (2017). Beijing Multi-Site Air Quality. UCI Machine Learning Repository。原始数据按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 提供，本仓库保留原始站点观测，未修改 CSV 内容。该许可说明仅针对数据集。

## 安装与启动

项目在 Python 3.14.4 环境中验证。请先安装 Python；建议使用 Python 3.14，并在项目目录建立独立环境。首次安装依赖需要联网；使用内置数据运行工具不需要连接外部服务。

```bash
git clone https://github.com/Master-Hui/data_analysis.git
cd data_analysis
python3 -m venv .venv
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/start_tool.py
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/start_tool.py
```

浏览器会自动打开 **http://127.0.0.1:8765**。如果没有自动打开，手动访问该地址。运行时保留终端窗口，按 `Ctrl+C` 停止服务。

macOS 完成首次环境安装后，也可以双击根目录中的 `启动分析工具.command`。如端口被占用，可改用 `python scripts/start_tool.py --port 8766`；浏览器地址随之改为 `http://127.0.0.1:8766`。不需要自动打开浏览器时添加 `--no-browser`。

`requirements.txt` 给出依赖范围；`requirements.lock.txt` 记录已有验证环境的精确版本。同平台和相同 Python 版本下，可用 `python -m pip install -r requirements.lock.txt` 复现这些依赖。

## 第一次使用

1. **导入并检查**：保留默认 Aotizhongxin 站点，日期选 2016-01-01 至 2016-03-01，点击“导入并检查”。结束日期不包含当天，该范围有 1440 条小时记录。
2. **查看清洗和历史窗口**：选择变量、3/6/12/24 小时窗口以及发起时刻所在行，点击“更新过程视图”。查看有效观测和填补后输入的对照曲线、窗口范围、逐小时记录及计算出的特征。行号从 0 开始，查看窗口至少需要 25 个小时。
3. **运行分析与训练**：首次保留 H1、提前量 6 小时、气象特征上限 40、气象状态数 4，点击“运行分析与训练”。页面显示处理进度、互信息排序、筛选理由、聚类二维投影及中心数值。
4. **查看预测并导出**：查看最新目标时刻的 PM2.5 浓度、历史留出段的真实值与预测曲线，以及 MAE、RMSE 和持久性基线比较。点击“下载本次完整过程记录 ZIP”保存本次分析记录。

修改站点或日期后需要重新导入；修改模型参数后需要重新训练。每次成功运行的记录自动保存至 `reports/tool_runs/时间戳/`，这些生成文件不会纳入版本控制。

“未来”相对于导入数据的最后观测时刻。例如默认数据最后时刻为 2016-02-29 23:00，提前量为 6 小时，最新预测目标就是 2016-03-01 05:00，并非现实中今天的空气质量。

## 导入自己的数据

每次导入一个站点的 CSV，文件不超过 8 MB、筛选后不超过 20,000 行，每次日期跨度不超过 366 天。选择文件后，将开始和结束日期改为该文件实际覆盖的日期，再点击导入。日期范围也会用于筛选上传文件。

| 字段 | 要求及单位 |
|---|---|
| `station` | 单个站点的名称 |
| `timestamp` | 北京时间整点，不带时区，例如 `2026-01-01 00:00:00`；也可用 `year`、`month`、`day`、`hour` 四列代替 |
| `TEMP`、`DEWP` | 气温、露点，℃ |
| `PRES` | 气压，hPa |
| `RAIN` | 降水量，mm |
| `WSPM` | 风速，m/s |
| `wd` | 风向，使用 N、NNE、NE、ENE、E、ESE、SE、SSE、S、SSW、SW、WSW、W、WNW、NW、NNW |
| `PM2.5` | PM2.5 浓度，μg/m³，用于训练目标；H1 也使用其已观测历史 |

缺失值可留空；其他污染物字段可以省略。真实预测目标不填补。相同站点、相同时刻存在冲突记录会报错；缺少的整小时会补为空值。训练至少需要 200 行有效训练样本和 30 行留出样本，数据不足时请扩大日期范围。

## 方法与模型

所有历史特征只使用预测发起时刻及此前的信息。输入按站点最多前向填补 3 小时；窗口统计使用未填补的有效观测，至少满足 75% 覆盖率（所需条数向上取整）才输出。构建风向循环编码、风矢量、滞后、窗口统计及差分，再在训练段内完成互信息与 Spearman 冗余筛选、气象聚类和回归拟合。

**M3、H1 是本项目自定义的方案编号，不是通用算法名称。**

| 方案 | 输入与用途 |
|---|---|
| M3 完整气象方案 | 筛选后的连续气象特征、质量指标、站点与日历编码，以及到各气象聚类中心的距离 |
| H1 融合方案 | 与 M3 同类的气象处理流程，另加入当前及历史 PM2.5、滞后、窗口、差分和质量指标 |

两种方案均使用 `HistGradientBoostingRegressor`（直方图梯度提升回归树），分别拟合各自的筛选器、聚类器和回归器。输出是未来单小时 PM2.5 浓度，不是未来区间均值，也不是 AQI 或空气质量等级。聚类图颜色表示气象状态，不表示污染等级；互信息也不表示因果关系或贡献率。

## 交互探索与正式实验

交互工具使用所选数据前 80% 时间训练、后 20% 时间留出检查，剔除跨分界的训练标签；仅在训练段拟合所有数据变换。固定种子 42、60 轮提升，互信息最多抽样 3000 行，不执行正式三折参数搜索。反复调整后的留出误差用于探索，不能代替独立正式测试。

项目另包含时间顺序验证、参数搜索、冻结测试、多模型对照和消融实验代码。以下命令需先激活环境，并在项目根目录执行。

先做单站流程检查：

```bash
python -m src.train --config configs/smoke.yaml --output reports/runs/my_smoke
```

该命令生成验证指标、预测、模型及图表；它是小规模流程检查，不是正式测试。输出目录必须是尚不存在的新目录。

完整复现实验（12 站，1/6/24 小时，三个任务并行，计算量明显大于交互工具）：

```bash
python scripts/run_formal_experiments.py
python scripts/summarize_formal_experiments.py
```

调度脚本会先完成全部时间交叉验证和参数冻结，再执行正式测试；摘要脚本在实验完成后汇总结果。它们使用固定运行标识 `formal_20261014`，重复执行时会拒绝覆盖已有实验或摘要；需要重跑时请使用新的仓库副本。正式模型、预测与指标生成在 `reports/runs/`，汇总生成在 `reports/formal_20261014_summary/`。

## 代码结构与检查

```text
src/                 数据读取、清洗、特征、筛选、聚类、训练、评价和预测
scripts/             工具启动、数据核验及完整实验调度和汇总
configs/             交互工具基础配置、单站验证和正式实验配置
tool_web/            本地网页界面
tests/               时间对齐、泄漏防护、预测和交互流程检查
datasets/raw/        12 个站点的原始数据
requirements.txt     依赖范围
requirements.lock.txt 已验证环境的依赖版本
启动分析工具.command  macOS 启动入口
```

运行自动检查与原始数据核验：

```bash
python -m pytest -q
python scripts/audit_dataset.py
```

本仓库仅提交运行项目所需源码、配置、测试、原始数据及本 README；不包含课程报告、个人阅读材料、虚拟环境、训练后的模型和历史运行结果。
