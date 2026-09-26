# GB-MACO Python 复现实验工具链

本目录为 `GB-MACO_python` 的独立实验辅助层。它不修改 `gb_maco/` 中的粒球划分、蚁群搜索、多信息素、分组、2-opt 或候选解维护逻辑。

## 目录和文件分别做什么

| 文件 / 目录 | 作用 | 什么时候使用 |
|---|---|---|
| `config/experiment_config.json` | 保存 25 个实例映射、源码真实参数、论文参数状态、超时和输出路径 | 所有脚本都会自动读取，一般不需要手动修改 |
| `config/paper_results.csv` | 保存论文 Table 2 的 GB-MACO Fβ；没有可靠原始值的 DI 留空 | 画论文与复现结果对比图时读取 |
| `scripts/common.py` | 公共底层代码：读配置、定位目录、判断 run 是否成功、启动一次原算法并保存证据 | 由单实例和批量脚本调用，不需要单独运行 |
| `scripts/run_single.py` | 只运行一个指定实例，可运行一次或少量多次 | 先验证 MSTSP1，或单独排查某个实例 |
| `scripts/run_batch.py` | 顺序运行多个/全部实例，支持断点续跑、失败继续和批次汇总 | 正式执行 25×30 实验时使用 |
| `scripts/parse_results.py` | 把分散的 metadata、日志和 `.alg_solution` 解析成统一的 `all_runs.csv` | 算法运行完成后第一步执行 |
| `scripts/calculate_metrics.py` | 用 `.solution` ground truth 计算 Fβ、DI，并统计均值、标准差、最好/最差结果 | 解析完成后执行 |
| `scripts/plot_results.py` | 根据统计 CSV 和论文结果自动生成当前能生成的全部图片 | 指标统计完成后执行 |
| `paper_notes.md` | 整理论文公式、参数、Table 2 数据，以及论文与 Python 实现的差异 | 查公式、解释结果或继续复现时阅读 |
| `requirements.txt` | 仅列出绘图工具需要的 pandas 和 matplotlib | 新环境第一次运行前安装 |
| `results/raw/` | 每次 run 的原始 stdout、stderr、metadata 和算法输出 | 保留原始证据，不手工改结果 |
| `results/parsed/` | 每次实验一行的结构化结果 | 检查单次实验或供统计脚本读取 |
| `results/summary/` | 每个实例一行的均值、标准差、最好/最差值 | 查看最终统计结论 |
| `results/figures/` | 自动生成的 PNG 图片 | 用于论文、报告或 PPT |

## 1. 环境与入口

当前对象是 Python 转写项目，不需要编译 C++。项目要求 Python 3.10+；核心算法只使用标准库，绘图额外安装：

```powershell
python -m pip install -r reproduction/requirements.txt
```

真实入口是 `python -m gb_maco.runner`。执行链为：

1. `runner.py:73-80` 读取坐标、生成整数距离矩阵、调用 `gbc` 构造粒球、建立城市到粒球映射并生成贪心初始路线。
2. `aco.py:189-221` 根据贪心路线计算 `tau0`，建立两张初始信息素矩阵并强化同球边。
3. `aco.py:445-491` 反复构造蚂蚁路线、维护候选集合、执行粒球/关键边约束 2-opt、重组生态位、更新和蒸发信息素，并逐步增加信息素矩阵。
4. `runner.py:83-99` 将当前最优长度对应的不同路线写成 `.alg_solution`。

## 2. Benchmark 映射

| MSTSP | 实际文件名 | 城市数 | 类型 |
|---|---|---:|---|
| MSTSP1 | simple1_9.tsp | 9 | Simple |
| MSTSP2 | simple2_10.tsp | 10 | Simple |
| MSTSP3 | simple3_10.tsp | 10 | Simple |
| MSTSP4 | simple4_11.tsp | 11 | Simple |
| MSTSP5 | simple5_12.tsp | 12 | Simple |
| MSTSP6 | simple6_12.tsp | 12 | Simple |
| MSTSP7 | geometry1_10.tsp | 10 | Geometry |
| MSTSP8 | geometry2_12.tsp | 12 | Geometry |
| MSTSP9 | geometry3_10.tsp | 10 | Geometry |
| MSTSP10 | geometry4_10.tsp | 10 | Geometry |
| MSTSP11 | geometry5_10.tsp | 10 | Geometry |
| MSTSP12 | geometry6_15.tsp | 15 | Geometry |
| MSTSP13 | composite1_28.tsp | 28 | Composite |
| MSTSP14 | composite2_34.tsp | 34 | Composite |
| MSTSP15 | composite3_22.tsp | 22 | Composite |
| MSTSP16 | composite4_33.tsp | 33 | Composite |
| MSTSP17 | composite5_35.tsp | 35 | Composite |
| MSTSP18 | composite6_39.tsp | 39 | Composite |
| MSTSP19 | composite7_42.tsp | 42 | Composite |
| MSTSP20 | composite8_45.tsp | 45 | Composite |
| MSTSP21 | composite9_48.tsp | 48 | Composite |
| MSTSP22 | composite10_55.tsp | 55 | Composite |
| MSTSP23 | composite11_59.tsp | 59 | Composite |
| MSTSP24 | composite12_60.tsp | 60 | Composite |
| MSTSP25 | composite13_66.tsp | 66 | Composite |

每个 `.tsp` 都有同名 `.solution`，其中包含 ground-truth 最优闭合路线集合。

## 3. 参数核对

| 参数 | 论文值 | Python 源码实际值 | 一致性 | 代码位置 | 真实含义 / 备注 |
|---|---|---|---|---|---|
| MaxFes | 1-12: 60000；13-25: 1200000 | 相同 | 是 | `runner.py:13-38` | 最大适应度评价预算 |
| NP | 1-12: 150；13-25: 300 | 无 `NP` 变量 | 未验证 | - | 禁止默认等于 antNum |
| antNum | 论文未单列 | 150（全部实例） | 无法直接比较 | `aco.py:18` | 每轮蚂蚁数量 |
| alpha | 未在 Table 1 给值 | 1.0 | 未核对 | `aco.py:19` | 信息素权重 |
| beta | 指标中 β²=0.3；ACO β 未在 Table 1 给值 | 1.5 | 未核对 | `aco.py:20` | 启发式距离权重；与 Fβ 的 β 不是同一参数 |
| rho | 未在 Table 1 给值 | 0.3 | 未核对 | `aco.py:21` | 信息素蒸发率 |
| 信息素更新比例 | 未明确给值 | 0.3 | 未核对 | `aco.py:23,429-443` | 每组排名靠前约 30% 的蚂蚁参与强化 |
| tau0 | 未明确给值 | antNum / 贪心路线长度 | 未核对 | `aco.py:216` | 初始非对角信息素 |
| 信息素下限 | 未明确给值 | 1e-64 | 未核对 | `aco.py:409-416` | 蒸发后的下限 |
| 初始路线强化 | 未明确给值 | 2.0 | 未核对 | `aco.py:232-261` | 同一粒球内连续边的额外强化 |
| Mmin / Mmax | 2 / 9 | 没有对应变量 | 否 | - | 论文用于生态位大小 |
| msize | Eq.(14) 自适应 | 未实现同义变量 | 否 | - | Python 的 2→9 是组/矩阵数量，不是生态位大小 |
| 信息素矩阵数 | 未作为 Table 1 参数 | 2 起步，最多 9 | 无法直接比较 | `aco.py:22,449-487` | 每个组关联一张矩阵 |
| 运行次数 | 论文实验章节未明示 | 30 | 未核对 | `runner.py:52` | Python 默认值继承公开源码 |
| 随机种子 | 未说明 | 默认按秒初始化两个随机流；可显式覆盖 | 未核对 | `aco.py:45-61` | 元数据中 `null` 表示真实种子不可观测 |
| 2-opt | 粒球辅助局部搜索 | 无独立配置项 | 部分对应 | `aco.py:329-362` | 关键边、跨球边及 0.1 增益阈值共同筛选 |
| archive | 外部 archive | `best_routes`，无固定容量 | 实现语义不同 | `aco.py:307-327` | 按共享边去重并替换/追加候选路线 |

详细论文摘录与公式解释见 `paper_notes.md`。

## 4. 配置档位

- `source_original`：可运行，使用当前 Python 真实参数。
- `paper_setting`：仅记录论文参数，状态为 `unverified`。因为 `NP→ant_count` 与论文 `msize` 尚未在 Python 中建立等价实现，运行工具会拒绝该档位，防止生成错误标注的“论文参数结果”。

## 5. 执行命令

在项目根目录运行。

单实例一次：

```powershell
python reproduction/scripts/run_single.py --instance MSTSP1 --runs 1
```

需要完全可追踪的显式种子时：

```powershell
python reproduction/scripts/run_single.py --instance MSTSP1 --runs 1 --seed-start 7
```

批量实验（当前工具已准备好，本轮不要执行）：

```powershell
python reproduction/scripts/run_batch.py --instances all --runs 30 --seed-start 1
```

选择部分实例：

```powershell
python reproduction/scripts/run_batch.py --instances MSTSP1 MSTSP2 MSTSP3 --runs 5 --seed-start 1
```

批量工具顺序执行；成功 run 自动跳过，失败只记录并继续，最终写出 `results/raw/batch_summary.json`。不传 `--seed-start` 时保留 runner 的时间种子行为，但实际种子无法从现有核心代码读取，元数据会诚实记录为 `null`。

解析、统计、绘图：

```powershell
python reproduction/scripts/parse_results.py
python reproduction/scripts/calculate_metrics.py
python reproduction/scripts/plot_results.py
```

## 6. 结果结构

```text
reproduction/results/
  raw/MSTSP1/run_001/
    stdout.txt
    stderr.txt
    metadata.json
    program_output/*.alg_solution
  parsed/all_runs.csv
  summary/instance_summary.csv
  figures/*.png
```

`metadata.json` 保存实例、文件、UTC 时间、参数、seed、Python 可执行文件、完整命令、耗时、超时状态、退出码和输出文件。每个 run 独立保存。

## 7. Fβ 与 DI 的计算边界

当前 25 个 `.solution` 提供完整 ground truth，因此 Fβ 和 DI 均可从真实输出计算。路线在比较前按闭合回路的旋转与反向等价关系规范化；不会把相同路线的不同起点或方向重复计数。

- Fβ 严格使用 Eq.(20)-(22)，`β²=0.3`。
- DI 使用 Eq.(13) 的无向边相似度，并按 Eq.(23) 对每条真值路线取返回集合中的最大相似度后求平均。
- `.alg_solution` 缺失或运行失败时不伪造指标。
- Table 2 的论文 Fβ 已录入 `paper_results.csv`。
- Fig. 4 只有舍入后的 DI 热力图，没有原始数据，因此 `paper_di` 保持为空，论文 vs 复现 DI 图会自动跳过。

## 8. 绘图

`plot_results.py` 使用统一英文标题、200+ dpi、无 GUI 后端并自动 `tight_layout`。数据存在时生成：

- `fbeta_reproduced.png`
- `fbeta_paper_vs_reproduced.png`
- `fbeta_difference.png`
- `di_reproduced.png`
- `di_paper_vs_reproduced.png`（当前因 paper_di 缺失而跳过）
- `runtime_by_instance.png`
- `reproduction_error_overview.png`

只有一个实例时也能生成单实例验证图；缺失数据对应的图会提示并跳过，不会导致整批绘图失败。
