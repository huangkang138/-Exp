# POMO 训练入口与结果

## Hard / Soft / None 与五组对照

`train_config.toml` 仍读取 `benachmark_Train_MATSP/manifest.csv` 指向的 5000 个落盘训练实例；固定的 1000 个 validation 实例只用于选择 checkpoint。原 Hard `best.pt` 保留在 `gb_maco_POMO/checkpoints/run_20261001_180735_1c0d01/`，其记录为 50 epoch、batch 64、micro batch 4、学习率 1e-4、seed 42、128 维/3 层/8 头。新训练会创建独立 run，不覆盖它。

在项目根目录运行，按需要用 `--device cpu` 覆盖默认 CUDA：

```powershell
python -m gb_maco_POMO.experiments.train --check-data --device cpu
python -m gb_maco_POMO.experiments.train --candidate-mode hard --results-root gb_maco_POMO/results/hard --checkpoints-root gb_maco_POMO/results/checkpoints/hard
python -m gb_maco_POMO.experiments.train --candidate-mode soft --same-ball-bias 2 --adjacent-ball-bias 1 --results-root gb_maco_POMO/results/soft --checkpoints-root gb_maco_POMO/results/checkpoints/soft
python -m gb_maco_POMO.experiments.train --candidate-mode none --results-root gb_maco_POMO/results/none --checkpoints-root gb_maco_POMO/results/checkpoints/none
```

Hard 的训练命令仅用于重新训练对照；五组正式实验应指定上面的原 Hard checkpoint。新 Soft / None 模型各自从 `results/checkpoints/<mode>/<run-id>/best.pt` 取权重。训练时的 Soft Bias 与推理时由同一 `candidate_guidance()` 和 Decoder 计算；None 仍保留 5 维城市输入。

五组入口（PowerShell，自动读取各模式最近一个 `best.pt`；运行前核对选中的 run）：

```powershell
$soft = (Get-ChildItem gb_maco_POMO/results/checkpoints/soft/run_*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$none = (Get-ChildItem gb_maco_POMO/results/checkpoints/none/run_*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python -m gb_maco_POMO.experiments.ablation --group all --hard-checkpoint gb_maco_POMO/checkpoints/run_20261001_180735_1c0d01/best.pt --soft-checkpoint $soft --none-checkpoint $none --benchmark-dir D:\Project\球粒计算\GB-MACO_PPSN2026-main\PPSN_GB-MACO\benchmark_MSTSP --samples-per-start 8 --quality-threshold 0.05 --seed 42
```

可用 `--group A` 等单独运行，`--instances MSTSP1` 做单例验证。结果在 `gb_maco_POMO/results/ablation/<时间>/A..E/`，各组分别保存 `config.json`、`run.log`、`per_instance.csv`、`summary.json`，根目录保存五个指定组间差异 `comparisons.json`。`generated_routes` 是 N 或 N×K；`unique_routes` 按旋转及逆向闭环去重；`high_quality_routes` 是相对本次候选最佳距离不超过阈值的**不同闭环**数量。`.solution` 在路线生成完毕后才读取，仅计算 Gap、命中与 Fβ/DI。

当前工作区的 `GB-MACO_python/benchmark_MSTSP/` 处于删除状态。上述替代目录的 25 个 `.tsp` 和 25 个 `.solution` 已与本仓库 Git HEAD 中的原件逐文件 SHA256 对比，50 个文件均一致；正式实验可通过 `--benchmark-dir` 指向该目录。

## 多实例训练：默认入口

在项目根目录运行 `python -m gb_maco_POMO.experiments.train`，或在 PyCharm 中直接运行 `experiments/train.py`。参数主要修改 `experiments/train_config.toml`；其中默认 `epochs = 1` 仅用于检查完整训练流程，正式轮数需结合实际耗时和验证曲线确定。这里 1 个 epoch 会遍历全部 5000 个训练实例；当前清单与 `batch_size = 64` 下，每轮有 80 次参数更新，与旧单算例的“2000 epoch = 2000 次更新”不是同一个计数方式。

运行前可以用 `python -m gb_maco_POMO.experiments.train --check-data --device cuda` 核对 5000 个训练文件、1000 个验证文件和 GPU，不会开始训练。临时覆盖参数可用 `--epochs 10`、`--batch-size 32` 等命令行选项；优先级依次是 TOML、`GB_POMO_DATA_DIR` 环境变量、命令行。`CUDA_VISIBLE_DEVICES` 由 PyTorch 读取。

`batch_size` 是一次参数更新覆盖的实例数，同一批城市数相同；`micro_batch_size` 是一次实际进入 GPU 的实例数，代码会累积梯度。例如 64 和 4 表示最多用 16 个 micro batch 完成一次参数更新。验证集的 `batch_size` 同样按 micro batch 分段处理，不更新参数。验证总体均值用于比较同一固定验证集在不同 epoch 的变化；还应查看按城市数和分布拆开的结果。

每次新运行都会创建独立的 `gb_maco_POMO/results/run_xxx/` 和 `gb_maco_POMO/checkpoints/run_xxx/`。结果目录保存原始配置、实际生效配置、逐轮记录、按分布和城市数汇总的验证记录；checkpoint 目录每轮更新 `latest.pt`，验证改善时更新 `best.pt`，每隔 `save_interval` 轮再留一份 `epoch_xxxx.pt`。

中断后把 `checkpoint.resume` 填为该运行的 `latest.pt` 路径，并把 `training.epochs` 改成**目标总轮数**；也可以临时运行 `python -m gb_maco_POMO.experiments.train --resume <latest.pt路径> --epochs <目标总轮数>`。续训会恢复网络、Adam 和随机状态，继续写原来的 run 目录；最初的实际配置保留，续训配置另存为 `resume_config_epoch_xxxx.toml`。训练数据只来自 `train/`，`validation/` 只用于评估和选择最佳模型；原 `benchmark_MSTSP` 留待最终对比。

## 训练完成后的同算例对比

在项目根目录运行 `python -m gb_maco_POMO.experiments.compare`，也可在 PyCharm 中直接运行 `experiments/compare.py`。它默认读取最近一个已完成 run 的 `best.pt`；需要固定指定模型时用 `--checkpoint gb_maco_POMO/checkpoints/run_xxx/best.pt`。可用 `--instances MSTSP1 MSTSP25` 先检查指定算例。

脚本对原 `benchmark_MSTSP` 的 25 个 `.tsp` 逐例测试：加载训练权重，分别以每个城市为起点，后续每步选概率最高的候选城市。和原算法输出一致，只把本次最短长度的路线去重后作为返回解集。`Fβ`、`DI` 直接复用 `reproduction/scripts/calculate_metrics.py` 的公式及 `.solution` 真值集合。主对比基准是 `reproduction/config/paper_results.csv` 中的论文 Table 2 Fβ；原 GB-MACO 复现值仍从 `reproduction/results/summary/instance_summary.csv` 读取并保存在 CSV 供核查，不会重跑原算法。算例 `MSTSP25` 有 66 个城市，超出本次训练集最大 60 城，阅读结果时应单独注意。

默认写入对应 `gb_maco_POMO/results/run_xxx/comparison/`：`per_instance.csv` 保存论文、GB-MACO 复现、POMO 的 Fβ，以及路线长度、DI、最优解数量和耗时；`summary.json` 记录 checkpoint、来源与指标均值。`figures/` 按 `reproduction/results/figures/` 的中文图样式生成六张：POMO Fβ、论文与 POMO Fβ 对比、Fβ 差值、POMO DI、POMO 推理耗时、Fβ 绝对差值。论文 Fig.4 没有可用于精确计算的 DI 原始数值，因此不画“论文 DI vs POMO DI”。原算法汇总当前每例只有 1 次运行，POMO 为确定性推理；POMO 耗时图包含 GPU 推理和粒球预处理，不与原算法的历史耗时直接比较。

## 原 simple1_9 单算例实验

运行 `python -m gb_maco_POMO.experiments.train --single` 后，旧结果保存在 `experiments/results/simple1_9/`。下面这些字段解释针对这个旧单算例结果；多实例训练的验证字段为 `validation_mean_best_length`，详见各 run 的 `effective_config.toml`、`training_history.csv` 和 `validation_by_group.csv`。

## 先看哪个文件

| 文件 | 用途 |
| --- | --- |
| `training_history.csv` | 每轮训练的路线长度、loss 和候选集统计。 |
| `fallback_stats.csv` | 每条 POMO 路线在每一步的候选集与 fallback 情况。 |
| `final_result.json` | 整次实验的配置和汇总结果。 |
| `training_curve.png` | 路线长度随训练轮数变化的曲线。 |

## training_history.csv：一行代表一轮训练

| 列名 | 含义 | 怎么看 |
| --- | --- | --- |
| `epoch` | 第几轮参数更新，从 1 开始。 | 横轴。 |
| `train_mean_length` | 本轮采样得到的所有 POMO 路线的平均闭合路线长度。 | 越小越好。 |
| `train_best_length` | 本轮采样路线中最短的一条。 | 越小越好。 |
| `eval_best_length` | 定期评估时，确定性选点得到的 POMO 路线中最短的一条。 | 越小越好；非评估轮留空。 |
| `mean_reward` | 本轮平均奖励，等于 `-train_mean_length`。 | 数值越大，表示路线越短。 |
| `loss` | 用来更新网络参数的策略梯度目标值。 | 可以为负；不能当作路线长度，也不要求逐轮下降。 |
| `fallback_count` | 本轮有多少次决策因局部候选为空，退回到全部未访问城市。 | 用来检查候选范围是否经常不够用。 |
| `fallback_rate` | `fallback_count / 本轮决策总次数`。 | 越接近 0，触发 fallback 越少。 |
| `average_candidate_size` | 本轮每次决策可选城市数量的平均值。 | 反映候选集大小，不直接代表路线质量。 |

这里的路线长度来自项目的距离矩阵，是按坐标计算并取整的距离；没有定义为公里。训练列来自本轮采样路线，`eval_best_length` 来自参数更新后的定期评估，两者不要直接当作同一次选点的结果。

## fallback_stats.csv：一行代表一次选点决策

| 列名 | 含义 |
| --- | --- |
| `epoch` | 决策发生在哪轮训练。 |
| `pomo_index` | 第几条 POMO 路线，从 0 开始编号。 |
| `step` | 路线中的第几次选择下一城市；起点已经选好，所以从 1 开始。 |
| `current_city` | 作出选择前所在城市的编号。 |
| `candidate_size` | 本次允许选择的未访问城市数量。 |
| `fallback_used` | 是否因当前球及邻接球内没有未访问城市而使用全局候选。 |
| `remaining_city_count` | 作出选择前，整张图还剩多少未访问城市。 |

例如 `1,0,1,0,8,False,8` 表示：第 1 轮、第 0 条路线，在城市 0 作第一次后续选点；还剩 8 个城市，8 个都是候选，没有触发 fallback。

## final_result.json：整次实验的汇总

| 字段 | 含义 |
| --- | --- |
| `dataset`、`seed`、`epochs`、`eval_interval` | 数据集、随机种子、总训练轮数、每隔多少轮评估一次。 |
| `initial_eval_length` | 训练前评估的最短路线长度。 |
| `final_eval_length` | 全部训练结束后评估的最短路线长度。 |
| `best_eval_length` | 训练期间及结束后各次评估中的最短路线长度。 |
| `final_train_mean_length`、`final_train_best_length` | 最后一轮采样训练路线的平均长度与最短长度。 |
| `total_decisions` | 所有训练轮次累计的选点决策次数。 |
| `total_fallback_count`、`overall_fallback_rate` | 累计 fallback 次数与占全部决策的比例。 |
| `average_candidate_size` | 整次训练中每次决策的平均候选城市数。 |
| `fallback_by_remaining_city_count` | 以剩余城市数为键，统计各阶段触发 fallback 的次数。 |

`simple1_9.tsp` 只有一个粒球，所以候选城市始终是全部未访问城市；该算例的 `fallback_count = 0` 不能说明多粒球数据也不会触发 fallback。旧单算例实验只输出统计文件和曲线，没有保存模型参数；再次运行 `--single` 会重写对应结果目录。多实例训练则会保存模型 checkpoint，并为每次新运行建立独立目录。
