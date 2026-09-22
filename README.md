# GB-MACO Python 转换

本项目按原 C++ 源码的运行顺序实现：粒球生成、贪心初始路径、多信息素蚁群搜索、组匹配、2-opt 局部搜索和结果输出。仅使用 Python 标准库。

函数注释中的 `main.cpp:行号`、`GranularBall.cpp:行号` 等，均指向原项目 `D:\Project\球粒计算\GB-MACO_PPSN2026-main\PPSN_GB-MACO\` 中的文件。

## 文件

- `gb_maco/granular_ball.py`：粒球数据结构、分裂、规范化和重叠分组。
- `gb_maco/tsp.py`：城市与回路、坐标读取、整数距离、粒球映射和贪心路径。
- `gb_maco/aco.py`：蚁群搜索、信息素更新、分组和 2-opt。
- `gb_maco/runner.py`：命令行入口，输出 `.alg_solution`。

## 运行

在 PyCharm 的项目终端执行，按原 `main.cpp` 遍历 25 个算例，每个运行 30 次：

```powershell
python -m gb_maco.runner
```

默认读取本项目的 `benchmark_MSTSP/`，结果写入原 C++ 同名目录 `Results_1_1.5 _0.3 150/`。原源码对 simple/geometry 算例使用 60000 FES，对 composite 算例使用 1200000 FES；`Head.h` 的 `antNum=150` 对所有算例都生效。完整实验在纯 Python 中耗时较长。

只检查单个算例时可执行：

```powershell
python -m gb_maco.runner benchmark_MSTSP/simple1_9.tsp --runs 1 --seed 7 --output-dir results
```

`--runs`、`--max-fes`、`--seed` 和 `--output-dir` 只用于单独检查或控制输出。覆盖源码实验参数后，结果不能标为原始参数实验。

## 与 C++ 源码的差异

- 每个 Python 类、函数和方法的注释均标明原 C++ 文件、实现起始行及名称。Python 的下划线命名与 C++ 的驼峰命名不同；一个 Python 方法对应多个 C++ 函数时会一并列出。标为“Python 辅助”的函数没有独立的 C++ 同名实现。
- 仅转换原 `main()` 实际调用的算法路径。`Head.h` 中只有声明、没有实现或未被主流程调用的函数，不算已逐一转换。
- 已按 `main.cpp` 分开维护 `random_int` 与 `random_probability` 的随机数流，并在默认实验的各次运行之间延续状态；Python 的 `Random` 与 Boost 的具体序列不同，给定相同种子也不会产生相同的逐步路径。
- 原代码把 `std::min_element` 与反向比较器组合使用，实际选到最长候选路径；Python 版按实际行为保留，见 `aco.py` 的 `_remember_route`。
- 原代码首次进入 2-opt 时的 `keyEdges` 数组尚未初始化；Python 版先用空关键边列表，避免读取未初始化内存。后续关键边按原哈希槽位及出现次数排序。
- FES 的加法、阈值计算已按 C++ 的单精度 `float` 舍入。Python 稳定排序与 C++ `std::sort` 在相同长度路径上的顺序仍可能不同，因此不保证逐步搜索轨迹一致。
- 命令行默认按原 `main()` 遍历 25 个文件，每个运行 30 次；默认输出目录名和 `.alg_solution` 的制表符格式也按源码设置。
- 原 `main.cpp` 中的 `tuple_n` 是组数及信息素矩阵数，从 2 增至 9；实际调用的是 `bubbleSortAndAccumulatePheromone`。贪心选点采用跨球惩罚，初始路径的同球边另有信息素强化。这里按源码保留，即使论文中的术语或参数表述不同。
- 不改写原始 `.tsp` 或 `.solution` 文件。

## 参数
先解释你那条命令。它要在 `D:\Project\球粒计算\GB-MACO_python` 的项目终端运行：

```powershell
python -m gb_maco.runner benchmark_MSTSP/simple1_9.tsp --runs 1 --seed 7 --output-dir results
```

| 部分 | 意思 |
|---|---|
| `python -m gb_maco.runner` | 运行 Python 项目的实验入口 |
| `benchmark_MSTSP/simple1_9.tsp` | 只用这个 9 城市算例 |
| `--runs 1` | 只运行 1 次 |
| `--seed 7` | 固定 Python 随机种子，方便重复检查 |
| `--output-dir results` | 把结果写入 `results` 文件夹 |

没写 `--max-fes`，所以这个算例自动用 **60000 FES**。这条命令是**单例检查**，不是 C++ 的完整实验：原程序每个算例跑 30 次，而且没有固定种子参数。

### 与原 C++ 对齐的参数

| 参数 | 原 C++ 的值 |
|---|---:|
| 蚂蚁数 `antNum` | **150，Composite 也是 150** |
| 信息素权重 `alpha` | 1.0 |
| 距离权重 `beta` | 1.5 |
| 蒸发率 `rho` | 0.3 |
| 每组参与信息素更新的比例 | 0.3 |
| 信息素矩阵／组数 | 从 2 开始，最多 9 |
| 每个算例运行次数 `RUNS` | 30 |
| simple、geometry 的 `MaxFES` | 60000 |
| composite 的 `MaxFES` | 1200000 |

这些值来自原 [Head.h](D:/Project/球粒计算/GB-MACO_PPSN2026-main/PPSN_GB-MACO/Head.h:32) 和 [main.cpp](D:/Project/球粒计算/GB-MACO_PPSN2026-main/PPSN_GB-MACO/main.cpp:994)。另外，源码内部固定使用跨球惩罚系数 `0.8`、初始路径信息素强化系数 `2.0`；它们不是命令行参数。

### 具体怎么验

1. **和数据集答案比。** `.tsp` 是城市坐标输入，`.solution` 是参考回路。检查 Python 输出是否每个城市恰好走一次、按源码距离规则重算的长度是否一致，再比较最优长度和回路。回路允许换起点或反向，不能直接逐字比文件。我已对 `simple1_9` 做过这一步：完整 60000 FES 的 3 条最优回路与参考答案对应。

2. **和原 C++ 实验比。** 在两边使用同一份 `.tsp`、上表参数和每例 30 次运行，比较每次最优长度、达到参考最优值的次数，以及找到的不同最优回路数量。Python 单例跑 30 次的命令是：
   ```powershell
   python -m gb_maco.runner benchmark_MSTSP/simple1_9.tsp --runs 30 --output-dir results
   ```
   **这里不要加 `--seed 7`**：C++ 原程序用时间初始化随机数，没有对应的固定种子选项。两边随机序列不同，因此验证的是解的有效性和多次运行的结果表现，不能要求第 1 次对第 1 次的路线完全相同。

只执行 `python -m gb_maco.runner` 才是 Python 当前按原 `main()` 设置跑 **25 个算例 × 每例 30 次**；这个完整对照目前还没有执行。