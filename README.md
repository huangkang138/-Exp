# GB-MACO Python 转换

本项目按原 C++ 源码的运行顺序实现：粒球生成、贪心初始路径、多信息素蚁群搜索、组匹配、2-opt 局部搜索和结果输出。仅使用 Python 标准库。

函数注释中的 `main.cpp:行号`、`GranularBall.cpp:行号` 等，均指向原项目 `D:\Project\球粒计算\GB-MACO_PPSN2026-main\PPSN_GB-MACO\` 中的文件。

## 文件

- `gb_maco/granular_ball.py`：粒球数据结构、分裂、规范化和重叠分组。
- `gb_maco/tsp.py`：城市与回路、坐标读取、整数距离、粒球映射和贪心路径。
- `gb_maco/aco.py`：蚁群搜索、信息素更新、分组和 2-opt。
- `gb_maco/runner.py`：命令行入口，输出 `.alg_solution`。

## 运行

在 PyCharm 的项目终端执行，直接读取本项目中的数据集：

```powershell
python -m gb_maco.runner benchmark_MSTSP/simple1_9.tsp --runs 1 --max-fes 60000 --seed 7
```

结果默认写到本项目的 `results/`。`--runs 30` 对应源码的独立运行次数；源码中 simple/geometry 算例的评价预算为 60000，composite 算例为 1200000。较大预算在纯 Python 中会耗费更多时间。

## 与 C++ 源码的差异

- 每个 Python 类、函数和方法的注释均标明原 C++ 文件、实现起始行及名称。Python 的下划线命名与 C++ 的驼峰命名不同；一个 Python 方法对应多个 C++ 函数时会一并列出。标为“Python 辅助”的函数没有独立的 C++ 同名实现。
- 仅转换原 `main()` 实际调用的算法路径。`Head.h` 中只有声明、没有实现或未被主流程调用的函数，不算已逐一转换。
- Python 使用自己的随机数生成器，给定相同种子也不会与 Boost 产生相同的逐步路径。
- 原代码把 `std::min_element` 与反向比较器组合使用，实际选到最长候选路径；Python 版按实际行为保留，见 `aco.py` 的 `_remember_route`。
- 原代码首次进入 2-opt 时的 `keyEdges` 数组尚未初始化；Python 版先用空关键边列表，避免读取未初始化内存。后续关键边按原哈希槽位及出现次数排序。
- Python 的 `float` 与 C++ 的 `float FES` 精度不同，排序中相同长度路径的先后顺序也可能不同；因此不保证逐步搜索轨迹一致。
- 命令行默认只运行一个数据文件、一次实验；原 `main()` 遍历 25 个文件且每个运行 30 次。需要多次实验时显式设置 `--runs`，composite 算例还需设置 `--max-fes 1200000`。
- 输出目录由命令行指定，不改写原始 `.tsp` 或 `.solution` 文件。
