# GB-MACO 论文复现实验摘录

## 文献信息

- 题目：*Granular Ball Enhanced Multimodal Ant Colony Optimization for the Multi-Solution Traveling Salesman Problem*
- 作者：De-Gang Chen, Peng Wang, Shuyin Xia
- 收录：PPSN 2026, LNCS 16986, pp. 452-468（卷内版权年份标为 2027）
- DOI：`10.1007/978-3-032-36223-0_28`
- 本地核对文件：`C:\Users\admin\Downloads\978-3-032-36223-0_28.pdf`
- 核对日期：2026-09-23

本文档只整理复现实验直接需要的公式、参数和结果，不代替论文全文。

## 实验指标

### 路线相似度 Eq. (13)

对两条路线 `π_i`、`π_j`，令 `Φ(π)` 为闭合路线的无向边集合：

```text
S(π_i, π_j) = |Φ(π_i) ∩ Φ(π_j)| / |Φ(π_i)|
```

因此路线的起点和平移表示不影响相似度；对称 TSP 中反向路线具有相同边集合，也视为同一路线。

### Fβ Eq. (20)-(22)

```text
P = TP / (TP + FP)
R = TP / (TP + FN)
Fβ = (1 + β²)PR / (β²P + R)
β² = 0.3
```

在本工具链中：返回路线集合与 `.solution` 真值集合先按旋转、反向等价关系去重；交集为 TP，仅返回集合中存在的路线为 FP，仅真值集合中存在的路线为 FN。

### DI Eq. (23)

论文将 DI 描述为：对每条真值最优路线，寻找返回集合中与它最相似的路线，再进行归一化平均。结合 Eq. (13) 可写为：

```text
DI(S, T) = (1 / |T|) × Σ_i max_j S(T_i, S_j)
```

论文排版的 Eq. (23) 在分母中额外显示 `N`，但 Eq. (13) 已经除以路线边数 `N`。若再次除以 `N`，DI 最大值只能是 `1/N`，与 Fig. 4 的 `[0, 1]` 数值范围及大量 `1.00` 结果矛盾。因此实现采用等价的“共享边数量先除以 `N`，再对真值路线取平均”，即上式。该解释保留在代码注释中，避免静默掩盖论文公式的记号歧义。

## Table 1：论文参数

| 实例 | MaxFes | NP |
|---|---:|---:|
| MSTSP1-MSTSP12 | 60,000 | 150 |
| MSTSP13-MSTSP25 | 1,200,000 | 300 |

论文还写明自适应生态位大小 `msize ∈ [Mmin=2, Mmax=9]`。Eq. (14) 为：

```text
msize = Mmin + (Mmax - Mmin) × Fes / MaxFes
```

需要注意：公式随 Fes 增大而增大，但紧随其后的文字描述为“msize gradually decreases”，原文内部存在方向不一致。

## Table 2：GB-MACO Fβ

| 实例 | Fβ | 实例 | Fβ | 实例 | Fβ |
|---|---:|---|---:|---|---:|
| MSTSP1 | 1.00 | MSTSP10 | 1.00 | MSTSP19 | 0.99 |
| MSTSP2 | 1.00 | MSTSP11 | 1.00 | MSTSP20 | 0.91 |
| MSTSP3 | 1.00 | MSTSP12 | 1.00 | MSTSP21 | 0.98 |
| MSTSP4 | 1.00 | MSTSP13 | 1.00 | MSTSP22 | 0.28 |
| MSTSP5 | 1.00 | MSTSP14 | 1.00 | MSTSP23 | 0.75 |
| MSTSP6 | 1.00 | MSTSP15 | 1.00 | MSTSP24 | 0.57 |
| MSTSP7 | 1.00 | MSTSP16 | 1.00 | MSTSP25 | 0.48 |
| MSTSP8 | 1.00 | MSTSP17 | 1.00 |  |  |
| MSTSP9 | 1.00 | MSTSP18 | 0.98 |  |  |

这些数值已录入 `config/paper_results.csv` 的 `paper_fbeta`。Fig. 4 的 DI 仅以两位小数热力图展示，没有论文原始数值文件，因此 `paper_di` 保持为空，不把图中舍入显示值冒充原始实验数据。

## 论文与当前 Python 实现的关键差异

1. 论文对 MSTSP13-MSTSP25 使用 `NP=300`；Python `SearchConfig.ant_count` 对全部实例固定为 150。
2. 尚无论文或源码证据证明 `NP` 与 Python `ant_count` 完全同义，所以 `paper_setting` 保持 `unverified`，工具不会自动执行。
3. 论文的 `msize=2..9` 是单个生态位的大小；Python 当前从 2 增加到 9 的变量是 `group_count` / 信息素矩阵数量，语义不同。
4. 论文描述外部 archive 的“相似且更优时替换”策略；Python 的 `best_routes` 是无固定容量的边去重候选集合，维护规则并非论文文字的直接实现。
5. 论文未在实验章节说明独立运行次数；Python 入口的默认 30 次来自转换所依据的公开 C++ 源码配置，不应误写成论文 Table 1 的参数。

这些差异只记录，不在复现辅助工具中修改算法。
