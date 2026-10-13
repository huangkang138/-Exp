"""GB-MACO 的蚁群搜索、信息素更新和粒球约束 2-opt。"""

from dataclasses import dataclass
from random import Random
from struct import pack, unpack
from time import time

from .candidate import (
    ADJACENCY_RULES,
    CANDIDATE_MODES,
    build_candidate_masks,
    mask_summary,
)
from .granular_ball import GranularBall
from .tsp import City, Route, path_len


@dataclass
class SearchConfig:
    """集中保存 main.cpp 中影响搜索的实验参数。
    原 C++ 参数：Head.h:32-37；main.cpp:994 MaxFES。
    """

    ant_count: int = 150
    alpha: float = 1.0
    beta: float = 1.5
    rho: float = 0.3
    max_pheromones: int = 9
    update_percentage: float = 0.3
    max_fes: int = 60_000
    seed: int | None = None

    # ------------------------------------------------------------------
    # 以下两组参数对应学长的两项改进，默认全部关闭，
    # 这样不传参数时 GBMACO 的行为与未改动前逐位一致（可直接复现原论文结果）。
    # ------------------------------------------------------------------

    # 改进一：候选城市判定（见 candidate.py）
    candidate_mode: str = "off"      # off / hard_ball / hard_neighbor / soft_neighbor
    candidate_rule: str = "knn"      # "临近粒球"怎么判：gap / knn / mst
    neighbor_margin: float = 1.0     # 仅 gap 口径使用
    neighbor_k: int = 1              # 仅 knn 口径使用：每个球连最近的 k 个球
    outside_scale: float = 0.05      # soft 模式下，候选集外城市的权重折扣
    anneal_fes: float = 0.0          # >0 时折扣在前这段时间里由 0 线性放开

    # 改进二：保留高质量路线（学长："这个高质量的路线要保留"）
    # 搜索中不改主流程，只把"接近当前最优"的路线额外抄进一个存档（route_archive），
    # 搜索结束后统一对存档做一次 2-opt 精修，把近似解打磨成精确解。
    keep_tolerance: float = 1.0      # 存档门槛：只收 length <= 当前最优 * 该值 的路线
    queue_limit: int = 0             # 存档上限（0 = 用默认值 300）
    final_polish: bool = False       # 搜索结束后是否对存档做收尾精修
    polish_relax: bool = True        # 收尾精修时是否关掉"关键边保护"（推荐开）
    polish_rotations: int = 1        # 每条路线换几个起点各扫一遍 2-opt（见 _polish_archive）
    polish_rounds: int = 3           # 收尾精修最多扫几遍（没有再改进就提前停）


@dataclass
class Ant:
    """一只蚂蚁本轮已形成的路径和城市访问状态。
    原 C++ 结构：Head.h:92 Ant。
    """

    route: Route
    visited: list[bool]


def _float32(value: float) -> float:
    """按 C++ 的单精度 float 舍入 FES 运算结果。
    原 C++：main.cpp:1127 的 float FES，main.cpp:173、1159、1170 的运算。
    """
    return unpack("f", pack("f", value))[0]


_INT_RANDOM: Random | None = None
_REAL_RANDOM: Random | None = None


def _source_randoms(seed: int | None) -> tuple[Random, Random]:
    """保留 C++ 两个独立的静态随机数流，并跨实验运行延续状态。
    原 C++：main.cpp:261 random_probability、267 random_int。
    Python 的 Random 与 Boost 的 mt19937 种子展开和分布实现仍不同。
    """
    global _INT_RANDOM, _REAL_RANDOM
    if seed is not None:
        return Random(seed), Random(seed)
    if _INT_RANDOM is None:
        _INT_RANDOM = Random(int(time()))
    if _REAL_RANDOM is None:
        _REAL_RANDOM = Random(int(time()))
    return _INT_RANDOM, _REAL_RANDOM


def _copy_route(route: Route) -> Route:
    """复制路径，避免后续 2-opt 原地修改已保存的候选解。
    Python 辅助：对应 Head.h:73 Route 拷贝构造与赋值语义。
    """
    return Route(route.city_path.copy(), route.length)


def _edges(path: list[int]) -> set[tuple[int, int]]:
    """把闭合回路转换成无向边集合，起点或行进方向不影响比较。
    Python 辅助：替代 main.cpp:857-885 的边哈希辅助逻辑。
    """
    return {
        tuple(sorted((path[i], path[(i + 1) % len(path)])))
        for i in range(len(path))
    }


def _signature(path: list[int]) -> frozenset[tuple[int, int]]:
    """回路的"边集合指纹"，用来做 O(1) 去重。

    为什么不直接比城市序列：同一个回路可以任意旋转起点、任意反向书写，
    比序列会把同一个解反复当成新解。边集合正好抹掉这两种等价。
    """
    return frozenset(_edges(path))


# 改进二的存档默认上限：既够装下几百条高质量路线，又不会把内存和时间吃满。
DEFAULT_ARCHIVE_LIMIT = 300


def edge_similarity(first: list[int], second: list[int]) -> float:
    """按源码计算共享边比例，供去重和蚂蚁分组使用。
    原 C++：main.cpp:885 jaccardEdgeSimilarity。
    """
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(_edges(first) & _edges(second)) / len(first)


def _group_sizes(ant_count: int, group_count: int) -> list[int]:
    """将蚂蚁尽量均匀地分给当前信息素矩阵。
    原 C++：main.cpp:275 distributeAnts。
    """
    base, remainder = divmod(ant_count, group_count)
    return [base + (i < remainder) for i in range(group_count)]


def _assignments(sizes: list[int]) -> list[int]:
    """将组大小展开成逐只蚂蚁对应的信息素矩阵编号。
    原 C++：main.cpp:519 initialize_group_assignments。
    """
    return [group for group, size in enumerate(sizes) for _ in range(size)]


def _sort_within_groups(ants: list[Ant], sizes: list[int]) -> None:
    """每组按回路长度排序，使组首成为该组的当前最优蚂蚁。
    原 C++：main.cpp:218 SortGroupAnts。
    """
    start = 0
    for size in sizes:
        ants[start : start + size] = sorted(
            ants[start : start + size], key=lambda ant: ant.route.length
        )
        start += size


def _form_groups(ants: list[Ant], sizes: list[int]) -> None:
    """先按质量选组长，再按与组长的边相似度吸收剩余蚂蚁。
    原 C++：main.cpp:229 formAntGroups。
    """
    ants.sort(key=lambda ant: ant.route.length)
    assigned = 0
    for size in sizes:
        if size <= 0 or assigned >= len(ants):
            continue
        leader = ants[assigned].route.city_path
        ants[assigned + 1 :] = sorted(
            ants[assigned + 1 :],
            key=lambda ant: edge_similarity(leader, ant.route.city_path),
            reverse=True,
        )
        assigned += min(size, len(ants) - assigned)
        ants[assigned:] = sorted(ants[assigned:], key=lambda ant: ant.route.length)


def _match_groups(
    ants: list[Ant], previous_best: list[Route], sizes: list[int],
    previous_least_similar: int,
) -> tuple[list[int], int]:
    """将当前组与上一轮信息素组做一对一相似度匹配。
    原 C++：main.cpp:676 Update_pheromoneMatrices。
    """
    group_count = len(sizes)
    similarity: list[list[float]] = []
    start = 0
    for size in sizes:
        leader = ants[start].route.city_path
        similarity.append(
            [edge_similarity(leader, old.city_path) for old in previous_best]
        )
        # 与源码一致：比较完当前组与所有旧组后，才推进一次组起点。
        start += size

    matched_current: set[int] = set()
    matched_previous: set[int] = set()
    mapping = [-1] * group_count
    lowest_similarity = 1.0
    # C++ 通过引用传入该编号；本轮没有更低相似度时沿用上轮结果。
    least_similar_previous = previous_least_similar
    for _ in range(group_count):
        best_pair = max(
            (
                (similarity[i][j], i, j)
                for i in range(group_count)
                if i not in matched_current
                for j in range(group_count)
                if j not in matched_previous
            ),
            key=lambda item: item[0],
        )
        score, current, previous = best_pair
        mapping[current] = previous
        matched_current.add(current)
        matched_previous.add(previous)
        if score < lowest_similarity:
            lowest_similarity = score
            least_similar_previous = previous
    assignments = [mapping[group] for group, size in enumerate(sizes) for _ in range(size)]
    return assignments, least_similar_previous


class GBMACO:
    """保存一次搜索的矩阵与路径状态，避免使用 C++ 原版的全局变量。
    Python 状态封装：原 C++ main.cpp 使用全局变量，主流程从 main.cpp:990 开始。
    """

    def __init__(
        self,
        cities: list[City],
        distance_matrix: list[list[int]],
        city_to_ball: list[int],
        granular_balls: list[GranularBall],
        initial_route: Route,
        config: SearchConfig | None = None,
    ) -> None:
        """初始化搜索参数、随机数、信息素矩阵和候选路径。
        原 C++ 代码段：main.cpp:1090-1123 初始化搜索状态；非同名函数。
        """
        self.config = config or SearchConfig()
        if not cities or initial_route.length <= 0:
            raise ValueError("城市列表不能为空，初始回路长度必须大于 0")
        if self.config.ant_count < 2 or self.config.max_fes < 1:
            raise ValueError("蚂蚁数至少为 2，最大评价次数必须大于 0")
        if self.config.max_pheromones < 2:
            raise ValueError("信息素矩阵上限至少为 2")
        if self.config.max_pheromones > self.config.ant_count:
            raise ValueError("信息素矩阵数量不能超过蚂蚁数量")
        if self.config.candidate_mode != "off":
            if self.config.candidate_mode not in CANDIDATE_MODES:
                raise ValueError(
                    f"未知的候选城市口径 {self.config.candidate_mode!r}，可选 {CANDIDATE_MODES}"
                )
            if self.config.candidate_rule not in ADJACENCY_RULES:
                raise ValueError(
                    f"未知的临近判定口径 {self.config.candidate_rule!r}，可选 {ADJACENCY_RULES}"
                )
            if self.config.neighbor_k < 1:
                raise ValueError("knn 口径下 neighbor_k 至少为 1")
            if not 0.0 <= self.config.outside_scale <= 1.0:
                raise ValueError("outside_scale 必须落在 0 到 1 之间")
            if self.config.anneal_fes and self.config.anneal_fes < 0:
                raise ValueError("anneal_fes 不能为负")
        if self.config.keep_tolerance < 1.0:
            raise ValueError("keep_tolerance 不能小于 1.0（否则会把最优解本身也挡在门外）")
        self.cities = cities
        self.distance = distance_matrix
        self.city_to_ball = city_to_ball
        self.balls = granular_balls
        self.count = len(cities)
        self.start_random, self.selection_random = _source_randoms(self.config.seed)
        self.tau_0 = self.config.ant_count / initial_route.length
        self.matrices = [self._new_matrix(), self._new_matrix()]
        self._boost_initial_route(initial_route)
        self.best_routes: list[Route] = []
        self.key_edges: list[tuple[int, int]] = []
        self.fes = 0.0

        # 改进一：预先算好"站在每个城市上时，哪些城市享有完整权重"。
        # mode == "off" 时为 None，_choose_next_city 里会跳过整套判断，等于原版。
        self.candidate_masks = build_candidate_masks(
            self.city_to_ball,
            self.balls,
            self.config.candidate_mode,
            self.config.neighbor_margin,
            self.count,
            self.config.candidate_rule,
            self.config.neighbor_k,
        )
        self.candidate_stats = mask_summary(self.candidate_masks, self.count)

        # 改进二：高质量路线存档。搜索过程中只做拷贝，不改搜索流程，
        # 打磨留到 solve() 结束后统一进行（理由见 _archive_route 的注释）。
        self.route_archive: dict[frozenset, Route] = {}

    def _new_matrix(self) -> list[list[float]]:
        """初始化一张对角线为零、其他位置为 tau_0 的矩阵。
        原 C++：main.cpp:589 InitpheromoneMatrices。
        """
        return [
            [0.0 if i == j else self.tau_0 for j in range(self.count)]
            for i in range(self.count)
        ]

    def _boost_initial_route(self, route: Route) -> None:
        """对贪心路径里连续位于同一粒球的边增加初始信息素。
        原 C++：main.cpp:605 boost_pheromone_by_init_route。
        """
        if self.count <= 1:
            return
        start = 0
        while start < self.count:
            ball_id = self.city_to_ball[route.city_path[start]]
            end = start
            while (
                end + 1 < self.count
                and self.city_to_ball[route.city_path[end + 1]] == ball_id
            ):
                end += 1
            length = end - start + 1
            if length >= 2:
                boost = 1.0 + 2.0 * (length - 1) / (length + 1)
                delta = self.tau_0 * (boost - 1.0)
                for matrix in self.matrices:
                    for index in range(start, end):
                        u, v = route.city_path[index : index + 2]
                        matrix[u][v] += delta
                        matrix[v][u] += delta
            start = end + 1
        last, first = route.city_path[-1], route.city_path[0]
        if self.city_to_ball[last] == self.city_to_ball[first]:
            for matrix in self.matrices:
                matrix[last][first] += self.tau_0
                matrix[first][last] += self.tau_0

    def _outside_scale(self) -> float:
        """候选集之外城市的权重折扣系数（改进一）。

        candidate_masks 为 None（mode="off"）时返回 1.0，调用方会整段跳过判断。
        硬模式恒为 0 —— 等价于学长原来的"别的直接屏蔽"。
        软模式返回 outside_scale；若 anneal_fes > 0，则在前这段预算里由 0 线性放开。
        """
        if self.candidate_masks is None:
            return 1.0
        if self.config.candidate_mode.startswith("hard"):
            return 0.0
        target = self.config.outside_scale
        span = self.config.anneal_fes
        if span > 0.0:
            progress = self.fes / (span * self.config.max_fes)
            return target * min(1.0, max(0.0, progress))
        return target

    def _choose_next_city(self, ant: Ant, matrix: list[list[float]]) -> int:
        """用信息素强度与距离倒数计算轮盘赌权重并选城市；可选候选城市加权。
        原 C++：main.cpp:441 choose_next_city。
        """
        current = ant.route.city_path[-1]
        available = [i for i, visited in enumerate(ant.visited) if not visited]
        scale = self._outside_scale()
        weights = []
        for i in available:
            distance = self.distance[current][i]
            if distance <= 0:
                weights.append(0.0)
                continue
            weight = (
                matrix[current][i] ** self.config.alpha
                * (1.0 / distance) ** self.config.beta
            )
            # scale < 1 才需要查候选表；mode="off"、soft 且 outside_scale=1 时都跳过。
            if scale < 1.0 and i not in self.candidate_masks[current]:
                weight *= scale
            weights.append(weight)
        total = sum(weights)
        if total <= 0:
            # 候选集内一个城市都没有（或权重全为 0）时，退回全部未访问城市，
            # 保证蚂蚁永远能构造出一条完整回路。
            return available[0]
        probability = self.selection_random.random()
        cumulative = 0.0
        for city, weight in zip(available, weights):
            cumulative += weight / total
            if probability <= cumulative:
                return city
        return available[-1]

    def _build_ants(self, assignments: list[int]) -> list[Ant]:
        """每只蚂蚁随机选起点，再用所属信息素矩阵构造完整回路。
        原 C++：main.cpp:404 init_ants、418 build_trip、435 add_to_Route。
        """
        ants: list[Ant] = []
        for group in assignments:
            start = self.start_random.randrange(self.count)
            visited = [False] * self.count
            visited[start] = True
            ant = Ant(Route([start]), visited)
            while len(ant.route.city_path) < self.count:
                city = self._choose_next_city(ant, self.matrices[group])
                ant.route.city_path.append(city)
                ant.visited[city] = True
            ant.route.length = path_len(ant.route, self.distance)
            ants.append(ant)
        return ants

    def _remember_route(self, route: Route) -> str:
        """维护候选队列，并告知调用方是否跳过后续局部搜索。
        原 C++ 代码段：main.cpp:749 AddRouteToQueue、125 twoOpt 中的候选队列更新。

        与改动前逐字一致 —— 改进二不碰这里，只在 _archive_route 里另开一份存档。
        """
        if not self.best_routes:
            self.best_routes.append(_copy_route(route))
            return "first"
        if any(
            edge_similarity(route.city_path, old.city_path) >= 0.999999
            for old in self.best_routes
        ):
            return "duplicate"
        longest = max(range(len(self.best_routes)), key=lambda i: self.best_routes[i].length)
        longest_length = self.best_routes[longest].length
        if route.length < longest_length:
            self.best_routes[longest] = _copy_route(route)
        elif route.length == longest_length:
            # C++ 的 min_element 配合 a.length > b.length 实际选中最长路；
            # 虽然变量名叫 shortestRouteIt，这里按原代码行为转换。
            self.best_routes.append(_copy_route(route))
        return "new"

    def _archive_route(self, route: Route) -> None:
        """改进二上半句：把"接近当前最优"的路线悄悄抄进存档。

        学长原话："把这个高质量的路线要保留 —— 实际训练好模型后，直接跑的时候
        取 argmax，导致命中的路线少。" 对应到这里：只留最优长度的那几条不够，
        差一点点的高质量路线也要留下来，才有机会被打磨成精确解。

        为什么要用独立的存档、而不动 best_routes：
        1. best_routes 是"重复路线跳过 2-opt"这条剪枝的依据。往里塞次优路线，
           几乎所有蚂蚁都会变成"新路线"，每个都要跑一遍 O(n^2) 的 2-opt ——
           实测 composite10_55（200k FES）从 16 秒涨到 3 分钟以上还没跑完。
        2. 存档只做一次 O(n) 拷贝 + 一次字典查表，开销可以忽略，
           真正的打磨放到搜索结束之后（_polish_archive）。
        """
        if self.config.keep_tolerance <= 1.0 or not self.best_routes:
            return
        bar = min(old.length for old in self.best_routes) * self.config.keep_tolerance
        if route.length > bar:
            return
        signature = _signature(route.city_path)
        if signature in self.route_archive:
            return
        self.route_archive[signature] = _copy_route(route)
        limit = self.config.queue_limit or DEFAULT_ARCHIVE_LIMIT
        if len(self.route_archive) > limit:
            # 一次性砍掉一半（而不是每来一条删一条），把排序代价摊薄到可以忽略。
            survivors = sorted(self.route_archive.values(), key=lambda item: item.length)
            self.route_archive = {
                _signature(item.city_path): item for item in survivors[: limit // 2]
            }

    def _polish_archive(self, rounds: int | None = None) -> None:
        """改进二下半句：对存档里每条路线跑 2-opt，把近似解打磨成精确解。

        只在 final_polish 打开时由 solve() 调用。打磨出更短的路线后，
        _two_opt 内部会自动交给 _remember_route，于是自然参与最终输出。

        三个细节都是有原因的：
        1. polish_relax（默认开）：关掉"关键边保护"。那条保护是给搜索过程用的 ——
           防止蚁群刚学会的好结构被反复切坏。搜索已经结束，没有收敛过程要保护了，
           再拦一刀只会挡住可改进的两刀交换。
        2. polish_rotations：_two_opt 是一次"从前往后"的扫描，先扫到哪一刀就先改哪一刀，
           所以结果和回路的起点有关。同一条回路换个起点就是另一套扫描顺序，
           等于多给几次机会；每次都留最短的那条，只赚不亏。
        3. rounds：扫完一遍之后，后面几刀的反转可能又给前面片段创造新机会，
           所以扫到不再有改进为止。
        """
        protect = not self.config.polish_relax
        rotations = max(1, self.config.polish_rotations)
        rounds = max(1, self.config.polish_rounds if rounds is None else rounds)
        for _ in range(rounds):
            improved = False
            for signature, item in list(self.route_archive.items()):
                if self.route_archive.get(signature) is not item:
                    continue          # 本轮已被别的条目改写过，跳过
                best = item
                for shift in range(rotations):
                    start = (self.count * shift) // rotations
                    path = item.city_path
                    candidate = Route(path[start:] + path[:start], item.length)
                    self._two_opt(candidate, protect_key_edges=protect)
                    if candidate.length < best.length:
                        best = candidate
                if best.length < item.length:
                    # 打磨后的路线要**换到新的指纹上**：它已经不是原来那条回路了。
                    # 不换的话，字典键会过期，多条打磨结果可能其实撞成同一条最优解
                    # （实测 composite6_39 有 147 次"精修到最优长度"其实都是同一个解）。
                    del self.route_archive[signature]
                    self.route_archive[_signature(best.city_path)] = _copy_route(best)
                    improved = True
            if not improved:
                break

    def _two_opt(self, route: Route, protect_key_edges: bool = True) -> None:
        """按粒球边与关键边规则筛选可反转片段，更新候选回路。
        原 C++：main.cpp:72 canSwap、112 reverseSegment、125 twoOpt。

        protect_key_edges=False 时关掉"命中关键边就跳过"这条保护，
        用于搜索结束后的收尾精修：那时已经没有收敛过程要保护了，
        保留这条限制只会挡住一部分可改进的两刀交换（见 _polish_archive）。
        """
        n = self.count
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                before = (i - 1 + n) % n
                after = (j + 1) % n
                a, b = route.city_path[before], route.city_path[i]
                c, d = route.city_path[j], route.city_path[after]
                removed_edges = (tuple(sorted((a, b))), tuple(sorted((c, d))))
                cross_ball = any(
                    self.city_to_ball[u] != self.city_to_ball[v]
                    for u, v in removed_edges
                )
                threshold = int(_float32(_float32(self.fes / self.config.max_fes) * n))
                if protect_key_edges and not cross_ball and any(
                    edge in self.key_edges[:threshold] for edge in removed_edges
                ):
                    continue
                if before == j or after == i:
                    continue
                old_distance = self.distance[a][b] + self.distance[c][d]
                new_distance = self.distance[a][c] + self.distance[b][d]
                if self.city_to_ball[b] == self.city_to_ball[c]:
                    if old_distance - new_distance < 0.1:
                        continue
                # 源码中的“边界点”分支没有实际操作，因此无需移植空分支。
                self.fes = _float32(self.fes + _float32(4.0 / n))
                if new_distance <= old_distance:
                    route.city_path[i : j + 1] = reversed(route.city_path[i : j + 1])
                    route.length = path_len(route, self.distance)
                    self._remember_route(route)

    def _update_key_edges(self) -> None:
        """统计候选回路中出现频率最高的 n 条无向边。
        原 C++：main.cpp:5 initEdgeCountTable、26 updateEdgeCountsForAllAnts、43 identifyAndUpdateKeyEdges。
        """
        # 保留原哈希表的线性探测位置；出现次数相同时，原冒泡排序保持原槽位顺序。
        table_size = 1000
        edge_table: list[tuple[int, int] | None] = [None] * table_size
        counts = [0] * table_size
        for route in self.best_routes:
            path = route.city_path
            for index, city in enumerate(path):
                edge = tuple(sorted((city, path[(index + 1) % len(path)])))
                slot = (edge[0] * 100 + edge[1]) % table_size
                for _ in range(table_size):
                    if edge_table[slot] is None:
                        edge_table[slot] = edge
                        counts[slot] = 1
                        break
                    if edge_table[slot] == edge:
                        counts[slot] += 1
                        break
                    slot = (slot + 1) % table_size
                else:
                    raise RuntimeError("关键边哈希表已满")
        slots = sorted(
            (index for index, count in enumerate(counts) if count > 0),
            key=lambda index: (-counts[index], index),
        )
        self.key_edges = [edge_table[index] for index in slots[: self.count]]

    def _add_routes(self, ants: list[Ant], sizes: list[int]) -> None:
        """逐组维护候选回路；首个和重复路径跳过 2-opt。
        原 C++：main.cpp:749 AddRouteToQueue。
        """
        start = 0
        for size in sizes:
            for ant in ants[start : start + size]:
                # 改进二：不管这条路线进不进候选队列，先看看值不值得存档。
                self._archive_route(ant.route)
                status = self._remember_route(ant.route)
                if status in ("first", "duplicate"):
                    continue
                if self.fes > self.config.max_fes:
                    return
                self._two_opt(ant.route)
            start += size

    def _evaporate(self) -> None:
        """让每张矩阵的信息素按 rho 衰减，并保留源码下限。
        原 C++：main.cpp:499 pheromone_evaporation。
        """
        for matrix in self.matrices:
            for row in matrix:
                for index, value in enumerate(row):
                    row[index] = max((1.0 - self.config.rho) * value, 1.0e-64)

    @staticmethod
    def _deposit(matrix: list[list[float]], route: Route, amount: float) -> None:
        """沿闭合回路双向累加信息素。
        Python 辅助：抽出 main.cpp:550 bubbleSortAndAccumulatePheromone 中的双向边累加。
        """
        path = route.city_path
        for index, u in enumerate(path):
            v = path[(index + 1) % len(path)]
            matrix[u][v] += amount
            matrix[v][u] += amount

    def _accumulate(self, ants: list[Ant], sizes: list[int], assignments: list[int]) -> None:
        """每组按名次挑选约 30% 的蚂蚁，分等级增补信息素。
        原 C++：main.cpp:550 bubbleSortAndAccumulatePheromone。
        """
        start = 0
        for size in sizes:
            ants[start : start + size] = sorted(
                ants[start : start + size], key=lambda ant: ant.route.length
            )
            update_count = max(1, int(size * self.config.update_percentage))
            matrix = self.matrices[assignments[start]]
            for rank, ant in enumerate(ants[start : start + update_count], 1):
                amount = (update_count - rank + 1) / ant.route.length
                self._deposit(matrix, ant.route, amount)
            start += size

    def solve(self) -> list[Route]:
        """运行 main.cpp 的搜索循环，返回按长度排序的候选回路。
        原 C++ 主循环：main.cpp:1127-1183；含 main.cpp:816 SavepreviousBestPaths、978 AddPheromoneMatrix。
        """
        group_count = 2
        sizes = _group_sizes(self.config.ant_count, group_count)
        assignments = _assignments(sizes)
        previous_best: list[Route] = []
        expansion_count = 1
        least_similar = 0
        while self.fes < self.config.max_fes:
            ants = self._build_ants(assignments)
            if not previous_best:
                _sort_within_groups(ants, sizes)
                previous_best = [_copy_route(ants[sum(sizes[:g])].route) for g in range(group_count)]

            self._add_routes(ants, sizes)
            _form_groups(ants, sizes)
            assignments, least_similar = _match_groups(
                ants, previous_best, sizes, least_similar
            )
            _sort_within_groups(ants, sizes)
            start = 0
            for size in sizes:
                previous_best[assignments[start]] = _copy_route(ants[start].route)
                start += size
            self._evaporate()
            self._accumulate(ants, sizes, assignments)

            if (
                group_count < self.config.max_pheromones
                and self.config.max_pheromones > 2
                and self.fes >= _float32(
                    _float32(float(expansion_count) * self.config.max_fes)
                    / (self.config.max_pheromones - 2)
                )
            ):
                self.matrices.append([row.copy() for row in self.matrices[least_similar]])
                previous_best.append(_copy_route(previous_best[least_similar]))
                group_count += 1
                sizes = _group_sizes(self.config.ant_count, group_count)
                assignments = _assignments(sizes)
                expansion_count += 1

            self.fes = _float32(self.fes + self.config.ant_count)
            self._update_key_edges()
        # 改进二：预算用完之后，再对高质量存档做一轮不计入预算的 2-opt 精修。
        # 关键边表此刻已经是最全的一份，正好用来保护已经找对的结构。
        if self.config.final_polish and self.route_archive:
            self._polish_archive()
        return sorted(self.best_routes, key=lambda route: route.length)
