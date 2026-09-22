"""GB-MACO 的蚁群搜索、信息素更新和粒球约束 2-opt。"""

from dataclasses import dataclass
from random import Random
from struct import pack, unpack
from time import time

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

    def _choose_next_city(self, ant: Ant, matrix: list[list[float]]) -> int:
        """用信息素强度与距离倒数计算轮盘赌权重并选城市。
        原 C++：main.cpp:441 choose_next_city。
        """
        current = ant.route.city_path[-1]
        available = [i for i, visited in enumerate(ant.visited) if not visited]
        weights = [
            (
                matrix[current][i] ** self.config.alpha
                * (1.0 / self.distance[current][i]) ** self.config.beta
                if self.distance[current][i] > 0
                else 0.0
            )
            for i in available
        ]
        total = sum(weights)
        if total <= 0:
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
        if route.length < self.best_routes[longest].length:
            self.best_routes[longest] = _copy_route(route)
        elif route.length == longest_length:
            # C++ 的 min_element 配合 a.length > b.length 实际选中最长路；
            # 虽然变量名叫 shortestRouteIt，这里按原代码行为转换。
            self.best_routes.append(_copy_route(route))
        return "new"

    def _two_opt(self, route: Route) -> None:
        """按粒球边与关键边规则筛选可反转片段，更新候选回路。
        原 C++：main.cpp:72 canSwap、112 reverseSegment、125 twoOpt。
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
                if not cross_ball and any(
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
        return sorted(self.best_routes, key=lambda route: route.length)
