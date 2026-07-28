"""
运动规划 (Motion Planning)

包含：
- 图搜索：Dijkstra, A*
- 采样规划：PRM, RRT, RRT*
- 势场法
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
import heapq
from collections import defaultdict


# =============================================================================
# 通用工具
# =============================================================================

def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def periodic_distance(q1: np.ndarray, q2: np.ndarray,
                      joint_types: List[str] = None) -> float:
    """
    考虑连续旋转关节周期性的构型空间距离。
    仅 'continuous' 类型关节使用 wrapToPi，其它使用线性差。
    """
    d = q1 - q2
    if joint_types is not None:
        for i, jt in enumerate(joint_types):
            if jt == 'continuous':
                d[i] = wrap_to_pi(d[i])
    return np.linalg.norm(d)


class ConfigurationSpace:
    """统一 C-space 描述，封装周期拓扑、距离、插值等。

    关节类型:
        - 'continuous': 严格周期关节 (如无限制旋转轴)，执行 wrapToPi
        - 'revolute':   带硬限位的旋转关节，线性差（不 wrap）
        - 'prismatic':  线性移动关节，线性差

    用法:
        cspace = ConfigurationSpace(bounds, joint_types=['continuous', 'revolute'])
        dist = cspace.distance(q1, q2)
        q_mid = cspace.interpolate(q1, q2, 0.5)  # 自动 normalize
        q_new = cspace.steer(q_near, q_rand, step_size)  # 自动 normalize
        q_norm = cspace.normalize(q)
        ok = cspace.within_bounds(q)
    """
    def __init__(self, bounds: np.ndarray, joint_types: List[str] = None):
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]
        # 默认全部为 revolute (带限位)，不默认 continuous
        self.joint_types = joint_types or ['revolute'] * self.dim

    def _wrap_dim(self, i: int, val: float) -> float:
        """仅对 continuous 关节执行 wrap"""
        if self.joint_types[i] == 'continuous':
            return wrap_to_pi(val)
        return val

    def difference(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """从 q_from 到 q_to 的最短弧向量（仅 continuous 关节 wrap）"""
        d = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
        for i, jt in enumerate(self.joint_types):
            if jt == 'continuous':
                d[i] = wrap_to_pi(d[i])
        return d

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """C-space 周期距离"""
        return float(np.linalg.norm(self.difference(q1, q2)))

    def interpolate(self, q1: np.ndarray, q2: np.ndarray,
                    alpha: float) -> np.ndarray:
        """C-space 插值（continuous 关节走最短弧），返回后 normalize"""
        diff = self.difference(q1, q2)
        result = np.asarray(q1) + alpha * diff
        return self.normalize(result)

    def steer(self, q_near: np.ndarray, q_rand: np.ndarray,
              step_size: float) -> np.ndarray:
        """从 q_near 向 q_rand 扩展 step_size，返回后 normalize"""
        diff = self.difference(q_near, q_rand)
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            return self.normalize(np.asarray(q_near).copy())
        eta = min(step_size, dist)
        result = np.asarray(q_near) + eta * diff / dist
        return self.normalize(result)

    def normalize(self, q: np.ndarray) -> np.ndarray:
        """仅对 continuous 关节 wrap 到 [-π, π]"""
        q = np.asarray(q, dtype=float).copy()
        for i, jt in enumerate(self.joint_types):
            if jt == 'continuous':
                q[i] = wrap_to_pi(q[i])
        return q

    def within_bounds(self, q: np.ndarray) -> bool:
        """检查 q 是否在边界内"""
        q = np.asarray(q)
        return bool(np.all(self.bounds[:, 0] <= q) and
                    np.all(q <= self.bounds[:, 1]))


def validate_planning_problem(c_space_free: Callable, bounds: np.ndarray,
                               start: np.ndarray, goal: np.ndarray,
                               step_size: float, max_iter: int,
                               joint_types: List[str] = None) -> None:
    """统一验证规划问题的输入。

    检查:
        bounds 形状、下界 < 上界、start/goal 在边界内且无碰撞、
        step_size > 0、max_iter > 0。

    失败时抛出 ValueError 或 RuntimeError。
    """
    bounds = np.asarray(bounds)
    start = np.asarray(start)
    goal = np.asarray(goal)
    cspace = ConfigurationSpace(bounds, joint_types)

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds 形状必须为 (dim, 2)，实际为 {bounds.shape}")
    if not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError(f"bounds 下界必须严格小于上界: {bounds}")
    if step_size <= 0:
        raise ValueError(f"step_size 必须 > 0, 实际为 {step_size}")
    if max_iter <= 0:
        raise ValueError(f"max_iter 必须 > 0, 实际为 {max_iter}")

    if not cspace.within_bounds(start):
        raise ValueError(f"Start {start} outside bounds {bounds}")
    if not cspace.within_bounds(goal):
        raise ValueError(f"Goal {goal} outside bounds {bounds}")
    if not c_space_free(start):
        raise ValueError(f"Start {start} in collision")
    if not c_space_free(goal):
        raise ValueError(f"Goal {goal} in collision")


def edge_collision_free(q_a: np.ndarray, q_b: np.ndarray,
                        collision_fn: Callable[[np.ndarray], bool],
                        resolution: float = 0.05,
                        joint_types: List[str] = None) -> bool:
    """
    沿整条边 (q_a → q_b) 以给定分辨率插值检测碰撞。

    仅 'continuous' 关节使用最短弧插值，其它使用线性插值。
    采样数 = max(2, ceil(periodic_distance / resolution))

    返回: True 如果整条边都无碰撞
    """
    dist = periodic_distance(q_a, q_b, joint_types)
    n_samples = max(2, int(np.ceil(dist / resolution)))
    for alpha in np.linspace(0, 1, n_samples):
        if joint_types is not None:
            q_mid = q_a.copy()
            for d, jt in enumerate(joint_types):
                if jt == 'continuous':
                    diff = wrap_to_pi(q_b[d] - q_a[d])
                    q_mid[d] = q_a[d] + alpha * diff
                else:
                    q_mid[d] = (1 - alpha) * q_a[d] + alpha * q_b[d]
        else:
            q_mid = (1 - alpha) * q_a + alpha * q_b
        if not collision_fn(q_mid):
            return False
    return True


def periodic_steer(q_near: np.ndarray, q_rand: np.ndarray,
                   step_size: float,
                   joint_types: List[str] = None) -> np.ndarray:
    """
    从 q_near 向 q_rand 扩展 step_size。
    仅 'continuous' 关节使用最短弧方向。
    """
    direction = q_rand - q_near
    if joint_types is not None:
        for i, jt in enumerate(joint_types):
            if jt == 'continuous':
                direction[i] = wrap_to_pi(direction[i])
    dist = np.linalg.norm(direction)
    if dist < 1e-10:
        return q_near.copy()
    eta = min(step_size, dist)
    return q_near + eta * direction / dist


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray,
                              seg_end: np.ndarray) -> float:
    """点到线段的最短距离"""
    seg = seg_end - seg_start
    seg_sq = np.dot(seg, seg)
    if seg_sq < 1e-15:
        return np.linalg.norm(point - seg_start)
    t = np.clip(np.dot(point - seg_start, seg) / seg_sq, 0.0, 1.0)
    projection = seg_start + t * seg
    return np.linalg.norm(point - projection)


# =============================================================================
# 地图与通用工具
# =============================================================================

def create_grid_map(width: int, height: int,
                    obstacles: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """创建二维网格地图，0=自由，1=障碍物"""
    grid = np.zeros((height, width), dtype=int)
    for ox, oy, ow, oh in obstacles:
        grid[oy:oy+oh, ox:ox+ow] = 1
    return grid


def get_neighbors_4(grid: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """4-连通邻域"""
    x, y = pos
    h, w = grid.shape
    neighbors = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
            neighbors.append((nx, ny))
    return neighbors


def get_neighbors_8(grid: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """8-连通邻域"""
    x, y = pos
    h, w = grid.shape
    neighbors = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
            cost = np.sqrt(2) if dx != 0 and dy != 0 else 1.0
            neighbors.append(((nx, ny), cost))
    return neighbors


# =============================================================================
# Dijkstra
# =============================================================================

def dijkstra(grid: np.ndarray, start: Tuple[int, int],
             goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
    """
    Dijkstra 最短路径搜索

    返回: (路径列表, 探索信息字典)
    """
    h, w = grid.shape
    dist = {start: 0.0}
    parent = {}
    visited = set()

    pq = [(0.0, start)]
    explored = []  # 记录探索顺序用于可视化

    while pq:
        d, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        explored.append(current)

        if current == goal:
            # 回溯路径
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1], {'explored': explored, 'dist': dist}

        for neighbor in get_neighbors_4(grid, current):
            new_dist = d + 1.0
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    return None, {'explored': explored, 'dist': dist}


# =============================================================================
# A*
# =============================================================================

def astar(grid: np.ndarray, start: Tuple[int, int],
          goal: Tuple[int, int],
          heuristic: Callable = None) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
    """
    A* 启发式搜索

    参数:
        grid: 网格地图
        start, goal: 起点、目标
        heuristic: 启发函数 h(n)，默认用欧氏距离
    """
    if heuristic is None:
        heuristic = lambda a, b: np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    h, w = grid.shape
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    parent = {}
    open_set = {start}
    explored = []

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, np.inf))
        explored.append(current)

        if current == goal:
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1], {'explored': explored, 'g_score': g_score}

        open_set.remove(current)

        for neighbor in get_neighbors_4(grid, current):
            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score.get(neighbor, np.inf):
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                open_set.add(neighbor)

    return None, {'explored': explored, 'g_score': g_score}


# =============================================================================
# PRM (Probabilistic Roadmap)
# =============================================================================

def prm_plan(c_space_free: Callable[[np.ndarray], bool],
             bounds: np.ndarray, n_samples: int = 200,
             k_neighbors: int = 5,
             start: np.ndarray = None, goal: np.ndarray = None,
             rng: np.random.RandomState = None,
             joint_types: List[str] = None) -> dict:
    """
    PRM 概率路图规划

    参数:
        c_space_free: 判断构型是否无碰撞的函数
        bounds: (dim, 2) 各维度的下界和上界
        n_samples: 采样数
        k_neighbors: 连接近邻数
        start, goal: 起点和终点（如果提供则构建路图后搜索）
        joint_types: 关节类型列表
    """
    if rng is None:
        rng = np.random.RandomState()

    cspace = ConfigurationSpace(bounds, joint_types)

    # 采样
    samples = []
    for _ in range(n_samples):
        q = rng.uniform(bounds[:, 0], bounds[:, 1])
        if c_space_free(q):
            samples.append(cspace.normalize(q))
    samples = np.array(samples)

    # 构建 k-NN 图 (使用 C-space 周期距离)
    adj = defaultdict(list)
    for i, qi in enumerate(samples):
        dists = np.array([cspace.distance(qi, sj) for sj in samples])
        neighbors = np.argsort(dists)[1:k_neighbors+1]
        for j in neighbors:
            if edge_collision_free(qi, samples[j], c_space_free,
                                   joint_types=cspace.joint_types):
                adj[i].append((j, dists[j]))
                adj[j].append((i, dists[j]))  # 无向图：对称连接

    # 如果提供了起终点，连接并搜索
    result: Dict[str, Any] = {'samples': samples, 'adj': dict(adj),
                                'path': None, 'success': False}
    if start is not None and goal is not None and len(samples) > 0:
        start_np = cspace.normalize(np.asarray(start))
        goal_np = cspace.normalize(np.asarray(goal))
        n = len(samples)
        for pt, idx in [(start_np, n), (goal_np, n+1)]:
            dists = np.array([cspace.distance(samples[k], pt)
                             for k in range(len(samples))])
            knn_idx = np.argsort(dists)[:k_neighbors]
            for k in knn_idx:
                if edge_collision_free(samples[k], pt, c_space_free,
                                       joint_types=cspace.joint_types):
                    adj[idx].append((k, dists[k]))
                    adj[k].append((idx, dists[k]))

        # Dijkstra 搜索从 start 到 goal
        start_id, goal_id = n, n+1
        pq = [(0.0, start_id)]
        dist_to = {start_id: 0.0}
        parent = {}
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal_id:
                path_ids = [u]
                while path_ids[-1] in parent:
                    path_ids.append(parent[path_ids[-1]])
                path_pts = []
                for pid in reversed(path_ids):
                    if pid < n:
                        path_pts.append(samples[pid])
                    elif pid == n:
                        path_pts.append(start_np)
                    else:
                        path_pts.append(goal_np)
                result['path'] = path_pts
                result['success'] = True
                break
            if d > dist_to.get(u, np.inf):
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist_to.get(v, np.inf):
                    dist_to[v] = nd; parent[v] = u
                    heapq.heappush(pq, (nd, v))

    return result


# =============================================================================
# RRT (Rapidly-exploring Random Tree)
# =============================================================================

class RRTNode:
    """RRT 节点"""
    __slots__ = ('q', 'parent', 'children', 'cost')
    def __init__(self, q, parent=None, cost=0.0):
        self.q = q
        self.parent = parent
        self.children = set()
        self.cost = cost


def rrt_plan(c_space_free: Callable[[np.ndarray], bool],
             bounds: np.ndarray, start: np.ndarray, goal: np.ndarray,
             max_iter: int = 1000, step_size: float = 0.1,
             goal_bias: float = 0.05,
             rng: np.random.RandomState = None,
             joint_types: List[str] = None) -> Tuple[Optional[List[np.ndarray]], List[RRTNode]]:
    """
    RRT 快速随机搜索树

    参数:
        c_space_free: 碰撞检测函数
        bounds: (dim, 2) 约束范围
        start, goal: 起点和目标
        max_iter: 最大迭代次数
        step_size: 每次扩展步长
        goal_bias: 向目标采样的概率
        joint_types: 关节类型列表 ('revolute'/'prismatic')，None 则全部为线性
    """
    if rng is None:
        rng = np.random.RandomState()

    validate_planning_problem(c_space_free, bounds, start, goal,
                              step_size, max_iter, joint_types)

    cspace = ConfigurationSpace(bounds, joint_types)
    start_np = np.asarray(start); goal_np = np.asarray(goal)

    nodes = [RRTNode(start_np)]

    for _ in range(max_iter):
        # 采样
        if rng.random() < goal_bias:
            q_rand = goal_np.copy()
        else:
            q_rand = rng.uniform(bounds[:, 0], bounds[:, 1])

        # 找最近节点（使用 C-space 周期距离）
        dists = [cspace.distance(n.q, q_rand) for n in nodes]
        nearest_idx = np.argmin(dists)
        q_near = nodes[nearest_idx].q

        # 向 q_rand 扩展一步
        q_new = cspace.steer(q_near, q_rand, step_size)

        # 节点碰撞检测
        if not c_space_free(q_new):
            continue

        # 边碰撞检测（沿整条边插值）
        if not edge_collision_free(q_near, q_new, c_space_free,
                                   joint_types=cspace.joint_types):
            continue

        new_node = RRTNode(q_new, nodes[nearest_idx])
        nodes[nearest_idx].children.add(new_node)
        nodes.append(new_node)

        # 检查是否到达目标
        if cspace.distance(q_new, goal_np) < step_size:
            if edge_collision_free(q_new, goal_np, c_space_free,
                                   joint_types=cspace.joint_types):
                path = [goal_np.copy()]
                node = new_node
                while node is not None:
                    path.append(node.q)
                    node = node.parent
                return path[::-1], nodes

    return None, nodes


def rrt_star_plan(c_space_free, bounds, start, goal,
                  max_iter=1000, step_size=0.1, search_radius=None,
                  rng=None, joint_types=None):
    """
    RRT* 规划（加入 rewire 步骤实现渐进最优）

    关键改进 vs RRT:
    1. 选最优父节点（非最近节点）
    2. Rewire: 新节点可能改善附近已有节点的代价
    3. 迭代结束后选连接目标的最优代价节点（非第一条路径）

    参数:
        c_space_free: 碰撞检测函数
        bounds: (dim, 2) 约束范围
        start, goal: 起点、目标
        max_iter: 最大迭代次数
        step_size: 扩展步长
        search_radius: 邻域半径（采用具有 RRT* 理论收缩形式的启发式邻域半径，
                       系数未按自由空间体积严格计算。None 则自动 = 3 * step_size）
        rng: 随机数生成器
        joint_types: 关节类型列表
    """
    if rng is None:
        rng = np.random.RandomState()

    validate_planning_problem(c_space_free, bounds, start, goal,
                              step_size, max_iter, joint_types)

    cspace = ConfigurationSpace(bounds, joint_types)
    start_np = np.asarray(start); goal_np = np.asarray(goal)

    dim = bounds.shape[0]
    nodes = [RRTNode(start_np, cost=0.0)]
    if search_radius is None:
        search_radius = 3.0 * step_size

    for n_iter in range(1, max_iter + 1):
        # 采样（含 goal bias）
        if rng.random() < 0.05:
            q_rand = goal_np.copy()
        else:
            q_rand = rng.uniform(bounds[:, 0], bounds[:, 1])

        # 找最近节点（C-space 周期距离）
        dists = [cspace.distance(n.q, q_rand) for n in nodes]
        nearest_idx = np.argmin(dists)
        q_near = nodes[nearest_idx].q

        # 向 q_rand 扩展
        q_new = cspace.steer(q_near, q_rand, step_size)

        # 节点碰撞检测
        if not c_space_free(q_new):
            continue

        # 边碰撞检测
        if not edge_collision_free(q_near, q_new, c_space_free,
                                   joint_types=cspace.joint_types):
            continue

        # 邻域搜索 — 采用具有 RRT* 理论收缩形式的启发式邻域半径
        # 系数未按自由空间体积严格计算，实际使用的是与 step_size 关联的启发式值
        r_n = min(search_radius, 2.0 * step_size * (np.log(n_iter + 1) / (n_iter + 1)) ** (1.0 / max(dim, 1)))

        nearby = []
        for j, n in enumerate(nodes):
            d = cspace.distance(n.q, q_new)
            if d < r_n:
                nearby.append((j, d))

        # 选最优父节点
        best_parent = nodes[nearest_idx]
        best_cost = best_parent.cost + cspace.distance(q_new, best_parent.q)

        for j, d in nearby:
            if edge_collision_free(nodes[j].q, q_new, c_space_free,
                                   joint_types=cspace.joint_types):
                cost = nodes[j].cost + d
                if cost < best_cost:
                    best_parent = nodes[j]
                    best_cost = cost

        new_node = RRTNode(q_new, best_parent, best_cost)
        best_parent.children.add(new_node)
        nodes.append(new_node)

        # Rewire: 用新节点改善附近节点的代价
        for j, d in nearby:
            candidate_cost = new_node.cost + d
            if candidate_cost < nodes[j].cost - 1e-10:
                if edge_collision_free(q_new, nodes[j].q, c_space_free,
                                       joint_types=cspace.joint_types):
                    # 从旧父节点移除
                    old_parent = nodes[j].parent
                    if old_parent is not None:
                        old_parent.children.discard(nodes[j])
                    # 加入新父节点
                    new_node.children.add(nodes[j])
                    cost_diff = candidate_cost - nodes[j].cost
                    nodes[j].parent = new_node
                    # 递归更新此节点及其所有后代的代价
                    _update_subtree_cost(nodes[j], cost_diff)

    # 在所有节点中找能连接到 goal 且代价最小的
    best_goal_cost = np.inf
    best_goal_node = None
    for n in nodes:
        d = cspace.distance(n.q, goal_np)
        if d < step_size * 1.5:
            if edge_collision_free(n.q, goal_np, c_space_free,
                                   joint_types=cspace.joint_types):
                total_cost = n.cost + d
                if total_cost < best_goal_cost:
                    best_goal_cost = total_cost
                    best_goal_node = n

    if best_goal_node is not None:
        path = [goal_np.copy()]
        node = best_goal_node
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path[::-1], nodes

    return None, nodes


def _update_subtree_cost(node: RRTNode, cost_diff: float):
    """递归更新节点及其所有后代的代价（DFS）"""
    node.cost += cost_diff
    for child in node.children:
        _update_subtree_cost(child, cost_diff)


# =============================================================================
# 势场法 (Potential Field)
# =============================================================================

def potential_field_plan(q_start: np.ndarray, q_goal: np.ndarray,
                         obstacles: List[dict],
                         k_att: float = 1.0, k_rep: float = 100.0,
                         rho_0: float = 2.0,
                         step_size: float = 0.05, max_iter: int = 1000,
                         tol: float = 0.1) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    势场法规划

    参数:
        q_start, q_goal: 起点和目标
        obstacles: 障碍物列表 [{'center': array, 'radius': float}, ...]
        k_att, k_rep: 引力/斥力增益
        rho_0: 斥力作用范围
        step_size, max_iter, tol: 迭代参数

    返回:
        path: 路径点列表
        U: 势能值历史
    """
    q = q_start.copy()
    path = [q.copy()]
    U_history = []

    for _ in range(max_iter):
        # 引力: F_att = -k_att * (q - q_goal)
        f_att = -k_att * (q - q_goal)
        U_att = 0.5 * k_att * np.dot(q - q_goal, q - q_goal)

        # 斥力
        f_rep = np.zeros_like(q)
        U_rep = 0
        for obs in obstacles:
            obs_radius = obs.get('radius', 0.0)
            # 到障碍物表面的净距离
            rho = np.linalg.norm(q - obs['center']) - obs_radius
            if rho <= 0:
                # 机器人已进入障碍物内部 — 碰撞
                U_history.append(U_att + 1e10)
                return path, np.array(U_history)
            if rho < rho_0:
                direction = (q - obs['center']) / np.linalg.norm(q - obs['center'])
                f_rep += k_rep * (1/rho - 1/rho_0) * (1/rho**2) * direction
                U_rep += 0.5 * k_rep * (1/rho - 1/rho_0)**2

        f_total = f_att + f_rep
        U_history.append(U_att + U_rep)

        if np.linalg.norm(f_total) < 1e-10:
            # 局部极小值
            break

        q = q + step_size * f_total / np.linalg.norm(f_total)
        path.append(q.copy())

        if np.linalg.norm(q - q_goal) < tol:
            break

    return path, np.array(U_history)
