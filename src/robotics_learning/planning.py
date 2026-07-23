"""
运动规划 (Motion Planning)

包含：
- 图搜索：Dijkstra, A*
- 采样规划：PRM, RRT, RRT*
- 势场法
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import heapq
from collections import defaultdict


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
             rng: np.random.RandomState = None) -> dict:
    """
    PRM 概率路图规划

    参数:
        c_space_free: 判断构型是否无碰撞的函数
        bounds: (dim, 2) 各维度的下界和上界
        n_samples: 采样数
        k_neighbors: 连接近邻数
        start, goal: 起点和终点（如果提供则构建路图后搜索）
    """
    if rng is None:
        rng = np.random.RandomState()

    dim = bounds.shape[0]

    # 采样
    samples = []
    for _ in range(n_samples):
        q = rng.uniform(bounds[:, 0], bounds[:, 1])
        if c_space_free(q):
            samples.append(q)
    samples = np.array(samples)

    # 构建 k-NN 图
    adj = defaultdict(list)
    for i, qi in enumerate(samples):
        dists = np.linalg.norm(samples - qi, axis=1)
        neighbors = np.argsort(dists)[1:k_neighbors+1]
        for j in neighbors:
            # 边碰撞检测（简化：只检查中点）
            mid = (qi + samples[j]) / 2
            if c_space_free(mid):
                adj[i].append((j, dists[j]))

    return {'samples': samples, 'adj': adj}


# =============================================================================
# RRT (Rapidly-exploring Random Tree)
# =============================================================================

class RRTNode:
    """RRT 节点"""
    __slots__ = ('q', 'parent', 'cost')
    def __init__(self, q, parent=None, cost=0.0):
        self.q = q
        self.parent = parent
        self.cost = cost


def rrt_plan(c_space_free: Callable[[np.ndarray], bool],
             bounds: np.ndarray, start: np.ndarray, goal: np.ndarray,
             max_iter: int = 1000, step_size: float = 0.1,
             goal_bias: float = 0.05,
             rng: np.random.RandomState = None) -> Tuple[Optional[List[np.ndarray]], List[RRTNode]]:
    """
    RRT 快速随机搜索树

    参数:
        c_space_free: 碰撞检测函数
        bounds: (dim, 2) 约束范围
        start, goal: 起点和目标
        max_iter: 最大迭代次数
        step_size: 每次扩展步长
        goal_bias: 向目标采样的概率
    """
    if rng is None:
        rng = np.random.RandomState()

    dim = bounds.shape[0]
    nodes = [RRTNode(np.array(start))]

    for _ in range(max_iter):
        # 采样
        if rng.random() < goal_bias:
            q_rand = np.array(goal)
        else:
            q_rand = rng.uniform(bounds[:, 0], bounds[:, 1])

        # 找最近节点
        dists = [np.linalg.norm(n.q - q_rand) for n in nodes]
        nearest_idx = np.argmin(dists)
        q_near = nodes[nearest_idx].q

        # 向 q_rand 扩展一步
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            continue
        q_new = q_near + step_size * direction / dist

        # 碰撞检测
        if not c_space_free(q_new):
            continue

        # 边碰撞检测（检查中点）
        if not c_space_free((q_near + q_new) / 2):
            continue

        new_node = RRTNode(q_new, nodes[nearest_idx])
        nodes.append(new_node)

        # 检查是否到达目标
        if np.linalg.norm(q_new - goal) < step_size:
            # 回溯路径
            path = [np.array(goal)]
            node = new_node
            while node is not None:
                path.append(node.q)
                node = node.parent
            return path[::-1], nodes

    return None, nodes


def rrt_star_plan(c_space_free, bounds, start, goal,
                  max_iter=1000, step_size=0.1, search_radius=0.3,
                  rng=None):
    """
    RRT* 规划（加入 rewire 步骤实现渐进最优）

    参数同 RRT，额外：
        search_radius: 重连线搜索半径
    """
    if rng is None:
        rng = np.random.RandomState()

    dim = bounds.shape[0]
    nodes = [RRTNode(np.array(start), cost=0.0)]

    for _ in range(max_iter):
        q_rand = rng.uniform(bounds[:, 0], bounds[:, 1])
        if rng.random() < 0.05:
            q_rand = np.array(goal)

        dists = [np.linalg.norm(n.q - q_rand) for n in nodes]
        nearest_idx = np.argmin(dists)
        q_near = nodes[nearest_idx].q

        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            continue
        q_new = q_near + step_size * direction / dist

        if not c_space_free(q_new):
            continue
        if not c_space_free((q_near + q_new) / 2):
            continue

        # 找搜索半径内的节点
        nearby = []
        for j, n in enumerate(nodes):
            d = np.linalg.norm(n.q - q_new)
            if d < search_radius:
                nearby.append((j, d))

        # 选最优父节点
        best_parent = nodes[nearest_idx]
        best_cost = best_parent.cost + np.linalg.norm(q_new - best_parent.q)

        for j, d in nearby:
            if c_space_free((nodes[j].q + q_new) / 2):
                cost = nodes[j].cost + d
                if cost < best_cost:
                    best_parent = nodes[j]
                    best_cost = cost

        new_node = RRTNode(q_new, best_parent, best_cost)
        nodes.append(new_node)

        # Rewire：用新节点改善附近节点的代价
        for j, d in nearby:
            candidate_cost = new_node.cost + d
            if candidate_cost < nodes[j].cost:
                if c_space_free((q_new + nodes[j].q) / 2):
                    nodes[j].parent = new_node
                    nodes[j].cost = candidate_cost

    # 找最优路径
    goal_dists = [np.linalg.norm(n.q - goal) for n in nodes]
    best_idx = np.argmin(goal_dists)
    if goal_dists[best_idx] < step_size * 2:
        path = [np.array(goal)]
        node = nodes[best_idx]
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path[::-1], nodes

    return None, nodes


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

        # 斥力
        f_rep = np.zeros_like(q)
        U_rep = 0
        for obs in obstacles:
            d = np.linalg.norm(q - obs['center'])
            if d < rho_0 and d > 1e-10:
                direction = (q - obs['center']) / d
                f_rep += k_rep * (1/d - 1/rho_0) * (1/d**2) * direction
                U_rep += 0.5 * k_rep * (1/d - 1/rho_0)**2

        f_total = f_att + f_rep

        if np.linalg.norm(f_total) < 1e-10:
            # 局部极小值
            break

        q = q + step_size * f_total / np.linalg.norm(f_total)
        path.append(q.copy())

        if np.linalg.norm(q - q_goal) < tol:
            break

    return path, np.array(U_history)
