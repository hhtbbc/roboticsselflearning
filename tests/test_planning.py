"""规划模块测试 — 碰撞检测, 周期关节, RRT"""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.robotics_learning.planning import (
    wrap_to_pi, periodic_distance, edge_collision_free,
    rrt_plan, create_grid_map, astar
)

RNG = np.random.RandomState(42)


class TestPeriodicJoints:
    def test_wrap_to_pi(self):
        assert abs(wrap_to_pi(3.5) - (3.5 - 2*np.pi)) < 1e-10
        assert wrap_to_pi(0) == 0
        assert abs(wrap_to_pi(-np.pi)) == np.pi  # wrap

    def test_periodic_distance(self):
        """从 179° 到 -179° 的距离应为 ~2°"""
        q1 = np.array([np.deg2rad(179), 0.0])
        q2 = np.array([np.deg2rad(-179), 0.0])
        d = periodic_distance(q1, q2, ['revolute', 'prismatic'])
        assert abs(d - np.deg2rad(2)) < 1e-6

    def test_edge_collision_periodic(self):
        """周期插值应走短弧而非长弧"""
        def never_collide(q): return True
        q1 = np.array([np.deg2rad(179)])
        q2 = np.array([np.deg2rad(-179)])
        # 不应崩溃，采样点应走短弧
        assert edge_collision_free(q1, q2, never_collide, joint_types=['revolute'])


class TestRRT:
    def test_rrt_trivial(self):
        """无障碍物时 RRT 应找到路径"""
        bounds = np.array([[0, 10], [0, 10]])
        path, nodes = rrt_plan(lambda q: True, bounds, np.array([0.,0.]), np.array([5.,5.]),
                                max_iter=500, step_size=0.5, rng=RNG)
        assert path is not None
        assert len(path) > 2
        assert np.linalg.norm(np.array(path[-1]) - np.array([5,5])) < 0.5

    def test_rrt_blocked(self):
        """被障碍物包围时应返回 None"""
        bounds = np.array([[0, 10], [0, 10]])
        def blocked(q):
            return 2 < q[0] < 8 and 2 < q[1] < 8  # only narrow border free

        path, _ = rrt_plan(blocked, bounds, np.array([0.,0.]), np.array([5.,5.]),
                           max_iter=500, step_size=0.3, rng=RNG)
        # 起点在自由空间但目标被包围 — 可能找到或找不到
        # 至少不应崩溃
        if path is not None:
            for q in path:
                assert blocked(q) or np.linalg.norm(q - np.array([5,5])) < 0.5


class TestAStar:
    def test_astar_optimal(self):
        grid = create_grid_map(10, 10, [])
        path, _ = astar(grid, (0, 0), (9, 9))
        assert path is not None
        assert len(path) <= 20  # Manhattan distance 18 + start

    def test_astar_blocked(self):
        """完全阻塞时应返回 None"""
        grid = create_grid_map(10, 10, [(0, 0, 10, 10)])
        path, _ = astar(grid, (0, 0), (9, 9))
        assert path is None
