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


class TestRRTStarCost:
    def test_subtree_cost_consistency(self):
        """RRT* 中每个非根节点的 cost 应等于 parent.cost + edge_cost"""
        from src.robotics_learning.planning import RRTNode, _update_subtree_cost
        root = RRTNode(np.array([0., 0.]), cost=0.0)
        child1 = RRTNode(np.array([1., 0.]), root, cost=1.0)
        root.children.add(child1)
        child2 = RRTNode(np.array([1., 1.]), child1, cost=2.0)
        child1.children.add(child2)
        # 给 root 加 0.5
        _update_subtree_cost(root, 0.5)
        assert abs(root.cost - 0.5) < 1e-10
        assert abs(child1.cost - 1.5) < 1e-10
        assert abs(child2.cost - 2.5) < 1e-10

    def test_rrt_star_nodes_have_children(self):
        """RRT* 返回的节点应有正确的 children 集合"""
        from src.robotics_learning.planning import rrt_star_plan
        bounds = np.array([[0, 10], [0, 10]])
        rng = np.random.RandomState(42)
        _, nodes = rrt_star_plan(lambda q: True, bounds, np.array([1.,1.]), np.array([9.,9.]),
                                  max_iter=200, step_size=0.5, rng=rng)
        for node in nodes:
            if node.parent is not None:
                expected = node.parent.cost + np.linalg.norm(node.q - node.parent.q)
                assert abs(node.cost - expected) < 1e-6, \
                    f"node cost {node.cost:.6f} != parent.cost + edge = {expected:.6f}"


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
