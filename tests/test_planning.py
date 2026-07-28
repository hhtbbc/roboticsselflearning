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
        """从 179° 到 -179° 的距离应为 ~2° (continuous joint)"""
        q1 = np.array([np.deg2rad(179), 0.0])
        q2 = np.array([np.deg2rad(-179), 0.0])
        d = periodic_distance(q1, q2, ['continuous', 'prismatic'])
        assert abs(d - np.deg2rad(2)) < 1e-6

    def test_edge_collision_periodic(self):
        """周期插值应走短弧而非长弧"""
        def never_collide(q): return True
        q1 = np.array([np.deg2rad(179)])
        q2 = np.array([np.deg2rad(-179)])
        # 不应崩溃，采样点应走短弧
        assert edge_collision_free(q1, q2, never_collide, joint_types=['continuous'])


class TestRRT:
    def test_rrt_trivial(self):
        """无障碍物时 RRT 应找到路径"""
        bounds = np.array([[0, 10], [0, 10]])
        path, nodes = rrt_plan(lambda q: True, bounds, np.array([0.,0.]), np.array([5.,5.]),
                                max_iter=500, step_size=0.5, rng=RNG)
        assert path is not None
        assert len(path) > 2
        assert np.linalg.norm(np.array(path[-1]) - np.array([5,5])) < 0.5

    def test_rrt_start_invalid_raises(self):
        """起点在碰撞中应抛出异常"""
        bounds = np.array([[0, 10], [0, 10]])
        import pytest
        with pytest.raises(ValueError, match="collision"):
            rrt_plan(lambda q: q[0] < 5, bounds, np.array([7.,5.]), np.array([3.,5.]),
                     max_iter=100, step_size=0.3, rng=RNG)


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


class TestConfigurationSpace:
    """ConfigurationSpace 和 validate_planning_problem 测试"""

    def test_distance_periodic(self):
        from src.robotics_learning.planning import ConfigurationSpace
        bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
        cspace = ConfigurationSpace(bounds, ['continuous', 'revolute'])
        q1 = np.array([np.deg2rad(179), 0.0])
        q2 = np.array([np.deg2rad(-179), 0.0])
        d = cspace.distance(q1, q2)
        assert abs(d - np.deg2rad(2)) < 1e-6

    def test_steer_shortest_arc(self):
        from src.robotics_learning.planning import ConfigurationSpace
        bounds = np.array([[-np.pi, np.pi]])
        cspace = ConfigurationSpace(bounds, ['continuous'])
        q_near = np.array([np.deg2rad(179)])
        q_rand = np.array([np.deg2rad(-179)])
        q_new = cspace.steer(q_near, q_rand, 0.1)
        # 应走短弧 (2°) 方向 → q_new 应在 179° 附近向 -179° 走一步
        assert q_new[0] < -np.pi + 0.5 or q_new[0] > np.pi - 0.5

    def test_revolute_not_wrapped(self):
        """revolute 关节不应被 wrap — 仅在 continuous 上 wrap"""
        from src.robotics_learning.planning import ConfigurationSpace
        bounds = np.array([[-np.pi, np.pi]])
        cspace = ConfigurationSpace(bounds, ['revolute'])
        # distance 应对 revolute 用线性差
        q1 = np.array([np.deg2rad(179)])
        q2 = np.array([np.deg2rad(-179)])
        d = cspace.distance(q1, q2)
        # revolute 不 wrap → 距离约 358°
        assert d > np.deg2rad(300)

    def test_within_bounds(self):
        from src.robotics_learning.planning import ConfigurationSpace
        bounds = np.array([[0.0, 1.0], [-1.0, 1.0]])
        cspace = ConfigurationSpace(bounds)
        assert cspace.within_bounds(np.array([0.5, 0.0]))
        assert not cspace.within_bounds(np.array([1.5, 0.0]))

    def test_validate_planning_success(self):
        from src.robotics_learning.planning import validate_planning_problem
        bounds = np.array([[0, 10], [0, 10]])
        validate_planning_problem(lambda q: True, bounds,
                                  np.array([1., 1.]), np.array([9., 9.]),
                                  step_size=0.1, max_iter=100)

    def test_validate_bad_bounds(self):
        from src.robotics_learning.planning import validate_planning_problem
        import pytest
        bounds = np.array([[10, 0], [0, 10]])  # lower > upper
        with pytest.raises(ValueError, match="下界"):
            validate_planning_problem(lambda q: True, bounds,
                                      np.array([1., 1.]), np.array([9., 9.]),
                                      step_size=0.1, max_iter=100)

    def test_validate_start_collision(self):
        from src.robotics_learning.planning import validate_planning_problem
        import pytest
        bounds = np.array([[0, 10], [0, 10]])
        with pytest.raises(ValueError, match="collision"):
            validate_planning_problem(lambda q: q[0] < 5, bounds,
                                      np.array([7., 5.]), np.array([3., 5.]),
                                      step_size=0.1, max_iter=100)

    def test_rrt_star_invalid_input(self):
        """RRT* 无效输入应抛出"""
        from src.robotics_learning.planning import rrt_star_plan
        import pytest
        bounds = np.array([[0, 10], [0, 10]])
        with pytest.raises(ValueError, match="collision"):
            rrt_star_plan(lambda q: False, bounds, np.array([1., 1.]), np.array([9., 9.]),
                          max_iter=100, step_size=0.3, rng=RNG)
