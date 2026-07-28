"""轨迹测试 — 梯形/多项式/样条边界条件 + TOPP 约束"""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.robotics_learning.trajectory import (
    trapezoidal_trajectory, quintic_trajectory, cubic_trajectory, via_point_trajectory
)


class TestTrapezoidal:
    def test_endpoints(self):
        t, q, _, _ = trapezoidal_trajectory(0.0, 2.0, 1.5, 2.0, 0.01)
        assert abs(q[0] - 0.0) < 1e-10
        assert abs(q[-1] - 2.0) < 1e-10

    def test_velocity_bound(self):
        t, q, qd, _ = trapezoidal_trajectory(0.0, 1.0, 1.0, 3.0, 0.01)
        assert np.max(np.abs(qd)) <= 1.0 + 1e-6

    def test_acceleration_bound(self):
        t, q, qd, qdd = trapezoidal_trajectory(0.0, 2.0, 1.5, 2.0, 0.01)
        assert np.max(np.abs(qdd)) <= 2.0 + 1e-6

    def test_triangle_profile(self):
        """位移太小，应是三角剖面"""
        t, q, qd, _ = trapezoidal_trajectory(0.0, 0.3, 2.0, 3.0, 0.01)
        # 三角剖面: v_max 达不到
        assert np.max(np.abs(qd)) < 2.0 - 0.1


class TestPolynomial:
    def test_quintic_endpoints(self):
        _, q, qd, qdd, _ = quintic_trajectory(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.01)
        assert abs(q[0] - 0.0) < 1e-10
        assert abs(q[-1] - 1.0) < 1e-10
        assert abs(qd[0]) < 1e-6
        assert abs(qd[-1]) < 1e-6

    def test_cubic_endpoints(self):
        _, q, qd, _ = cubic_trajectory(0.0, 1.0, 0.0, 0.0, 2.0, 0.01)
        assert abs(q[0] - 0.0) < 1e-10
        assert abs(q[-1] - 1.0) < 1e-10


class TestViaPoint:
    def test_spline_passes_points(self):
        via = np.array([[0., 0.], [0.5, 0.3], [1., 0.]])
        t_via = np.array([0., 1., 2.])
        _, q, _, _ = via_point_trajectory(via, t_via, 0.05)
        for i in range(3):
            assert np.allclose(q[int(t_via[i]/0.05)], via[i], atol=0.1)
