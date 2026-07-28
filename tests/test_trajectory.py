"""轨迹测试 — 梯形/多项式/样条边界条件 + TOPP 约束"""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.robotics_learning.trajectory import (
    trapezoidal_trajectory, quintic_trajectory, cubic_trajectory, via_point_trajectory,
    time_optimal_parameterization
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


class TestTOPP:
    """时间参数化测试"""

    def test_feasible_linear_path(self):
        """简单线性路径应生成可行 MVC"""
        def path(s):
            s_val = np.atleast_1d(s)[0]
            return np.array([s_val, 0.5*s_val])
        def path_deriv(s):
            return np.array([1.0, 0.5])
        s_grid = np.linspace(0, 1, 200)
        q_dot_limits = np.array([2.0, 3.0])
        q_ddot_limits = np.array([10.0, 12.0])

        s_dot_max, s_dot_opt = time_optimal_parameterization(
            path, path_deriv, s_grid, q_dot_limits, q_ddot_limits
        )
        # MVC 应处处为正
        assert np.all(s_dot_max >= 0)
        # 最优曲线应非负
        assert np.all(s_dot_opt >= -1e-12)
        # 不应超过 MVC
        assert np.all(s_dot_opt <= s_dot_max + 1e-10)

    def test_zero_derivative_constraint(self):
        """f'(s)=0 处不应使 MVC 发散"""
        def path(s):
            s_val = np.atleast_1d(s)[0]
            return np.array([s_val, 0.0])
        def path_deriv(s):
            return np.array([1.0, 0.0])
        s_grid = np.linspace(0, 1, 100)
        q_dot_limits = np.array([2.0, 5.0])
        q_ddot_limits = np.array([10.0, 20.0])

        s_dot_max, s_dot_opt = time_optimal_parameterization(
            path, path_deriv, s_grid, q_dot_limits, q_ddot_limits
        )
        # f'_2 = 0 应对 MVC 无约束 → MVC 由 f'_1 决定
        assert np.all(np.isfinite(s_dot_max))

    def test_boundary_velocity(self):
        """MVC 曲线应在边界处有限"""
        def path(s):
            s_val = np.atleast_1d(s)[0]
            return np.array([s_val, 0.0])
        def path_deriv(s):
            return np.array([1.0, 0.0])
        s_grid = np.linspace(0, 1, 100)
        q_dot_limits = np.array([3.0, 5.0])
        q_ddot_limits = np.array([10.0, 20.0])

        s_dot_max, s_dot_opt = time_optimal_parameterization(
            path, path_deriv, s_grid, q_dot_limits, q_ddot_limits
        )
        # 两端和中间都应是有限值
        assert np.isfinite(s_dot_max[0])
        assert np.isfinite(s_dot_max[-1])
        assert np.all(np.isfinite(s_dot_opt))
