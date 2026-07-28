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


class TestTOPP:
    """时间参数化测试 — 使用真实的 topp_forward_backward_parameterization"""

    def test_feasible_linear_path(self):
        """简单线性路径 (f''=0) 应满足约束"""
        from src.robotics_learning.trajectory import topp_forward_backward_parameterization
        n_s = 200
        s_vals = np.linspace(0, 1, n_s)
        n_dof = 2
        # 线性路径: q(s) = [s, 0.5s], f'=[1,0.5], f''=[0,0]
        f_vals = np.column_stack([s_vals, 0.5 * s_vals])
        fp_vals = np.tile(np.array([1.0, 0.5]), (n_s, 1))
        fpp_vals = np.zeros((n_s, n_dof))
        q_dot_limits = np.array([2.0, 3.0])
        q_ddot_limits = np.array([10.0, 12.0])

        s_dot_mvc, s_dot_fwd, s_dot_bwd, s_dot_approx, t_vals = \
            topp_forward_backward_parameterization(
                f_vals, fp_vals, fpp_vals, s_vals,
                q_dot_limits, q_ddot_limits,
                s_dot_start=0.0, s_dot_end=0.0
            )
        # MVC 应处处为正
        assert np.all(s_dot_mvc >= 0)
        # 近似曲线应非负且不超过 MVC
        assert np.all(s_dot_approx >= -1e-12)
        assert np.all(s_dot_approx <= s_dot_mvc + 1e-10)
        # 边界应为零
        assert abs(s_dot_approx[0]) < 1e-10
        assert abs(s_dot_approx[-1]) < 1e-10
        # 应完成积分并有正的总时间
        assert t_vals[-1] > 0

    def test_zero_derivative_constraint(self):
        """f'(s)=0 处不应使 MVC 发散"""
        from src.robotics_learning.trajectory import topp_forward_backward_parameterization
        n_s = 100
        s_vals = np.linspace(0, 1, n_s)
        # 路径: q₁=s, q₂=0 → fp₁=1, fp₂=0, fpp₁=fpp₂=0
        f_vals = np.column_stack([s_vals, np.zeros(n_s)])
        fp_vals = np.column_stack([np.ones(n_s), np.zeros(n_s)])
        fpp_vals = np.zeros((n_s, 2))
        q_dot_limits = np.array([2.0, 5.0])
        q_ddot_limits = np.array([10.0, 20.0])

        s_dot_mvc, _, _, s_dot_approx, t_vals = \
            topp_forward_backward_parameterization(
                f_vals, fp_vals, fpp_vals, s_vals,
                q_dot_limits, q_ddot_limits,
                s_dot_start=0.0, s_dot_end=0.0
            )
        # f'_2 = 0 应对 MVC 无约束 → MVC 由 f'_1 决定
        assert np.all(np.isfinite(s_dot_mvc))
        assert t_vals[-1] > 0

    def test_boundary_velocity_zero(self):
        """s_dot_start=0, s_dot_end=0 应满足"""
        from src.robotics_learning.trajectory import topp_forward_backward_parameterization
        n_s = 100
        s_vals = np.linspace(0, 1, n_s)
        f_vals = np.column_stack([s_vals, np.zeros(n_s)])
        fp_vals = np.column_stack([np.ones(n_s), np.zeros(n_s)])
        fpp_vals = np.zeros((n_s, 2))
        q_dot_limits = np.array([3.0, 5.0])
        q_ddot_limits = np.array([10.0, 20.0])

        s_dot_mvc, _, _, s_dot_approx, _ = \
            topp_forward_backward_parameterization(
                f_vals, fp_vals, fpp_vals, s_vals,
                q_dot_limits, q_ddot_limits,
                s_dot_start=0.0, s_dot_end=0.0
            )
        assert abs(s_dot_approx[0]) < 1e-10
        assert abs(s_dot_approx[-1]) < 1e-10
        assert np.all(np.isfinite(s_dot_approx))

    def test_infeasible_path_raises(self):
        """不可行路径应抛出 RuntimeError"""
        from src.robotics_learning.trajectory import topp_forward_backward_parameterization
        import pytest
        n_s = 50
        s_vals = np.linspace(0, 1, n_s)
        f_vals = np.column_stack([s_vals, s_vals])
        fp_vals = np.ones((n_s, 2))  # f' = [1, 1]
        # 大曲率路径 + 极严格的加速度约束 → 传播不可行
        fpp_vals = np.ones((n_s, 2)) * 100.0
        q_dot_limits = np.array([10.0, 10.0])
        q_ddot_limits = np.array([0.05, 0.05])

        with pytest.raises(RuntimeError):
            topp_forward_backward_parameterization(
                f_vals, fp_vals, fpp_vals, s_vals,
                q_dot_limits, q_ddot_limits,
                s_dot_start=0.0, s_dot_end=0.0
            )

    def test_acceleration_constraint_satisfied(self):
        """验证实际关节加速度满足约束"""
        from src.robotics_learning.trajectory import topp_forward_backward_parameterization
        n_s = 200
        s_vals = np.linspace(0, 1, n_s)
        ds = s_vals[1] - s_vals[0]
        # 线性路径 + 小二阶导: q(s)=[s, 0.5s], f'=[1,0.5], f''=[0.1*sin(2πs), 0]
        f_vals = np.column_stack([s_vals, 0.5 * s_vals])
        fp_vals = np.tile(np.array([1.0, 0.5]), (n_s, 1))
        fpp_vals = np.column_stack([0.1 * np.sin(2 * np.pi * s_vals), np.zeros(n_s)])
        q_dot_limits = np.array([3.0, 4.0])
        q_ddot_limits = np.array([20.0, 25.0])

        result = topp_forward_backward_parameterization(
            f_vals, fp_vals, fpp_vals, s_vals,
            q_dot_limits, q_ddot_limits,
            s_dot_start=0.0, s_dot_end=0.0
        )
        s_dot_mvc, s_dot_fwd, s_dot_bwd, s_dot_approx, t_vals = result

        # 区间加速度: s̈_k = (ṡ_{k+1}² - ṡ_k²) / (2 Δs)
        s_ddot = (s_dot_approx[1:]**2 - s_dot_approx[:-1]**2) / (2 * ds)

        # 关节速度: q̇ = f'(s) ṡ
        q_dot = fp_vals * s_dot_approx[:, None]
        # 关节加速度: q̈ = f'(s) s̈ + f''(s) ṡ²  (注意尺寸: s̈ 有 n_s-1 个点)
        q_ddot = (fp_vals[:-1] * s_ddot[:, None] +
                  fpp_vals[:-1] * s_dot_approx[:-1, None]**2)

        tol = 1e-6
        assert np.all(np.abs(q_dot) <= q_dot_limits[None, :] * 1.01 + tol), \
            f"Velocity violated: max|q̇|={np.max(np.abs(q_dot))}"
        assert np.all(np.abs(q_ddot) <= q_ddot_limits[None, :] * 1.01 + tol), \
            f"Acceleration violated: max|q̈|={np.max(np.abs(q_ddot))}"

    def test_input_validation(self):
        """输入验证应拒绝明显非法的参数"""
        from src.robotics_learning.trajectory import topp_forward_backward_parameterization
        import pytest

        n_s = 10
        s_vals = np.linspace(0, 1, n_s)
        fp = np.ones((n_s, 2))
        fpp = np.zeros((n_s, 2))
        fv = np.column_stack([s_vals, s_vals])

        # 非正速度上限
        with pytest.raises(ValueError, match="q_dot_limits"):
            topp_forward_backward_parameterization(
                fv, fp, fpp, s_vals,
                np.array([0.0, 1.0]), np.array([10.0, 10.0]))

        # s_dot_start 超过 MVC
        with pytest.raises(ValueError, match="MVC"):
            topp_forward_backward_parameterization(
                fv, fp, fpp, s_vals,
                np.array([1.0, 1.0]), np.array([10.0, 10.0]),
                s_dot_start=1e9, s_dot_end=0.0)
