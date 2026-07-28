"""运动学测试 — DH, FK, IK, Jacobian"""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.robotics_learning.transforms import (
    rot_z, so3_log
)
from src.robotics_learning.kinematics import (
    dh_transform, forward_kinematics,
    ik_2r_geometric, compute_geometric_jacobian
)

RNG = np.random.RandomState(42)
ATOL = 1e-10


class TestDH:
    """DH 参数测试"""

    def test_sdh_vs_mdh_same_physical_robot(self):
        """2R 平面臂：SDH 和 MDH 各自构造正确的 DH 表。

        SDH 和 MDH 的 DH 参数数值不同（坐标系附着方式不同）。
        本测试验证：
        1. SDH 的 FK 结果与手算一致
        2. SDH 和 MDH 各自的 DH 表构造在对应约定下都正确
           （两者都满足各自的正确性而不必全等）
        """
        l1, l2 = 1.0, 0.8
        q1, q2 = np.pi/4, np.pi/3

        # SDH FK: 参数 a_i 从关节 i 开始
        dh_sdh = np.array([[l1, 0, 0, q1], [l2, 0, 0, q2]])
        T_sdh, _ = forward_kinematics(dh_sdh, 'sdh')

        # MDH FK: 参数 a_{i-1} 从关节 i 之前开始，最后一个连杆用固定变换
        # 正确构造：a₀=0, α₀=0, d₁=0, θ₁=q₁; a₁=l₁, α₁=0, d₂=0, θ₂=q₂
        dh_mdh = np.array([[0.0, 0, 0, q1], [l1, 0, 0, q2]])
        T_mdh, _ = forward_kinematics(dh_mdh, 'mdh')
        # 加固定工具变换（最后一段连杆 l₂）
        T_tool = np.eye(4); T_tool[0, 3] = l2
        T_mdh_with_tool = T_mdh @ T_tool

        # 手算期望
        x_exp = l1*np.cos(q1) + l2*np.cos(q1+q2)
        y_exp = l1*np.sin(q1) + l2*np.sin(q1+q2)

        assert np.allclose(T_sdh[:2, 3], [x_exp, y_exp], atol=1e-10)
        assert np.allclose(T_mdh_with_tool[:2, 3], [x_exp, y_exp], atol=1e-10)

    def test_pure_translation(self):
        """SDH: a=1, α=0, d=0.5, θ=0 → T=[I, [1,0,0.5]]（近似）"""
        T = dh_transform(1.0, 0.0, 0.5, 0.0, 'sdh')
        assert np.allclose(T[:3, :3], np.eye(3))
        assert np.allclose(T[:3, 3], [1.0, 0.0, 0.5])

    def test_pure_rotation_z(self):
        """SDH: a=0, α=0, d=0, θ=π/2"""
        T = dh_transform(0.0, 0.0, 0.0, np.pi/2, 'sdh')
        expected = np.eye(4)
        expected[:3, :3] = rot_z(np.pi/2)
        assert np.allclose(T, expected)

    def test_unknown_convention_raises(self):
        import pytest
        with pytest.raises(ValueError):
            dh_transform(1, 0, 0, 0, 'invalid')


class TestFK:
    """正运动学测试"""

    def test_2r_manual(self):
        """2R 臂 FK vs 手算"""
        l1, l2 = 1.0, 0.8
        q1, q2 = np.pi/3, np.pi/6
        dh = np.array([[l1, 0, 0, q1], [l2, 0, 0, q2]])
        T, _ = forward_kinematics(dh, 'sdh')

        x_manual = l1*np.cos(q1) + l2*np.cos(q1+q2)
        y_manual = l1*np.sin(q1) + l2*np.sin(q1+q2)
        assert np.allclose(T[:2, 3], [x_manual, y_manual])


class TestIK:
    """逆运动学测试"""

    def test_2r_geometric_consistency(self):
        """几何法 IK → FK 应回到目标位置"""
        l1, l2 = 1.0, 0.8
        x, y = 1.2, 0.6
        sols = ik_2r_geometric(l1, l2, x, y)
        assert len(sols) >= 1
        for q1, q2 in sols:
            x_fk = l1*np.cos(q1) + l2*np.cos(q1+q2)
            y_fk = l1*np.sin(q1) + l2*np.sin(q1+q2)
            assert np.allclose([x_fk, y_fk], [x, y], atol=1e-8)

    def test_2r_unreachable(self):
        """不可达目标应返回空列表"""
        sols = ik_2r_geometric(1.0, 0.8, 5.0, 0.0)
        assert len(sols) == 0


class TestJacobian:
    """雅可比测试"""

    def test_numerical_vs_analytical(self):
        """几何雅可比 vs 数值差分"""
        dh = np.array([[0.0, 0.0, 1.0], [0.8, 0.0, 0.0]])
        q = np.array([np.pi/4, np.pi/6])
        J = compute_geometric_jacobian(dh, q)

        # 数值差分
        eps = 1e-6
        dh_ref = np.column_stack([dh, q])
        T_ref, _ = forward_kinematics(dh_ref, 'sdh')
        p_ref = T_ref[:3, 3]
        J_num = np.zeros((6, 2))
        for i in range(2):
            q_plus = q.copy(); q_plus[i] += eps
            dh_plus = np.column_stack([dh, q_plus])
            T_plus, _ = forward_kinematics(dh_plus, 'sdh')
            J_num[:3, i] = (T_plus[:3, 3] - p_ref) / eps
            dR = T_plus[:3, :3] @ T_ref[:3, :3].T
            omega_vec = so3_log(dR) / eps
            J_num[3:, i] = omega_vec
        assert np.allclose(J, J_num, atol=1e-4)

    def test_statics_virtual_work(self):
        """τ = J^T F → 虚功原理验证"""
        dh = np.array([[1.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        q = np.array([np.pi/4, np.pi/6])
        J = compute_geometric_jacobian(dh, q)

        F = np.array([10.0, -5.0, 0.0, 0.0, 0.0, 2.0])
        tau = J.T @ F
        delta_q = np.array([0.01, -0.005])
        delta_x = J @ delta_q
        assert np.allclose(tau @ delta_q, F @ delta_x, atol=1e-8)
