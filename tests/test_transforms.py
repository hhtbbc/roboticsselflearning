"""变换模块测试 — SO(3), SE(3), 四元数, 欧拉角"""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.robotics_learning.transforms import (
    rot_x, rot_y, rot_z, skew, unskew, is_valid_rotation,
    homogenous_transform, homogeneous_transform, inv_homogenous, transform_point,
    euler_zyx_to_rot, rot_to_euler_zyx,
    axis_angle_to_rot, rot_to_axis_angle,
    Quaternion, slerp, so3_exp, so3_log, se3_exp, se3_log,
    adjoint, adjoint_inv_transpose
)

RNG = np.random.RandomState(42)
ATOL = 1e-10


class TestRotationMatrix:
    """SO(3) 基础测试"""

    def test_identity(self):
        R = np.eye(3)
        assert is_valid_rotation(R)

    def test_basic_rotations(self):
        for theta in [0, np.pi/6, np.pi/2, np.pi, -np.pi/4]:
            for rot_fn in [rot_x, rot_y, rot_z]:
                R = rot_fn(theta)
                assert is_valid_rotation(R), f"{rot_fn.__name__}({theta}) failed"
                assert np.abs(np.linalg.det(R) - 1.0) < ATOL

    def test_rotation_composition(self):
        """R_z(θ₁)R_y(θ₂) 仍是 SO(3)"""
        R = rot_z(np.pi/3) @ rot_y(np.pi/4)
        assert is_valid_rotation(R)

    def test_not_rotation(self):
        """非正交矩阵应返回 False"""
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert not is_valid_rotation(A)

    def test_skew_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        assert np.allclose(S + S.T, np.zeros((3, 3)))

    def test_skew_unskew_roundtrip(self):
        v = RNG.randn(3)
        assert np.allclose(unskew(skew(v)), v)

    def test_skew_cross_product(self):
        v = RNG.randn(3); w = RNG.randn(3)
        assert np.allclose(skew(v) @ w, np.cross(v, w))


class TestHomogeneousTransform:
    """SE(3) 基础测试"""

    def test_identity(self):
        T = np.eye(4)
        T_inv = inv_homogenous(T)
        assert np.allclose(T_inv, np.eye(4))

    def test_inverse_roundtrip(self):
        for _ in range(10):
            R = axis_angle_to_rot(RNG.randn(3), RNG.uniform(0, np.pi))
            p = RNG.uniform(-5, 5, 3)
            T = homogenous_transform(R, p)
            T_inv = inv_homogenous(T)
            assert np.allclose(T @ T_inv, np.eye(4), atol=ATOL)
            assert np.allclose(T_inv @ T, np.eye(4), atol=ATOL)

    def test_transform_point(self):
        R = rot_z(np.pi/2)
        p = np.array([2.0, 1.0, 0.0])
        T = homogenous_transform(R, p)
        pt = np.array([1.0, 0.0, 0.0])
        result = transform_point(T, pt)
        expected = R @ pt + p
        assert np.allclose(result, expected)


class TestEulerAngles:
    """欧拉角转换测试"""

    def test_zyx_roundtrip(self):
        for _ in range(20):
            rpy = RNG.uniform(-np.pi, np.pi, 3)
            R = euler_zyx_to_rot(*rpy)
            rpy2 = rot_to_euler_zyx(R)
            R2 = euler_zyx_to_rot(*rpy2)
            assert np.allclose(R, R2, atol=1e-8), f"rpy={rpy}, rpy2={rpy2}"

    def test_gimbal_lock_near(self):
        """接近但不进入万向锁阈值：欧拉角往返应一致"""
        rpy = np.array([0.5, np.pi/2 - 0.01, 0.3])
        R = euler_zyx_to_rot(*rpy)
        rpy2 = rot_to_euler_zyx(R)
        R2 = euler_zyx_to_rot(*rpy2)
        assert np.allclose(R, R2, atol=1e-8)

    def test_gimbal_lock_rotation_consistency(self):
        """万向锁处：旋转矩阵对向量作用应一致"""
        R = euler_zyx_to_rot(0.8, np.pi/2, 0.3)
        rpy2 = rot_to_euler_zyx(R)
        R2 = euler_zyx_to_rot(*rpy2)
        # 提取的欧拉角不同，但旋转矩阵对向量的作用应等价
        v_test = np.array([1.0, -0.5, 2.0])
        assert np.allclose(R @ v_test, R2 @ v_test, atol=1e-10)

    def test_zero_rotation(self):
        R = euler_zyx_to_rot(0, 0, 0)
        assert np.allclose(R, np.eye(3))


class TestAxisAngle:
    """轴角表示测试"""

    def test_roundtrip(self):
        for _ in range(20):
            axis = RNG.randn(3); axis /= np.linalg.norm(axis)
            angle = RNG.uniform(0.1, np.pi * 0.99)
            R = axis_angle_to_rot(axis, angle)
            k, theta = rot_to_axis_angle(R)
            R2 = axis_angle_to_rot(k, theta)
            assert np.allclose(R, R2, atol=1e-8)

    def test_zero_rotation(self):
        R = axis_angle_to_rot(np.array([1, 0, 0]), 0.0)
        assert np.allclose(R, np.eye(3))

    def test_180_degree(self):
        """180° 旋转的特殊处理"""
        axis = np.array([0.0, 0.0, 1.0])
        R = axis_angle_to_rot(axis, np.pi)
        k, theta = rot_to_axis_angle(R)
        assert np.abs(theta - np.pi) < 1e-6 or np.abs(theta + np.pi) < 1e-6


class TestQuaternion:
    """四元数测试"""

    def test_unit_norm(self):
        q = Quaternion.from_axis_angle(RNG.randn(3), 0.5)
        n = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert np.abs(n - 1.0) < 1e-10

    def test_roundtrip_rot(self):
        for _ in range(20):
            R = axis_angle_to_rot(RNG.randn(3), RNG.uniform(0, np.pi))
            q = Quaternion.from_rot(R)
            R2 = q.to_rot()
            assert np.allclose(R, R2, atol=1e-8)

    def test_double_cover(self):
        """q 和 -q 表示同一旋转"""
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/3)
        q2 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        assert np.allclose(q1.to_rot(), q2.to_rot(), atol=1e-8)

    def test_slerp_endpoints(self):
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)
        q0 = slerp(q1, q2, 0.0)
        q1_res = slerp(q1, q2, 1.0)
        assert np.allclose(q0.to_rot(), q1.to_rot(), atol=1e-8)
        assert np.allclose(q1_res.to_rot(), q2.to_rot(), atol=1e-8)


class TestLieGroup:
    """SO(3) 李群测试"""

    def test_exp_log_roundtrip(self):
        for _ in range(20):
            omega = RNG.uniform(-2.0, 2.0, 3)
            while np.linalg.norm(omega) > np.pi - 0.5:
                omega = RNG.uniform(-1.0, 1.0, 3)
            R = so3_exp(omega)
            omega2 = so3_log(R)
            assert np.allclose(omega, omega2, atol=1e-6), \
                f"ω={omega} (|ω|={np.linalg.norm(omega):.3f}), ω2={omega2}"

    def test_log_180_degree(self):
        """精确 180° 旋转：so3_log 不应退化"""
        for axis in [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0.3,0.6,0.2]]:
            k = np.array(axis, dtype=float); k /= np.linalg.norm(k)
            R = axis_angle_to_rot(k, np.pi)
            omega = so3_log(R)
            # 恢复的旋转向量应和原始轴平行
            if np.linalg.norm(omega) > 1e-10:
                k2 = omega / np.linalg.norm(omega)
                assert abs(np.dot(k, k2)) > 0.999, \
                    f"axis={k}, recovered={k2}, dot={np.dot(k,k2):.4f}"
                assert abs(np.linalg.norm(omega) - np.pi) < 1e-6

    def test_log_near_180(self):
        """接近 180° (±1e-8)：应能稳定处理"""
        k = np.array([0.0, 0.0, 1.0])
        for angle in [np.pi - 1e-8, np.pi + 1e-8]:
            R = axis_angle_to_rot(k, angle)
            omega = so3_log(R)
            R2 = so3_exp(omega)
            assert np.allclose(R @ np.array([1,0,0]), R2 @ np.array([1,0,0]), atol=1e-6)

    def test_exp_zero(self):
        assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))

    def test_log_identity(self):
        assert np.allclose(so3_log(np.eye(3)), np.zeros(3))

    def test_exp_equals_rodrigues(self):
        omega = RNG.uniform(-2, 2, 3)
        theta = np.linalg.norm(omega)
        if theta > 1e-10:
            k = omega / theta
            R_rod = axis_angle_to_rot(k, theta)
            R_exp = so3_exp(omega)
            assert np.allclose(R_exp, R_rod, atol=1e-10)

    def test_se3_exp_translation(self):
        """纯平移 twist 的 SE(3) exp"""
        twist = np.array([0, 0, 0, 1.0, 2.0, 3.0])  # [ω=0; v]
        T = se3_exp(twist)
        assert np.allclose(T[:3, :3], np.eye(3))
        assert np.allclose(T[:3, 3], twist[3:])

    def test_se3_log_roundtrip(self):
        """se3_log(se3_exp(twist)) ≈ twist"""
        for _ in range(10):
            twist = RNG.uniform(-1, 1, 6)
            twist[:3] *= 0.5  # limit rotation
            T = se3_exp(twist)
            twist2 = se3_log(T)
            # 旋转部分应在 2π 等价内一致
            assert np.allclose(so3_exp(twist[:3]) @ np.array([1,0,0]),
                              so3_exp(twist2[:3]) @ np.array([1,0,0]), atol=1e-6)

    def test_adjoint_consistency(self):
        """Ad_{T1 T2} = Ad_{T1} Ad_{T2}"""
        R1 = axis_angle_to_rot(RNG.randn(3), 1.0)
        R2 = axis_angle_to_rot(RNG.randn(3), 0.5)
        T1 = homogeneous_transform(R1, RNG.uniform(-1, 1, 3))
        T2 = homogeneous_transform(R2, RNG.uniform(-1, 1, 3))
        Ad1 = adjoint(T1); Ad2 = adjoint(T2)
        Ad12 = adjoint(T1 @ T2)
        assert np.allclose(Ad1 @ Ad2, Ad12, atol=1e-10)

    def test_adjoint_power_invariance(self):
        """V_b^T F_b = V_s^T F_s (功率不变性)"""
        V_b = RNG.randn(6); F_b = RNG.randn(6)
        T = homogeneous_transform(axis_angle_to_rot(RNG.randn(3), 0.7), RNG.uniform(-1,1,3))
        V_s = adjoint(T) @ V_b
        F_s = adjoint_inv_transpose(T) @ F_b
        assert np.allclose(V_b @ F_b, V_s @ F_s, atol=1e-10)
