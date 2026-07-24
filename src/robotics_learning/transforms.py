"""
位姿变换与旋转表示 (Transforms & Rotation Representations)

包含：
- 旋转矩阵（SO(3)）
- 齐次变换矩阵（SE(3)）
- 欧拉角（多种约定）
- 轴角表示（Axis-Angle）
- 四元数（Quaternion）
- SO(3)/SE(3) 的指数映射与对数映射
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 旋转矩阵 (Rotation Matrix)
# =============================================================================

def rot_x(theta: float) -> np.ndarray:
    """绕 X 轴的旋转矩阵（3×3）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    """绕 Y 轴的旋转矩阵（3×3）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rot_z(theta: float) -> np.ndarray:
    """绕 Z 轴的旋转矩阵（3×3）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """验证矩阵是否为合法旋转矩阵（SO(3)）"""
    if R.shape != (3, 3):
        return False
    ortho = np.allclose(R @ R.T, np.eye(3), atol=tol)
    det = np.abs(np.linalg.det(R) - 1.0) < tol
    return ortho and det


def skew(v: np.ndarray) -> np.ndarray:
    """构造叉乘矩阵（skew-symmetric matrix / hat operator）"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def unskew(S: np.ndarray) -> np.ndarray:
    """从叉乘矩阵恢复向量（vee operator）"""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# =============================================================================
# 齐次变换矩阵 (Homogeneous Transformation, SE(3))
# =============================================================================

def homogenous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """构造齐次变换矩阵 T = [R p; 0 1] ∈ SE(3)"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_homogenous(T: np.ndarray) -> np.ndarray:
    """齐次变换矩阵的逆"""
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """用齐次变换矩阵变换点（dim 3 或 4）"""
    if len(p) == 3:
        p_h = np.append(p, 1)
    else:
        p_h = p
    p_transformed = T @ p_h
    return p_transformed[:3]


# =============================================================================
# 欧拉角 (Euler Angles)
# =============================================================================

def euler_zyx_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角（Tait-Bryan）→ 旋转矩阵 R = R_z(yaw) R_y(pitch) R_x(roll)"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """旋转矩阵 → ZYX 欧拉角（Tait-Bryan）"""
    # 处理万向锁附近的情况
    if np.abs(R[2, 0]) > 0.99999:
        # Gimbal lock: pitch = ±π/2, roll 和 yaw 耦合
        # 令 yaw = 0，从矩阵元素恢复 roll
        yaw = 0.0
        if R[2, 0] < 0:
            # pitch = +π/2: R[0,1] = sin(roll - yaw), R[0,2] = cos(roll - yaw)
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            # pitch = -π/2: R[0,1] = -sin(roll + yaw), R[0,2] = -cos(roll + yaw)
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_zyz_to_rot(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYZ 欧拉角 → 旋转矩阵 R = R_z(phi) R_y(theta) R_z(psi)"""
    return rot_z(phi) @ rot_y(theta) @ rot_z(psi)


# =============================================================================
# 轴角表示 (Axis-Angle)
# =============================================================================

def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    轴角 → 旋转矩阵（罗德里格斯公式 Rodrigues' Formula）

    R = I + sin(θ) [k]_× + (1 − cos(θ)) [k]_×²
    """
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def rot_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """旋转矩阵 → 轴角"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if np.abs(theta) < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    elif np.abs(theta - np.pi) < 1e-6:
        # 180度旋转的特殊处理: R + I = 2 k k^T (k 是旋转轴)
        # 取 (R + I) 的任意非零列归一化
        RpI = R + np.eye(3)
        for col in range(3):
            k = RpI[:, col]
            if np.linalg.norm(k) > 1e-10:
                return k / np.linalg.norm(k), theta
        # 退化情况：R = I
        return np.array([1.0, 0.0, 0.0]), 0.0
    else:
        k = np.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
        return k, theta


# =============================================================================
# 四元数 (Quaternion) — 类定义
# =============================================================================

class Quaternion:
    """四元数 q = w + xi + yj + zk = (w, [x, y, z])"""

    __slots__ = ('w', 'x', 'y', 'z')

    def __init__(self, w: float = 1.0, x: float = 0.0,
                 y: float = 0.0, z: float = 0.0):
        self.w, self.x, self.y, self.z = w, x, y, z

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """从轴角构造四元数"""
        axis = axis / np.linalg.norm(axis)
        half = angle / 2
        return cls(np.cos(half),
                   axis[0] * np.sin(half),
                   axis[1] * np.sin(half),
                   axis[2] * np.sin(half))

    @classmethod
    def from_rot(cls, R: np.ndarray) -> 'Quaternion':
        """从旋转矩阵构造四元数"""
        axis, angle = rot_to_axis_angle(R)
        return cls.from_axis_angle(axis, angle)

    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    def to_rot(self) -> np.ndarray:
        """四元数 → 旋转矩阵"""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
        ])

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> 'Quaternion':
        """单位四元数的逆 = 共轭"""
        return self.conjugate()

    def normalize(self) -> 'Quaternion':
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        return Quaternion(self.w/norm, self.x/norm, self.y/norm, self.z/norm)

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """用四元数旋转向量 (q ⊗ v ⊗ q*)"""
        q_v = Quaternion(0, v[0], v[1], v[2])
        q_result = self * q_v * self.conjugate()
        return np.array([q_result.x, q_result.y, q_result.z])

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """四元数乘法（Hamilton 约定）"""
        return Quaternion(
            self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        )

    def __repr__(self):
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """球面线性插值 (Spherical Linear Interpolation)"""
    # 确保走最短路径
    cos_omega = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
    if cos_omega < 0:
        q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
        cos_omega = -cos_omega

    if cos_omega > 0.9999:
        # 线性插值近似
        return Quaternion(
            q1.w + t*(q2.w - q1.w),
            q1.x + t*(q2.x - q1.x),
            q1.y + t*(q2.y - q1.y),
            q1.z + t*(q2.z - q1.z)
        ).normalize()

    omega = np.arccos(cos_omega)
    sin_omega = np.sin(omega)
    a = np.sin((1-t)*omega) / sin_omega
    b = np.sin(t*omega) / sin_omega
    return Quaternion(
        a*q1.w + b*q2.w, a*q1.x + b*q2.x,
        a*q1.y + b*q2.y, a*q1.z + b*q2.z
    )


# =============================================================================
# SO(3) / SE(3) 指数映射与对数映射
# =============================================================================

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    so(3) → SO(3) 指数映射（罗德里格斯公式形式）

    参数:
        omega: 3×1 旋转向量 (axis × angle)
    返回:
        R: 3×3 旋转矩阵
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    axis = omega / theta
    return axis_angle_to_rot(axis, theta)


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    SO(3) → so(3) 对数映射

    返回:
        omega: 3×1 旋转向量
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-10:
        return np.zeros(3)
    ln_R = theta / (2 * np.sin(theta)) * (R - R.T)
    return np.array([ln_R[2, 1], ln_R[0, 2], ln_R[1, 0]])


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """
    se(3) → SE(3) 指数映射

    参数:
        twist: 6×1 [v; omega] (线速度部分 + 角速度部分)
    返回:
        T: 4×4 齐次变换矩阵
    """
    v = twist[:3]
    omega = twist[3:]
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        return homogenous_transform(np.eye(3), v)

    axis = omega / theta
    K = skew(axis)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    V = (np.eye(3) + (1 - np.cos(theta)) / theta * K
         + (1 - np.sin(theta) / theta) * (K @ K))
    p = V @ v

    return homogenous_transform(R, p)


def adjoint(T: np.ndarray) -> np.ndarray:
    """计算 SE(3) 的 Adjoint 矩阵 Ad_T ∈ R^{6×6}。

    Ad_T 将物体系 twist 映射到空间系: V_s = Ad_T V_b。

    Ad_T = [R,     [p]×R ]
           [0,      R    ]
    """
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew(p) @ R
    return Ad


def adjoint_inv_transpose(T: np.ndarray) -> np.ndarray:
    """计算 Ad_T^{-T}，用于 wrench 变换: F_s = Ad_T^{-T} F_b。

    Ad_T^{-T} = [R,         0    ]
                [[p]× R,    R    ]
    """
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_invT = np.zeros((6, 6))
    Ad_invT[:3, :3] = R
    Ad_invT[3:, :3] = skew(p) @ R
    Ad_invT[3:, 3:] = R
    return Ad_invT
