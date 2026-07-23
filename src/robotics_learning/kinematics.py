"""
运动学 (Kinematics) — 正运动学、逆运动学、雅可比

包含：
- DH 参数与正运动学（FK）
- 几何法逆运动学（IK）
- 数值法 IK（雅可比伪逆 / DLS）
- 几何雅可比与解析雅可比
"""

import numpy as np
from typing import List, Tuple, Optional
from .transforms import rot_x, rot_z, homogenous_transform


# =============================================================================
# DH 参数与正运动学
# =============================================================================

def dh_transform(a: float, alpha: float, d: float, theta: float,
                 convention: str = 'standard') -> np.ndarray:
    """
    单连杆 DH 变换矩阵

    标准 DH（Craig 约定）: T_i = Rot_{z,θ_i} Trans_{z,d_i} Trans_{x,a_i} Rot_{x,α_i}
    改进 DH:                T_i = Rot_{x,α_{i-1}} Trans_{x,a_{i-1}} Rot_{z,θ_i} Trans_{z,d_i}

    参数:
        a: 连杆长度 (link length)
        alpha: 连杆扭转角 (link twist)
        d: 连杆偏置 (link offset)
        theta: 关节角 (joint angle)
        convention: 'standard' (标准DH/改进DH) 或 'modified' (经典DH/Craig)
    """
    if convention == 'standard':
        # 标准 DH: T = Rz(θ) Tz(d) Tx(a) Rx(α)
        T_rz_tz = np.eye(4)
        T_rz_tz[:3, :3] = rot_z(theta)
        T_rz_tz[:3, 3] = [0, 0, d]

        T_tx = np.eye(4)
        T_tx[:3, 3] = [a, 0, 0]

        T_rx = np.eye(4)
        T_rx[:3, :3] = rot_x(alpha)

        return T_rz_tz @ T_tx @ T_rx
    else:
        # 改进 DH（经典 DH / Craig）: T = Rx(α_{i-1}) Tx(a_{i-1}) Rz(θ_i) Tz(d_i)
        T_rx = np.eye(4)
        T_rx[:3, :3] = rot_x(alpha)

        T_tx = np.eye(4)
        T_tx[:3, 3] = [a, 0, 0]

        T_rz = rot_z(theta)
        T_rz_h = np.eye(4)
        T_rz_h[:3, :3] = T_rz

        T_tz = np.eye(4)
        T_tz[:3, 3] = [0, 0, d]

        return T_rx @ T_tx @ T_rz_h @ T_tz


def forward_kinematics(dh_table: np.ndarray,
                       convention: str = 'standard') -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    正运动学计算

    参数:
        dh_table: (n, 4) 数组，每行 [a, alpha, d, theta]（θ 对旋转关节是变量）
        convention: 'standard' 或 'modified'
    返回:
        T_end: 4×4 末端位姿矩阵
        transforms: 每个连杆的变换矩阵列表
    """
    T = np.eye(4)
    transforms = [T.copy()]

    for params in dh_table:
        a, alpha, d, theta = params
        T_i = dh_transform(a, alpha, d, theta, convention)
        T = T @ T_i
        transforms.append(T.copy())

    return T, transforms


# =============================================================================
# 逆运动学 — 几何法（2R 平面臂）
# =============================================================================

def ik_2r_geometric(l1: float, l2: float, x: float, y: float) -> List[Tuple[float, float]]:
    """
    2R 平面机械臂的几何法 IK

    参数:
        l1, l2: 连杆长度
        x, y: 目标末端位置
    返回:
        解列表，每解为 (θ1, θ2)
    """
    r = np.sqrt(x**2 + y**2)

    # 不可达
    if r > l1 + l2 + 1e-10 or r < abs(l1 - l2) - 1e-10:
        return []

    # 用余弦定理求 θ2
    cos_theta2 = np.clip((r**2 - l1**2 - l2**2) / (2 * l1 * l2), -1, 1)
    theta2_1 = np.arccos(cos_theta2)   # elbow down
    theta2_2 = -theta2_1                # elbow up

    solutions = []
    for theta2 in [theta2_1, theta2_2]:
        # tan⁻¹(y/x) − tan⁻¹(l₂ sin θ₂ / (l₁ + l₂ cos θ₂))
        psi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = np.arctan2(y, x) - psi
        solutions.append((theta1, theta2))

    return solutions


# =============================================================================
# 逆运动学 — 数值法
# =============================================================================

def ik_numerical(dh_table: np.ndarray, T_des: np.ndarray,
                 q_init: np.ndarray, max_iter: int = 100,
                 tol: float = 1e-6, damping: float = 0.0) -> Optional[np.ndarray]:
    """
    数值法 IK（基于雅可比伪逆 / 阻尼最小二乘）

    参数:
        dh_table: DH 参数表
        T_des: 目标位姿 4×4
        q_init: 初始关节角
        max_iter: 最大迭代次数
        tol: 收敛容差
        damping: 阻尼因子（>0 启用 DLS）
    返回:
        q: 解关节角，不收敛时返回 None
    """
    q = q_init.copy()
    n = len(q)

    for _ in range(max_iter):
        # 计算当前 FK
        T_curr, _ = forward_kinematics(
            np.column_stack([dh_table[:, :3], q])
        )

        # 位姿误差
        p_err = T_des[:3, 3] - T_curr[:3, 3]
        R_err = T_des[:3, :3] @ T_curr[:3, :3].T
        omega_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        error = np.concatenate([p_err, omega_err])

        if np.linalg.norm(error) < tol:
            return q

        # 雅可比
        J = compute_geometric_jacobian(dh_table, q)

        # 更新关节角
        if damping > 0:
            # 阻尼最小二乘 (DLS)
            JJT = J @ J.T
            delta_q = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), error)
        else:
            # 雅可比伪逆
            J_pinv = np.linalg.pinv(J)
            delta_q = J_pinv @ error

        q = q + delta_q

    return None  # 不收敛


# =============================================================================
# 几何雅可比 (Geometric/Manipulator Jacobian)
# =============================================================================

def compute_geometric_jacobian(dh_table: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    逐列构造几何雅可比矩阵 J(q) ∈ ℝ^{6×n}

    对于旋转关节 i：
        J_v_i = z_{i-1} × (p_n − p_{i-1})
        J_ω_i = z_{i-1}

    参数:
        dh_table: (n,3) [a, alpha, d] 不含 θ
        q: (n,) 关节角
    返回:
        J: 6×n 几何雅可比 [J_v; J_ω]
    """
    n = len(q)
    dh_full = np.column_stack([dh_table, q])

    # 计算所有变换
    T, transforms = forward_kinematics(dh_full)

    p_n = T[:3, 3]  # 末端位置
    J = np.zeros((6, n))

    for i in range(n):
        T_prev = transforms[i]  # T_{i-1}
        z_i = T_prev[:3, 2]     # z_{i-1} (关节轴)
        p_i = T_prev[:3, 3]     # p_{i-1}

        J[:3, i] = np.cross(z_i, p_n - p_i)  # 线速度部分
        J[3:, i] = z_i                       # 角速度部分

    return J


def compute_analytical_jacobian(dh_table: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    解析雅可比（通过差分 FK 计算）

    解析雅可比将关节速度映射到最小姿态表示的导数（如欧拉角速率）。
    J_A 与几何雅可比的关系: J = B_A(φ) J_A，其中 B_A 是变换矩阵。

    参数:
        dh_table, q: 同 compute_geometric_jacobian
    返回:
        J_A: 6×n 解析雅可比
    """
    n = len(q)
    eps = 1e-6

    def fk_pose(dh_table, q):
        T, _ = forward_kinematics(np.column_stack([dh_table, q]))
        p = T[:3, 3]
        # 用 ZYX 欧拉角作为最小表示
        from .transforms import rot_to_euler_zyx
        rpy = rot_to_euler_zyx(T[:3, :3])
        return np.concatenate([p, rpy])

    f0 = fk_pose(dh_table, q)
    J_A = np.zeros((6, n))

    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += eps
        fi = fk_pose(dh_table, q_plus)
        J_A[:, i] = (fi - f0) / eps

    return J_A
