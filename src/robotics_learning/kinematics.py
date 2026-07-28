"""
运动学 (Kinematics) — 正运动学、逆运动学、雅可比

包含：
- DH 参数与正运动学（FK）
- 几何法逆运动学（IK）
- 数值法 IK（雅可比伪逆 / DLS）
- 几何雅可比与解析雅可比
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from .transforms import rot_x, rot_z


@dataclass
class IKResult:
    """数值 IK 的结构化返回结果"""
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    reason: str = ""


# =============================================================================
# DH 参数与正运动学
# =============================================================================

def dh_transform(a: float, alpha: float, d: float, theta: float,
                 convention: str = 'sdh') -> np.ndarray:
    """
    单连杆 DH 变换矩阵

    支持两种 DH 约定：

    SDH (Standard DH):
        T_i = R_z(θ_i) · T_z(d_i) · T_x(a_i) · R_x(α_i)
        参数下标均为 i。
        这是 Spong/Vidyasagar 等教材使用的约定。

    MDH (Modified DH / Khalil-Kleinfinger):
        T_i = R_x(α_{i-1}) · T_x(a_{i-1}) · R_z(θ_i) · T_z(d_i)
        参数下标: a_{i-1}, α_{i-1}, d_i, θ_i。
        这是 Craig 教材实际使用的约定（Craig 称之为 Modified DH）。
        注意: URDF 直接描述父子 Link 间的固定变换和关节轴，
        并不等同于 MDH。DH 模型转 URDF 通常需要额外坐标系或固定关节。

    注意：不同的教材可能对"标准 DH"和"改进 DH"使用不同的名称。
    本课程以矩阵乘法顺序为准，不使用模糊的"经典DH"等命名。

    参数:
        a: 连杆长度 (link length) — SDH: a_i, MDH: a_{i-1}
        alpha: 连杆扭转角 (link twist) — SDH: α_i, MDH: α_{i-1}
        d: 连杆偏置 (link offset) — SDH: d_i, MDH: d_i
        theta: 关节角 (joint angle) — SDH: θ_i, MDH: θ_i
        convention: 'sdh' (Standard DH) 或 'mdh' (Modified DH)
    """
    if convention == 'sdh':
        # SDH: T = Rz(θ_i) Tz(d_i) Tx(a_i) Rx(α_i)
        T = np.eye(4)
        T[:3, :3] = rot_z(theta)
        T[:3, 3] = [0, 0, d]
        T_rz_tz = T

        T_tx = np.eye(4)
        T_tx[:3, 3] = [a, 0, 0]

        T_rx = np.eye(4)
        T_rx[:3, :3] = rot_x(alpha)

        return T_rz_tz @ T_tx @ T_rx
    elif convention == 'mdh':
        # MDH: T = Rx(α_{i-1}) Tx(a_{i-1}) Rz(θ_i) Tz(d_i)
        T_rx = np.eye(4)
        T_rx[:3, :3] = rot_x(alpha)

        T_tx = np.eye(4)
        T_tx[:3, 3] = [a, 0, 0]

        T_rz = np.eye(4)
        T_rz[:3, :3] = rot_z(theta)

        T_tz = np.eye(4)
        T_tz[:3, 3] = [0, 0, d]

        return T_rx @ T_tx @ T_rz @ T_tz
    else:
        raise ValueError(f"Unknown DH convention: '{convention}'. Use 'sdh' or 'mdh'.")


def forward_kinematics(dh_table: np.ndarray,
                       convention: str = 'sdh') -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    正运动学计算

    参数:
        dh_table: (n, 4) 数组，每行 [a, alpha, d, theta]（θ 对旋转关节是变量）
        convention: 'sdh' (Standard DH) 或 'mdh' (Modified DH)
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
                 position_tol: float = 1e-4, orientation_tol: float = 1e-4,
                 damping: float = 0.0,
                 max_step: float = 0.5) -> IKResult:
    """
    数值法 IK（基于雅可比伪逆 / 阻尼最小二乘）

    使用 SO(3) Log 提取姿态误差，对 180° 旋转也有效。

    参数:
        dh_table: DH 参数表 (n,3) [a, alpha, d]
        T_des: 目标位姿 4×4
        q_init: 初始关节角
        max_iter: 最大迭代次数
        position_tol: 位置容差 (m)
        orientation_tol: 姿态容差 (rad)
        damping: 阻尼因子（>0 启用 DLS）
        max_step: 最大单步关节增量 (rad)
    返回:
        IKResult 包含解关节角、是否成功、迭代次数、误差和失败原因
    """
    q = q_init.copy()

    for iteration in range(max_iter):
        # 计算当前 FK
        T_curr, _ = forward_kinematics(
            np.column_stack([dh_table[:, :3], q])
        )

        # 位置误差 (m)
        p_err = T_des[:3, 3] - T_curr[:3, 3]
        # 姿态误差 (rad) — SO(3) Log map
        R_err_mat = T_des[:3, :3] @ T_curr[:3, :3].T
        from .transforms import so3_log
        omega_err = so3_log(R_err_mat)

        pos_err_norm = np.linalg.norm(p_err)
        rot_err_norm = np.linalg.norm(omega_err)

        if pos_err_norm < position_tol and rot_err_norm < orientation_tol:
            return IKResult(q=q, success=True, iterations=iteration + 1,
                           position_error=pos_err_norm,
                           orientation_error=rot_err_norm,
                           reason="Converged")

        error = np.concatenate([p_err, omega_err])

        # 经典几何雅可比
        J = compute_geometric_jacobian(dh_table, q)

        if damping > 0:
            # DLS: Δq = J^T (J J^T + λ² I)^{-1} e
            JJT = J @ J.T
            delta_q = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(6), error)
        else:
            # 伪逆
            J_pinv = np.linalg.pinv(J)
            delta_q = J_pinv @ error

        # 最大单步关节增量限制
        step_norm = np.max(np.abs(delta_q))
        if step_norm > max_step:
            delta_q = delta_q * max_step / step_norm

        q = q + delta_q

    return IKResult(q=q, success=False, iterations=max_iter,
                   position_error=np.linalg.norm(p_err),
                   orientation_error=np.linalg.norm(omega_err),
                   reason="Max iterations exceeded")


# =============================================================================
# 几何雅可比 (Geometric/Manipulator Jacobian)
# =============================================================================

def compute_geometric_jacobian(dh_table: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    逐列构造经典几何雅可比（末端点速度 + 角速度）。

    上半部分 J_v 映射关节速度到**末端点的实际线速度** ṗ_E = J_v q̇。
    下半部分 J_ω 映射关节速度到末端角速度 ω_E = J_ω q̇。

    这是经典教材 (Craig, Siciliano) 中的几何雅可比，
    不是 Lie 群中的 Spatial Jacobian（后者的线速度部分是 v_s ≠ ṗ_E）。

    对于旋转关节 i：
        J_v_i = z_{i-1} × (p_n − p_{i-1})     （产生末端点速度）
        J_ω_i = z_{i-1}                        （产生末端角速度）

    参数:
        dh_table: (n,3) [a, alpha, d] 不含 θ
        q: (n,) 关节角
    返回:
        J: 6×n [J_v; J_ω] — 上半 3 行 = 线速度, 下半 3 行 = 角速度
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


def compute_pose_parameter_jacobian_fd(dh_table: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    通过 FK 有限差分和欧拉角提取的姿态参数雅可比。

    此函数将关节速度映射到最小姿态表示（ZYX 欧拉角）的导数。
    注意这是有限差分近似，会受到欧拉角 ±π 跳变和万向锁影响。

    如需解析形式: J_A = diag(I, E^{-1}(φ)) · J_G,
    其中 E(φ) 将欧拉角速率映射到角速度 (ω = E(φ) φ̇)。

    参数:
        dh_table: (n,3) DH 参数
        q: (n,) 关节角
    返回:
        J_A: 6×n 姿态参数雅可比 (有限差分)
    """
    n = len(q)
    eps = 1e-6

    def fk_pose(dh_table, q):
        T, _ = forward_kinematics(np.column_stack([dh_table, q]))
        p = T[:3, 3]
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


# 向后兼容别名
compute_analytical_jacobian = compute_pose_parameter_jacobian_fd


def compute_space_jacobian_poe(screw_axes: list, q: np.ndarray) -> np.ndarray:
    """
    基于螺旋轴递推计算 Lie 群空间雅可比 (Space Jacobian)。

    遵循 Modern Robotics 约定: twist = [ω; v] (角速度在前)。

    J_s 的第 i 列 = Ad_{e^{[S_1]q_1} ... e^{[S_{i-1}]q_{i-1}}} S_i

    参数:
        screw_axes: n 个螺旋轴列表，每个 S_i ∈ R^6 = [ω_i; v_i]
        q: (n,) 关节角
    返回:
        J_s: 6×n 空间雅可比
    """
    from .transforms import se3_exp, adjoint
    n = len(q)
    J_s = np.zeros((6, n))
    T_cum = np.eye(4)

    for i in range(n):
        S_i = screw_axes[i]
        Ad_cum = adjoint(T_cum)
        J_s[:, i] = Ad_cum @ S_i
        T_cum = T_cum @ se3_exp(S_i * q[i])

    return J_s
