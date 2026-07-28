"""
轨迹生成 (Trajectory Generation)

包含：
- 梯形速度曲线 (Trapezoidal Velocity Profile)
- 多项式轨迹（三次、五次、七次）
- 样条轨迹
- 时间参数化（TOPP 简化版）
"""

import numpy as np
from typing import Tuple, Callable
from scipy.interpolate import CubicSpline


# =============================================================================
# 梯形速度曲线
# =============================================================================

def trapezoidal_trajectory(q0: float, qf: float,
                           v_max: float, a_max: float,
                           dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    梯形速度曲线

    参数:
        q0, qf: 起点和终点位置
        v_max: 最大速度
        a_max: 最大加速度
        dt: 采样时间
    返回:
        t, q, q_dot, q_ddot
    """
    dq = abs(qf - q0)
    sign = np.sign(qf - q0)

    # 判断是否可以达到 v_max（对比三角剖面和梯形剖面）
    t_acc = v_max / a_max
    d_triangle = a_max * t_acc**2  # 三角剖面的位移

    if d_triangle <= dq:
        # 梯形剖面：加速 → 巡航 → 减速
        T = dq / v_max + v_max / a_max
        t1 = t_acc
        t2 = T - t_acc
    else:
        # 三角剖面：加速 → 立即减速
        t1 = np.sqrt(dq / a_max)
        t2 = t1
        T = 2 * t1
        v_max = a_max * t1  # 实际最大速度

    N = int(T / dt) + 1
    t = np.linspace(0, T, N)
    q = np.zeros(N)
    q_dot = np.zeros(N)
    q_ddot = np.zeros(N)

    for i, ti in enumerate(t):
        if ti < t1:
            # 加速段
            q_ddot[i] = a_max
            q_dot[i] = a_max * ti
            q[i] = 0.5 * a_max * ti**2
        elif ti < t2:
            # 巡航段
            q_ddot[i] = 0
            q_dot[i] = v_max
            q[i] = 0.5 * v_max * t1 + v_max * (ti - t1)
        else:
            # 减速段
            tau = T - ti
            q_ddot[i] = -a_max
            q_dot[i] = a_max * tau
            q[i] = dq - 0.5 * a_max * tau**2

    # 调整方向
    q = q0 + sign * q
    q_dot = sign * q_dot
    q_ddot = sign * q_ddot

    return t, q, q_dot, q_ddot


# =============================================================================
# 多项式轨迹
# =============================================================================

def quintic_trajectory(q0: float, qf: float,
                       v0: float, vf: float,
                       a0: float, af: float,
                       T: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    五次多项式轨迹

    q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵

    6 个边界条件: q(0), q(T), q̇(0), q̇(T), q̈(0), q̈(T)
    """
    # 解线性方程组求系数
    # [1, 0, 0,     0,      0,       0     ] [a0]   [q0]
    # [1, T, T²,    T³,     T⁴,      T⁵    ] [a1]   [qf]
    # [0, 1, 0,     0,      0,       0     ] [a2] = [v0]
    # [0, 1, 2T,    3T²,    4T³,     5T⁴   ] [a3]   [vf]
    # [0, 0, 2,     0,      0,       0     ] [a4]   [a0]
    # [0, 0, 2,     6T,     12T²,    20T³  ] [a5]   [af]

    A = np.array([
        [1, 0, 0,       0,        0,         0],
        [1, T, T**2,    T**3,     T**4,      T**5],
        [0, 1, 0,       0,        0,         0],
        [0, 1, 2*T,     3*T**2,   4*T**3,    5*T**4],
        [0, 0, 2,       0,        0,         0],
        [0, 0, 2,       6*T,      12*T**2,   20*T**3]
    ])
    b = np.array([q0, qf, v0, vf, a0, af])
    coeffs = np.linalg.solve(A, b)

    N = int(T / dt) + 1
    t = np.linspace(0, T, N)

    q = np.polyval(coeffs[::-1], t)
    q_dot = np.polyval(np.polyder(coeffs[::-1]), t)
    q_ddot = np.polyval(np.polyder(np.polyder(coeffs[::-1])), t)
    q_jerk = np.polyval(np.polyder(np.polyder(np.polyder(coeffs[::-1]))), t)

    return t, q, q_dot, q_ddot, q_jerk


def cubic_trajectory(q0: float, qf: float,
                     v0: float, vf: float,
                     T: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    三次多项式轨迹

    q(t) = a0 + a1*t + a2*t² + a3*t³
    4 个边界条件: q(0), q(T), q̇(0), q̇(T)
    """
    A = np.array([
        [1, 0, 0,     0],
        [1, T, T**2,  T**3],
        [0, 1, 0,     0],
        [0, 1, 2*T,   3*T**2]
    ])
    b = np.array([q0, qf, v0, vf])
    coeffs = np.linalg.solve(A, b)

    N = int(T / dt) + 1
    t = np.linspace(0, T, N)

    q = np.polyval(coeffs[::-1], t)
    q_dot = np.polyval(np.polyder(coeffs[::-1]), t)
    q_ddot = np.polyval(np.polyder(np.polyder(coeffs[::-1])), t)

    return t, q, q_dot, q_ddot


# =============================================================================
# 多路径点轨迹拼接
# =============================================================================

def via_point_trajectory(via_points: np.ndarray,
                         via_times: np.ndarray,
                         dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    通过多个路径点的三次样条轨迹（C² 连续）

    参数:
        via_points: (M, n_dof) 路径点
        via_times: (M,) 各点的时间
        dt: 采样时间
    返回:
        t, q, q_dot, q_ddot
    """
    M, n_dof = via_points.shape
    cs_list = []

    for d in range(n_dof):
        cs = CubicSpline(via_times, via_points[:, d],
                         bc_type='natural')
        cs_list.append(cs)

    T = via_times[-1]
    N = int(T / dt) + 1
    t = np.linspace(0, T, N)
    q = np.zeros((N, n_dof))
    q_dot = np.zeros((N, n_dof))
    q_ddot = np.zeros((N, n_dof))

    for d in range(n_dof):
        q[:, d] = cs_list[d](t)
        q_dot[:, d] = cs_list[d](t, 1)
        q_ddot[:, d] = cs_list[d](t, 2)

    return t, q, q_dot, q_ddot


# =============================================================================
# 时间参数化
# =============================================================================

def velocity_mvc_from_joint_limits(path_deriv: Callable[[float], np.ndarray],
                                    s_grid: np.ndarray,
                                    q_dot_limits: np.ndarray) -> np.ndarray:
    """仅根据关节速度约束计算 MVC 曲线。

    s_dot_max(s) = min_i q_dot_max,i / |f'_i(s)|
    若 f'_i(s)=0，该关节对此处ṡ无约束。

    此函数只做速度约束，不做加速度约束/前向后向传播。
    完整教学版前向-后向参数化见：
        topp_forward_backward_parameterization()
    """
    n_s = len(s_grid)
    n_dof = len(q_dot_limits)
    s_dot_max = np.full(n_s, np.inf)

    for i in range(n_s):
        s = s_grid[i]
        fp = path_deriv(s)
        for d in range(n_dof):
            if abs(fp[d]) > 1e-10:
                s_dot_max[i] = min(s_dot_max[i],
                                   q_dot_limits[d] / abs(fp[d]))
    return s_dot_max


def topp_forward_backward_parameterization(
        f_vals: np.ndarray,
        fp_vals: np.ndarray,
        fpp_vals: np.ndarray,
        s_vals: np.ndarray,
        q_dot_limits: np.ndarray,
        q_ddot_limits: np.ndarray,
        s_dot_start: float = 0.0,
        s_dot_end: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """教学版前向-后向路径速度参数化近似。

    在 (s, ṡ) 相平面上:
    1. 计算 MVC (最大速度曲线)
    2. 前向传播: 从 s=0 用最大加速 β(s,ṡ) 前进
    3. 后向传播: 从 s=1 用最大减速 α(s,ṡ) 后退
    4. 取 min(前向, 后向, MVC) 作为近似曲线
    5. 通过时间积分得到 t(s)

    注意: 此为教学近似，不保证严格 TOPP-RA 时间最优或全部约束满足。
    生产级应用请使用 toppra 库。

    参数:
        f_vals: (n_s, n_dof) 路径值 q = f(s)
        fp_vals: (n_s, n_dof) 路径一阶导 f'(s)
        fpp_vals: (n_s, n_dof) 路径二阶导 f''(s)
        s_vals: (n_s,) s 网格
        q_dot_limits: (n_dof,) 关节速度上限
        q_ddot_limits: (n_dof,) 关节加速度上限
        s_dot_start: s=0 处初始 ṡ
        s_dot_end: s=1 处终端 ṡ

    返回:
        s_dot_mvc: MVC 曲线
        s_dot_fwd: 前向传播曲线
        s_dot_bwd: 后向传播曲线
        s_dot_approx: 近似 ṡ(s) = min(fwd, bwd, MVC)
        t_vals: 从 ṡ(s) 积分得到的时间数组 t(s)
    """
    n_s = len(s_vals)
    n_dof = len(q_dot_limits)
    tolerance = 1e-6

    # ── 输入验证 ──
    s_vals = np.asarray(s_vals, dtype=float)
    fp_vals = np.asarray(fp_vals, dtype=float)
    fpp_vals = np.asarray(fpp_vals, dtype=float)
    q_dot_limits = np.asarray(q_dot_limits, dtype=float)
    q_ddot_limits = np.asarray(q_ddot_limits, dtype=float)

    if n_s < 2:
        raise ValueError(f"s_vals must have at least 2 points, got {n_s}")
    ds_all = np.diff(s_vals)
    if not np.all(ds_all > 0):
        raise ValueError("s_vals must be strictly increasing")
    if not np.allclose(ds_all, ds_all[0], rtol=1e-10):
        raise ValueError("s_vals must be uniformly spaced (current impl limitation)")

    if fp_vals.shape != (n_s, n_dof):
        raise ValueError(f"fp_vals shape {fp_vals.shape} != (n_s={n_s}, n_dof={n_dof})")
    if fpp_vals.shape != (n_s, n_dof):
        raise ValueError(f"fpp_vals shape {fpp_vals.shape} != (n_s={n_s}, n_dof={n_dof})")

    if np.any(q_dot_limits <= 0):
        raise ValueError(f"q_dot_limits must all be > 0, got {q_dot_limits}")
    if np.any(q_ddot_limits <= 0):
        raise ValueError(f"q_ddot_limits must all be > 0, got {q_ddot_limits}")

    if s_dot_start < -tolerance or s_dot_end < -tolerance:
        raise ValueError(f"s_dot_start/end must be >= 0, got {s_dot_start}, {s_dot_end}")

    ds = ds_all[0]

    # 1. MVC: 速度约束
    s_dot_mvc = np.full(n_s, np.inf)
    for i in range(n_s):
        for d in range(n_dof):
            if abs(fp_vals[i, d]) > 1e-10:
                s_dot_mvc[i] = min(s_dot_mvc[i],
                                   q_dot_limits[d] / abs(fp_vals[i, d]))

    # 边界速度不应超过 MVC
    if s_dot_start > s_dot_mvc[0] + tolerance:
        raise ValueError(f"s_dot_start={s_dot_start} exceeds MVC at s=0 ({s_dot_mvc[0]:.4f})")
    if s_dot_end > s_dot_mvc[-1] + tolerance:
        raise ValueError(f"s_dot_end={s_dot_end} exceeds MVC at s=1 ({s_dot_mvc[-1]:.4f})")

    # 2. α,β 辅助函数 (加速度约束下的可行动态)
    def compute_alpha_beta(s_idx, s_dot_val):
        alpha = -np.inf
        beta = np.inf
        for d in range(n_dof):
            fp = fp_vals[s_idx, d]
            fpp = fpp_vals[s_idx, d]
            term = fpp * s_dot_val**2
            if abs(fp) < 1e-10:
                if abs(term) > q_ddot_limits[d] + 1e-6:
                    return 0.0, 0.0, False
                continue
            lo = (-q_ddot_limits[d] - term) / fp
            hi = (q_ddot_limits[d] - term) / fp
            if fp > 0:
                alpha = max(alpha, lo)
                beta = min(beta, hi)
            else:
                alpha = max(alpha, hi)
                beta = min(beta, lo)
        feasible = alpha <= beta + 1e-6
        return alpha, beta, feasible

    # 3. 前向传播 (最大加速)
    s_dot_fwd = np.zeros(n_s)
    s_dot_fwd[0] = s_dot_start
    for i in range(1, n_s):
        _, beta, feasible = compute_alpha_beta(i - 1, s_dot_fwd[i - 1])
        if not feasible:
            raise RuntimeError(
                f"Forward pass infeasible at s={s_vals[i-1]:.6f}: "
                f"α > β at ṡ={s_dot_fwd[i-1]:.4f}"
            )
        s_dot_sq = s_dot_fwd[i - 1]**2 + 2 * beta * ds
        if s_dot_sq < -1e-10:
            raise RuntimeError(
                f"Forward pass unreachable at s={s_vals[i]:.6f}"
            )
        s_dot_fwd[i] = np.sqrt(max(0, s_dot_sq))
        s_dot_fwd[i] = min(s_dot_fwd[i], s_dot_mvc[i])

    # 4. 后向传播 (最大减速)
    s_dot_bwd = np.zeros(n_s)
    s_dot_bwd[-1] = s_dot_end
    for i in range(n_s - 2, -1, -1):
        alpha, _, feasible = compute_alpha_beta(i + 1, s_dot_bwd[i + 1])
        if not feasible:
            raise RuntimeError(
                f"Backward pass infeasible at s={s_vals[i+1]:.6f}: "
                f"α > β at ṡ={s_dot_bwd[i+1]:.4f}"
            )
        s_dot_sq = s_dot_bwd[i + 1]**2 - 2 * alpha * ds
        if s_dot_sq < -1e-10:
            raise RuntimeError(
                f"Backward pass unreachable at s={s_vals[i]:.6f}"
            )
        s_dot_bwd[i] = np.sqrt(max(0, s_dot_sq))
        s_dot_bwd[i] = min(s_dot_bwd[i], s_dot_mvc[i])

    # 5. 近似曲线
    s_dot_approx = np.minimum(np.minimum(s_dot_fwd, s_dot_bwd), s_dot_mvc)

    # 6. 时间积分 t(s) = ∫ ds / ṡ(s)
    t_vals = np.zeros(n_s)
    for i in range(1, n_s):
        s_dot_avg = max((s_dot_approx[i] + s_dot_approx[i - 1]) / 2, 1e-6)
        t_vals[i] = t_vals[i - 1] + ds / s_dot_avg

    return s_dot_mvc, s_dot_fwd, s_dot_bwd, s_dot_approx, t_vals


# 旧函数保留为兼容别名（重命名以反映其实际功能）
time_optimal_parameterization = velocity_mvc_from_joint_limits
