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
# 时间参数化 (简化 TOPP)
# =============================================================================

def time_optimal_parameterization(path: Callable[[float], np.ndarray],
                                  path_deriv: Callable[[float], np.ndarray],
                                  s_grid: np.ndarray,
                                  q_dot_limits: np.ndarray,
                                  q_ddot_limits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化的时间最优路径参数化

    在 (s, ṡ) 相平面上计算最大速度曲线 (MVC) 和时间最优轨迹。

    参数:
        path: q = f(s), s ∈ [0,1]
        path_deriv: q = f'(s)
        s_grid: s 的离散点
        q_dot_limits, q_ddot_limits: 关节速度/加速度约束

    返回:
        s_dot_max: MVC 曲线
        s_dot_opt: 时间最优 ṡ 曲线
    """
    n_s = len(s_grid)
    n_dof = len(q_dot_limits)
    s_dot_max = np.full(n_s, np.inf)

    # 1. 速度约束: |f'_i(s) * ṡ| ≤ q_dot_max_i
    for i in range(n_s):
        s = s_grid[i]
        fp = path_deriv(s)
        for d in range(n_dof):
            if abs(fp[d]) > 1e-10:
                s_dot_max[i] = min(s_dot_max[i],
                                   q_dot_limits[d] / abs(fp[d]))

    # 2. 加速度约束: |f'_i(s) * s̈ + f''_i(s) * ṡ²| ≤ q_ddot_max_i
    # s̈_max = (q̈_max - f'' ṡ²) / f'（简化处理）

    return s_dot_max, s_dot_max  # 简化版返回 MVC
