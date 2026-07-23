"""
控制 (Control)

包含：
- PID 控制器
- 重力补偿 PD
- 计算力矩控制（CTC）
- 滑模控制（SMC）
"""

import numpy as np
from typing import Callable


class PIDController:
    """离散 PID 控制器（含积分抗饱和）"""

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 dt: float = 0.01, integral_limit: float = None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float) -> float:
        # P
        p_term = self.kp * error

        # I（含抗饱和）
        self._integral += error * self.dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral,
                                     -self.integral_limit,
                                     self.integral_limit)
        i_term = self.ki * self._integral

        # D（用误差导数，避免 setpoint 阶跃时的导数冲击）
        if self._prev_error is None:
            d_term = 0.0
        else:
            d_term = self.kd * (error - self._prev_error) / self.dt

        self._prev_error = error

        return p_term + i_term + d_term


def gravity_compensation_pd(q_des: np.ndarray, q_dot_des: np.ndarray,
                            q: np.ndarray, q_dot: np.ndarray,
                            Kp: np.ndarray, Kd: np.ndarray,
                            gravity_func: Callable) -> np.ndarray:
    """
    重力补偿 PD 控制

    τ = Kp (q_des − q) + Kd (q̇_des − q̇) + g(q)

    参数:
        q_des, q_dot_des: 期望关节位置和速度
        q, q_dot: 实际关节位置和速度
        Kp, Kd: 对角增益矩阵（或向量，会自动转为 diag）
        gravity_func: g(q) 函数
    返回:
        τ: 关节力矩
    """
    if Kp.ndim == 1:
        Kp = np.diag(Kp)
    if Kd.ndim == 1:
        Kd = np.diag(Kd)

    tau = Kp @ (q_des - q) + Kd @ (q_dot_des - q_dot) + gravity_func(q)
    return tau


def computed_torque_control(q_des: np.ndarray, q_dot_des: np.ndarray,
                            q_ddot_des: np.ndarray,
                            q: np.ndarray, q_dot: np.ndarray,
                            Kp: np.ndarray, Kd: np.ndarray,
                            M_func: Callable, C_func: Callable,
                            g_func: Callable) -> np.ndarray:
    """
    计算力矩控制（Computed Torque Control / Inverse Dynamics Control）

    τ = M(q) [q̈_des + Kd (q̇_des − q̇) + Kp (q_des − q)] + C(q,q̇) q̇ + g(q)

    等效闭环误差动力学: ë + Kd ė + Kp e = 0

    参数:
        q_des, q_dot_des, q_ddot_des: 期望轨迹
        q, q_dot: 实际状态
        Kp, Kd: 增益矩阵（向量形式）
        M_func(q), C_func(q, q_dot), g_func(q): 动力学函数
    返回:
        τ: 关节力矩
    """
    if Kp.ndim == 1:
        Kp = np.diag(Kp)
    if Kd.ndim == 1:
        Kd = np.diag(Kd)

    e = q_des - q
    e_dot = q_dot_des - q_dot

    # 期望闭环加速度
    q_ddot_ref = q_ddot_des + Kd @ e_dot + Kp @ e

    M = M_func(q)
    C_qdot = C_func(q, q_dot) @ q_dot if C_func(q, q_dot).ndim > 1 else C_func(q, q_dot)
    g = g_func(q)

    return M @ q_ddot_ref + C_qdot + g


def sliding_mode_control(q_des: np.ndarray, q_dot_des: np.ndarray,
                         q_ddot_des: np.ndarray,
                         q: np.ndarray, q_dot: np.ndarray,
                         Lambda: np.ndarray, K: np.ndarray,
                         M_func: Callable, C_func: Callable,
                         g_func: Callable,
                         boundary: float = 0.0) -> np.ndarray:
    """
    滑模控制（Sliding Mode Control）

    滑模面: s = ė + Λ e
    控制律: τ = M (q̈_des + Λ ė) + C s + C q̇_des + g − K sign(s)
            (或边界层: − K sat(s/boundary))

    参数:
        Lambda: 滑模面参数 λ（对角）
        K: 切换增益（对角）
        boundary: 边界层厚度（>0 使用 sat 替代 sign 减少抖振）
    """
    if Lambda.ndim == 1:
        Lambda = np.diag(Lambda)
    if K.ndim == 1:
        K = np.diag(K)

    e = q_des - q
    e_dot = q_dot_des - q_dot

    s = e_dot + Lambda @ e  # 滑模面

    # 饱和函数
    if boundary > 0:
        sat_s = np.where(np.abs(s) < boundary, s / boundary, np.sign(s))
    else:
        sat_s = np.sign(s)

    M = M_func(q)
    C_full = C_func(q, q_dot)
    C_qdot = C_full @ q_dot
    g = g_func(q)

    q_ddot_ref = q_ddot_des + Lambda @ e_dot + K @ sat_s
    tau = M @ q_ddot_ref + C_qdot + g

    return tau
