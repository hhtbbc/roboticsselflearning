"""
动力学 (Dynamics) — 拉格朗日法、牛顿-欧拉法

包含：
- 2R 臂拉格朗日动力学（显式公式）
- RNEA（递归牛顿-欧拉算法）
- 数值积分仿真
"""

import numpy as np
from typing import Tuple, Callable


# =============================================================================
# 2R 平面机械臂 — 拉格朗日动力学显式公式
# =============================================================================

class TwoLinkArmDynamics:
    """
    2R 平面机械臂的拉格朗日动力学

    连杆参数:
        m1, m2: 质量
        l1, l2: 长度（到质心的距离，假设质心在连杆中点）
        I1, I2: 绕质心的转动惯量
        g: 重力加速度（正值 = 朝下）

    动力学方程:
        M(q) q̈ + C(q, q̇) q̇ + g(q) = τ

    使用复合连杆参数简化:
        α = I1 + I2 + m1*lc1² + m2*(l1² + lc2²)
        β = m2*l1*lc2
        δ = I2 + m2*lc2²
    """

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0,
                 lc1=None, lc2=None, I1=None, I2=None, g=9.81):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1 = lc1 if lc1 is not None else l1 / 2
        self.lc2 = lc2 if lc2 is not None else l2 / 2
        self.I1 = I1 if I1 is not None else m1 * l1**2 / 12
        self.I2 = I2 if I2 is not None else m2 * l2**2 / 12
        self.g = g

        # 复合参数
        self.alpha = (self.I1 + self.I2 + self.m1 * self.lc1**2
                      + self.m2 * (self.l1**2 + self.lc2**2))
        self.beta = self.m2 * self.l1 * self.lc2
        self.delta = self.I2 + self.m2 * self.lc2**2

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """质量矩阵 M(q) ∈ ℝ^{2×2}"""
        c2 = np.cos(q[1])
        m11 = self.alpha + 2 * self.beta * c2
        m12 = self.delta + self.beta * c2
        m22 = self.delta
        return np.array([[m11, m12], [m12, m22]])

    def coriolis_matrix(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """科氏力/离心力矩阵 C(q, q̇) ∈ ℝ^{2×2}"""
        s2 = np.sin(q[1])
        c = -self.beta * s2
        return np.array([
            [c * q_dot[1], c * (q_dot[0] + q_dot[1])],
            [-c * q_dot[0], 0]
        ])

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """重力向量 g(q) ∈ ℝ^{2}"""
        c1 = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])

        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * c1 \
             + self.m2 * self.lc2 * self.g * c12
        g2 = self.m2 * self.lc2 * self.g * c12
        return np.array([g1, g2])

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """逆动力学: τ = M q̈ + C q̇ + g"""
        M = self.mass_matrix(q)
        Cq_dot = self.coriolis_matrix(q, q_dot) @ q_dot
        g_vec = self.gravity_vector(q)
        return M @ q_ddot + Cq_dot + g_vec

    def forward_dynamics(self, q, q_dot, tau):
        """正动力学: q̈ = M⁻¹(τ − C q̇ − g)"""
        M = self.mass_matrix(q)
        Cq_dot = self.coriolis_matrix(q, q_dot) @ q_dot
        g_vec = self.gravity_vector(q)
        return np.linalg.solve(M, tau - Cq_dot - g_vec)


# =============================================================================
# 数值积分
# =============================================================================

def rk4_step(f: Callable, x: np.ndarray, u: np.ndarray,
             dt: float) -> np.ndarray:
    """四阶 Runge-Kutta 积分一步"""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def euler_step(f: Callable, x: np.ndarray, u: np.ndarray,
               dt: float) -> np.ndarray:
    """显式欧拉积分一步"""
    return x + dt * f(x, u)


def simulate_dynamics(dyn: TwoLinkArmDynamics,
                      q0: np.ndarray, q_dot0: np.ndarray,
                      tau_func: Callable,
                      t_span: float, dt: float,
                      method: str = 'rk4') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    仿真 2R 臂动力学

    状态: x = [q0, q1, q_dot0, q_dot1]

    参数:
        dyn: 动力学模型
        q0, q_dot0: 初始状态
        tau_func: 控制律 τ = tau_func(t, q, q_dot)
        t_span: 仿真时长
        dt: 步长
        method: 'rk4' 或 'euler'

    返回:
        t: 时间序列 (N,)
        q: 关节角历史 (N, 2)
        q_dot: 关节速度历史 (N, 2)
    """
    N = int(t_span / dt)
    t = np.linspace(0, t_span, N)

    q_hist = np.zeros((N, 2))
    q_dot_hist = np.zeros((N, 2))
    tau_hist = np.zeros((N, 2))

    q = q0.copy()
    q_dot = q_dot0.copy()

    integrator = rk4_step if method == 'rk4' else euler_step

    for i in range(N):
        q_hist[i] = q
        q_dot_hist[i] = q_dot

        tau = tau_func(t[i], q, q_dot)
        tau_hist[i] = tau

        # 状态 x = [q; q_dot]
        def f(x, u):
            q_cur, q_dot_cur = x[:2], x[2:]
            q_ddot = dyn.forward_dynamics(q_cur, q_dot_cur, u)
            return np.concatenate([q_dot_cur, q_ddot])

        x = np.concatenate([q, q_dot])
        x_next = integrator(f, x, tau, dt)
        q = x_next[:2]
        q_dot = x_next[2:]

    return t, q_hist, q_dot_hist, tau_hist


# =============================================================================
# 简化的递归牛顿-欧拉 (RNEA) — 2R 臂专用
# =============================================================================

def rnea_2r(dyn: TwoLinkArmDynamics, q, q_dot, q_ddot,
            g_vec=np.array([0, 0, -9.81])) -> np.ndarray:
    """
    2R 臂的简化 RNEA 实现

    参数:
        dyn: 动力学参数
        q, q_dot, q_ddot: 关节位置/速度/加速度
        g_vec: 基座参考系中的重力加速度向量 (默认 Z 向下)
    返回:
        tau: (2,) 关节力矩
    """
    # Simplified RNEA — delegates to Lagrangian ID for correctness.
    # Full recursive implementation is in Notebook 11.
    return dyn.inverse_dynamics(q, q_dot, q_ddot)
