"""
仿真工具 (Simulation Utilities)

包含：
- 数值积分（Euler, RK4）
- 动力学仿真循环
- 传感器噪声模拟
"""

import numpy as np
from typing import Callable, Tuple


def rk4_integrate(f: Callable, x0: np.ndarray, u: Callable,
                  t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    四阶 Runge-Kutta 积分

    参数:
        f: 状态导数函数 dx/dt = f(x, t, u(t))
        x0: 初始状态
        u: 控制输入函数 u(t)
        t_span: (t_start, t_end)
        dt: 步长

    返回:
        t: (N,) 时间序列
        x: (N, n) 状态历史
        u_hist: (N, m) 控制输入历史
    """
    t_start, t_end = t_span
    N = int((t_end - t_start) / dt)
    t = np.linspace(t_start, t_end, N)

    n = len(x0)
    x = np.zeros((N, n))
    x[0] = x0

    u0 = u(t_start)
    u_hist = np.zeros((N, len(u0) if hasattr(u0, '__len__') else 1))
    u_hist[0] = u0

    for i in range(N - 1):
        ti = t[i]
        xi = x[i]
        ui = u(ti)

        k1 = f(xi, ti, ui)
        k2 = f(xi + 0.5*dt*k1, ti + 0.5*dt, u(ti + 0.5*dt))
        k3 = f(xi + 0.5*dt*k2, ti + 0.5*dt, u(ti + 0.5*dt))
        k4 = f(xi + dt*k3, ti + dt, u(ti + dt))

        x[i+1] = xi + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        u_hist[i+1] = u(t[i+1])

    return t, x, u_hist


# =============================================================================
# 传感器噪声模拟
# =============================================================================

def simulate_encoder(q_true: np.ndarray, std: float = 0.001,
                     rng: np.random.RandomState = None) -> np.ndarray:
    """编码器测量模拟: z = q + noise"""
    if rng is None:
        rng = np.random.RandomState()
    return q_true + rng.normal(0, std, size=q_true.shape)


def simulate_accelerometer(a_true: np.ndarray, std: float = 0.01,
                           bias: np.ndarray = None,
                           rng: np.random.RandomState = None) -> np.ndarray:
    """加速度计测量模拟: z = a + bias + noise"""
    if rng is None:
        rng = np.random.RandomState()
    if bias is None:
        bias = np.zeros(3)
    return a_true + bias + rng.normal(0, std, size=3)


def simulate_gyroscope(omega_true: np.ndarray, std: float = 0.005,
                       bias: np.ndarray = None,
                       rng: np.random.RandomState = None) -> np.ndarray:
    """陀螺仪测量模拟: z = ω + bias + noise"""
    if rng is None:
        rng = np.random.RandomState()
    if bias is None:
        bias = np.zeros(3)
    return omega_true + bias + rng.normal(0, std, size=3)


def simulate_lidar_2d(robot_pose: np.ndarray,
                      landmarks: np.ndarray,
                      max_range: float = 10.0,
                      std_range: float = 0.05,
                      std_bearing: float = 0.01,
                      rng: np.random.RandomState = None) -> np.ndarray:
    """
    2D 激光雷达测量模拟

    参数:
        robot_pose: (3,) [x, y, θ]
        landmarks: (M, 2) 路标位置
        max_range: 最大测量距离
        std_range, std_bearing: 距离和方位噪声

    返回:
        measurements: (M', 2) [range, bearing] 对每个可见路标
    """
    if rng is None:
        rng = np.random.RandomState()

    x, y, theta = robot_pose
    measurements = []

    for lm in landmarks:
        dx, dy = lm[0] - x, lm[1] - y
        r = np.sqrt(dx**2 + dy**2)
        b = np.arctan2(dy, dx) - theta

        # 归一化方位到 [-π, π]
        b = np.arctan2(np.sin(b), np.cos(b))

        if r < max_range:
            r_meas = r + rng.normal(0, std_range)
            b_meas = b + rng.normal(0, std_bearing)
            measurements.append([r_meas, b_meas])

    return np.array(measurements) if measurements else np.zeros((0, 2))
