# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (robotics-learning)
#     language: python
#     name: robotics-learning
# ---

# %% [markdown]
# # Notebook 23b：Error State Kalman Filter (ESKF)

# ## 1. 定位
# 标准 EKF 直接在状态空间上线性化，但姿态（SO(3)）不是向量空间。ESKF 将状态分解为 nominal state（在流形上演化）和 error state（在切空间中线性化），解决了姿态估计中的流形约束问题。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ 理解 nominal state vs error state 的分离
# - ⭐ 掌握 SO(3) 上的扰动：$\mathbf{R} = \hat{\mathbf{R}} \exp([\delta\boldsymbol{\theta}]_\times)$
# - ⭐ ESKF 传播、更新与 error injection
# - ⭐ 理解 reset Jacobian
# - 📖 左误差 vs 右误差

# %% [markdown]
# ## 3. ESKF 框架 ⭐

# %% [markdown]
# ### 3.1 状态分解
# $$\mathbf{x} = \hat{\mathbf{x}} \oplus \delta\mathbf{x}$$
# - $\hat{\mathbf{x}}$: nominal state（大信号、非线性演化）
# - $\delta\mathbf{x}$: error state（小信号、线性高斯演化）
# - $\oplus$: 流形上的"加法"

# %% [markdown]
# ### 3.2 传播
# Nominal state 用完整非线性动力学预测。
# Error state covariance 用线性化动力学传播（与 EKF 相同）。

# %% [markdown]
# ### 3.3 更新
# 卡尔曼更新修正 error state：$\delta\mathbf{x} \leftarrow \mathbf{K}(\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}))$
# 然后将 error state **注入** nominal state：$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} \oplus \delta\mathbf{x}$
# 注入后 error state 重置为 0（但协方差不重置为 0，需要 reset Jacobian 修正）。

# %% [markdown]
# ## 4. Python — 2D IMU ESKF

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from src.robotics_learning.transforms import so3_exp, so3_log, axis_angle_to_rot
from src.robotics_learning.estimation import KalmanFilter
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 4.1 ESKF — 简化 2D 姿态 + 位置估计

# %%
dt_eskf = 0.01; N_eskf = 500
# Nominal state: [x, y, vx, vy, theta]
# Error state:   [δx, δy, δvx, δvy, δθ]  ← 线性高斯
# IMU: gyro ω + accel a_x, a_y (body frame)

# 真值轨迹（圆）
t_vals = np.arange(N_eskf) * dt_eskf
true_x = 2*np.cos(0.5*t_vals); true_y = 2*np.sin(0.5*t_vals)
true_theta = 0.5*t_vals + np.pi/2
true_vx = np.gradient(true_x, dt_eskf); true_vy = np.gradient(true_y, dt_eskf)

# Nominal state
x_nom = np.array([true_x[0]+0.5, true_y[0]-0.3, 0.0, 0.0, true_theta[0]+0.2])
P_eskf = np.eye(5) * 0.1
P_eskf[4, 4] = 0.5  # 角度不确定性大

x_nom_hist = [x_nom.copy()]; P_hist = [P_eskf.copy()]

Q_eskf = np.diag([1e-4, 1e-4, 0.1, 0.1, 0.01])  # 过程噪声
R_eskf_gps = np.diag([0.5, 0.5])  # GPS 观测噪声（位置）

for k in range(1, N_eskf):
    # 1. Nominal state 预测（IMU dead-reckoning）
    # 简化: 用真值加小噪声模拟 IMU
    omega = 0.5 + rng.normal(0, 0.05)  # gyro
    a_body = np.array([0.0, 0.0]) + rng.normal(0, 0.1, 2)  # accel (近似)

    theta_nom = x_nom[4]
    R_bw = np.array([[np.cos(theta_nom), -np.sin(theta_nom)],
                     [np.sin(theta_nom), np.cos(theta_nom)]])
    a_world = R_bw @ a_body

    x_nom_pred = x_nom.copy()
    x_nom_pred[2:4] += a_world * dt_eskf  # v += a*dt
    x_nom_pred[0:2] += x_nom_pred[2:4] * dt_eskf  # p += v*dt
    x_nom_pred[4] += omega * dt_eskf

    # 2. Error state 协方差预测
    # 线性化: A ≈ I + dt * ∂f/∂x
    A_eskf = np.eye(5)
    A_eskf[0, 2] = dt_eskf; A_eskf[1, 3] = dt_eskf
    # 速度受姿态影响: δv̇ = R [a]× δθ (简化)
    A_eskf[2, 4] = (-R_bw[0,0]*a_body[1] + R_bw[0,1]*a_body[0]) * dt_eskf
    A_eskf[3, 4] = (-R_bw[1,0]*a_body[1] + R_bw[1,1]*a_body[0]) * dt_eskf

    P_pred = A_eskf @ P_eskf @ A_eskf.T + Q_eskf

    # 3. GPS 更新（每 50 步，模拟 2Hz GPS vs 100Hz IMU）
    if k % 50 == 0:
        z_gps = np.array([true_x[k], true_y[k]]) + rng.normal(0, 0.7, 2)
        C_gps = np.array([[1,0,0,0,0],[0,1,0,0,0]])
        S = C_gps @ P_pred @ C_gps.T + R_eskf_gps
        K = P_pred @ C_gps.T @ np.linalg.solve(S, np.eye(2))
        innov = z_gps - x_nom_pred[:2]
        dx = K @ innov
        P_eskf = (np.eye(5) - K @ C_gps) @ P_pred
        # Error injection
        x_nom_pred[:5] += dx
        theta_nom = x_nom_pred[4] % (2*np.pi)
        x_nom_pred[4] = theta_nom
    else:
        P_eskf = P_pred

    x_nom = x_nom_pred
    x_nom_hist.append(x_nom.copy()); P_hist.append(P_eskf.copy())

x_nom_hist = np.array(x_nom_hist); P_hist = np.array(P_hist)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(true_x, true_y, 'k-', linewidth=2, label='True')
axes[0].plot(x_nom_hist[:,0], x_nom_hist[:,1], 'b--', linewidth=2, label='ESKF Estimate')
axes[0].scatter(true_x[::50], true_y[::50], c='red', s=20, marker='o', label='GPS (every 50 steps)')
for i in range(0, N_eskf, 100):
    cov = P_hist[i, :2, :2]
    ev, evec = np.linalg.eigh(cov)
    ell = plt.matplotlib.patches.Ellipse(x_nom_hist[i,:2], 2*np.sqrt(max(ev[0],1e-6)), 2*np.sqrt(max(ev[1],1e-6)),
           angle=np.degrees(np.arctan2(evec[1,0],evec[0,0])), facecolor='blue', alpha=0.05, edgecolor='blue', lw=0.5)
    axes[0].add_patch(ell)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal')
axes[0].set_title('ESKF — 2D Pose + Velocity Estimation'); axes[0].legend()

theta_err = np.array([np.arctan2(np.sin(true_theta - x_nom_hist[:,4]), np.cos(true_theta - x_nom_hist[:,4]))])
axes[1].plot(t_vals, theta_err[0], 'b-', linewidth=1.5)
axes[1].fill_between(t_vals, -2*np.sqrt(P_hist[:,4,4]), 2*np.sqrt(P_hist[:,4,4]), color='blue', alpha=0.15)
axes[1].set_xlabel('t (s)'); axes[1].set_ylabel('θ error (rad)')
axes[1].set_title('Attitude Error ±2σ'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/23b_eskf_2d.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. 与标准 EKF 的关键区别
# | | EKF | ESKF |
# |---|---|---|
# | 状态空间 | 向量空间 | 流形 + 切空间 |
# | 姿态更新 | $\mathbf{q} \leftarrow \mathbf{q} + \delta\mathbf{q}$（违反单位范数）| $\mathbf{q} \leftarrow \mathbf{q} \otimes \exp(\delta\boldsymbol{\theta})$（保持在 S³）|
# | 协方差 | 在过度参数化空间 | 在最小参数切空间 |
# | 数值稳定性 | 可能漂移出 SO(3) | 始终合法 |

# %% [markdown]
# ## 6. 练习题
# 1. ESKF 比 EKF 更适合姿态估计的根本原因？
# 2. Error injection 后为什么需要 reset Jacobian？
# 3. 左误差和右误差的物理区别？
