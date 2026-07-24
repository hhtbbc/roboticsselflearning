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
# # Notebook 23c：IMU 预积分

# %% [markdown]
# ## 1. 定位
# 在因子图优化（VIO/SLAM）中，200Hz IMU 若每次重积分计算量爆炸。预积分将两关键帧间的 IMU 测量压缩为相对运动增量 $\Delta\mathbf{R}_{ij}, \Delta\mathbf{v}_{ij}, \Delta\mathbf{p}_{ij}$。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ IMU 连续测量模型：比力 vs 绝对加速度
# - ⭐ 预积分三增量定义
# - ⭐ bias Jacobian 的作用
# - 📖 预积分协方差传播

# %% [markdown]
# ## 3. 测量模型 ⭐
# 陀螺仪：${}^{B}\tilde{\boldsymbol{\omega}} = {}^{B}\boldsymbol{\omega} + \mathbf{b}_g + \boldsymbol{\eta}_g$
# 加速度计：${}^{B}\tilde{\mathbf{a}} = \mathbf{R}_{WB}^T({}^{W}\mathbf{a} - {}^{W}\mathbf{g}) + \mathbf{b}_a + \boldsymbol{\eta}_a$

# %% [markdown]
# ## 4. Python — 2D 简化预积分

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %%
dt_imu = 0.005; T_imu = 2.0; N_imu = int(T_imu/dt_imu)
t_imu = np.arange(N_imu) * dt_imu

# 真值：常角速度绕圆
omega_true = 0.5 * np.ones(N_imu)
a_world = np.column_stack([-0.5*np.cos(0.5*t_imu), -0.5*np.sin(0.5*t_imu)])
theta_true = np.cumsum(omega_true) * dt_imu
g_world = np.array([0.0, 0.0])

# IMU 测量（噪声+偏置）
bias_gyro = 0.02; bias_acc = np.array([0.05, -0.03])
gyro_meas = omega_true + bias_gyro + rng.normal(0, 0.01, N_imu)
acc_meas_body = np.zeros((N_imu, 2))
for k in range(N_imu):
    R_wb_T = np.array([[np.cos(theta_true[k]), np.sin(theta_true[k])],
                       [-np.sin(theta_true[k]), np.cos(theta_true[k])]])
    acc_meas_body[k] = R_wb_T @ (a_world[k] - g_world) + bias_acc + rng.normal(0, 0.05, 2)

# 预积分
dtheta_acc = 0.0; dv = np.zeros(2); dp = np.zeros(2)
dtheta_hist = [0.0]; dv_hist = [np.zeros(2)]; dp_hist = [np.zeros(2)]

# 真值积分
R_wb = np.eye(2); p_w = np.zeros(2); v_w = np.zeros(2)
p_w_hist = [np.zeros(2)]

for k in range(N_imu):
    omega_k = gyro_meas[k] - bias_gyro
    a_k = acc_meas_body[k] - bias_acc

    dtheta = omega_k * dt_imu
    R_d = np.array([[np.cos(dtheta_acc), -np.sin(dtheta_acc)],
                    [np.sin(dtheta_acc), np.cos(dtheta_acc)]])
    dtheta_acc += dtheta
    dv = dv + R_d @ a_k * dt_imu
    dp = dp + dv * dt_imu + 0.5 * R_d @ a_k * dt_imu**2

    dtheta_hist.append(dtheta_acc)
    dv_hist.append(dv.copy())
    dp_hist.append(dp.copy())

    R_wb = R_wb @ np.array([[np.cos(dtheta), -np.sin(dtheta)],
                            [np.sin(dtheta), np.cos(dtheta)]])
    v_w = v_w + R_wb @ a_k * dt_imu
    p_w = p_w + v_w * dt_imu
    p_w_hist.append(p_w.copy())

dtheta_hist = np.array(dtheta_hist)
dp_hist = np.array(dp_hist); dv_hist = np.array(dv_hist); p_w_hist = np.array(p_w_hist)

# IMU 死推算（直接积分，有偏置）
p_dead = np.zeros((N_imu+1, 2)); v_dead = np.zeros(2); R_dead = np.eye(2)
for k in range(N_imu):
    omega_k = gyro_meas[k]; a_k = acc_meas_body[k]
    dtheta = omega_k * dt_imu
    Rd = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
    R_dead = R_dead @ Rd
    v_dead = v_dead + R_dead @ a_k * dt_imu
    p_dead[k+1] = p_dead[k] + v_dead * dt_imu

# 真值
p_w_true = np.zeros((N_imu+1, 2)); p_w_true[0] = [2, 0]
for k in range(N_imu):
    R_t = np.array([[np.cos(theta_true[k]), -np.sin(theta_true[k])],
                    [np.sin(theta_true[k]), np.cos(theta_true[k])]])
    p_w_true[k+1] = p_w_true[k] + R_t @ np.array([0.0, 0.0]) * dt_imu  # simplified
p_w_true = np.column_stack([2*np.cos(0.5*np.concatenate([[0], t_imu])+np.pi/2),
                             2*np.sin(0.5*np.concatenate([[0], t_imu])+np.pi/2)])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
t_plot = np.concatenate([[0], t_imu])

axes[0,0].plot(t_plot, dp_hist[:,0], label='Δp_x'); axes[0,0].plot(t_plot, dp_hist[:,1], label='Δp_y')
axes[0,0].set_xlabel('t (s)'); axes[0,0].set_ylabel('Δp (m)')
axes[0,0].set_title('Preintegrated Position Increment'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(p_w_true[:,0], p_w_true[:,1], 'k-', linewidth=2, label='True')
axes[0,1].plot(p_dead[:,0], p_dead[:,1], 'r--', linewidth=2, label='IMU Dead-Reckoning')
axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
axes[0,1].set_title('Trajectory: True vs IMU (drift!)'); axes[0,1].set_aspect('equal')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
drift = np.linalg.norm(p_dead[-1] - p_w_true[-1])
axes[0,1].annotate(f'Drift: {drift:.2f}m', (p_dead[-1,0], p_dead[-1,1]))

axes[1,0].plot(t_plot, np.degrees(dtheta_hist))
axes[1,0].set_xlabel('t (s)'); axes[1,0].set_ylabel('Δθ (°)')
axes[1,0].set_title('Preintegrated Rotation Angle (linear in ω)'); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(t_plot, dp_hist[:,0], label='Δp_x'); axes[1,1].plot(t_plot, dv_hist[:,0], '--', label='Δv_x')
axes[1,1].set_xlabel('t (s)'); axes[1,1].set_title('Δp_x and Δv_x'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('IMU Preintegration — 2D Simplified', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/23c_imu_preintegration.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"IMU 死推算漂移: {drift:.2f}m (仅 2 秒!)")
print("预积分 + 视觉/GPS 可在因子图中高效修正此漂移。")

# %% [markdown]
# ## 5. 练习题
# 1. 预积分为何需要 bias Jacobian？偏置估计更新时如何修正预积分？
# 2. 加速度计测量比力——对预积分公式的影响？
