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
# # Notebook 24：传感器模型与多传感器融合
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB22-23 KF/EKF ──→ NB24 传感器融合 ──→ NB25 综合项目
# NB02 坐标变换 ──→ (tf 树 + 多传感器 EKF)     (完整闭环)
# ```
#
# 真实机器人系统需要融合多种传感器——编码器、IMU、相机、激光雷达——各自有不同的测量模型、噪声特性和采样频率。本节建立多传感器融合的工程框架。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解编码器/IMU/视觉/激光雷达的基本测量模型
# - ⭐ 掌握 tf 树（坐标变换链）的结构
# - ⭐ 理解多传感器异步 EKF 框架
# - ⭐ 理解可观测性分析在传感器选型中的意义
# - 📖 前融合 vs 后融合

# %% [markdown]
# ## 3. 传感器模型 ⭐

# %% [markdown]
# ### 3.1 编码器（Encoder）
#
# $$\mathbf{z}_{enc} = \mathbf{q} + \mathbf{n}_{enc}, \quad \mathbf{n}_{enc} \sim \mathcal{N}(\mathbf{0}, \sigma_{enc}^2\mathbf{I})$$
#
# 最简单、最可靠的传感器——直接测量关节角。通常作为 KF 过程中最基础的观测。

# %% [markdown]
# ### 3.2 IMU（加速度计 + 陀螺仪）
#
# - 加速度计：$\mathbf{a}_{meas} = \mathbf{R}^T(\mathbf{a} - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$
#   - 测量的是**比力**（specific force），不是绝对加速度！
#   - $\mathbf{g}$ 在无加速度时测量到的是 $-\mathbf{g}$（即向上 1g）
# - 陀螺仪：$\boldsymbol{\omega}_{meas} = \boldsymbol{\omega} + \mathbf{b}_\omega + \mathbf{n}_\omega$
# - 偏置 $\mathbf{b}$ 随时间缓慢漂移——需要编码器或视觉来校正

# %% [markdown]
# ### 3.3 视觉/相机
#
# 针孔相机模型：从 3D 点 $\mathbf{p}^C = [X,Y,Z]^T$ 到 2D 像素 $[u,v]^T$：
# $$u = f_x\frac{X}{Z} + c_x, \quad v = f_y\frac{Y}{Z} + c_y$$
# 重投影误差用于 EKF 更新。

# %% [markdown]
# ## 4. tf 树与时空配准 ⭐

# %% [markdown]
# ### 4.1 tf 树结构
#
# ```
# {world} → {base} → {link_1} → ... → {link_n} → {camera}
#                          ↘ {imu}
#                          ↘ {encoder_joint_2}
# ```
#
# 每个传感器有自己的参考系。要将传感器测量用于 EKF 更新，必须通过 tf 变换链将测量转换到统一参考系。

# %% [markdown]
# ### 4.2 多传感器异步 EKF
#
# ```python
# for t in time_steps:
#     ekf.predict(u, dt)          # 恒定预测
#     if encoder_ready(t):
#         ekf.update(z_enc, C_enc, R_enc)
#     if imu_ready(t):
#         ekf.update(z_imu, C_imu, R_imu)
#     if camera_ready(t):
#         ekf.update(z_cam, C_cam, R_cam)
# ```

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.estimation import KalmanFilter
from src.robotics_learning.simulation import simulate_encoder, simulate_accelerometer, simulate_gyroscope
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 编码器 + 陀螺仪融合（无偏置简化模型）
# 本简化模型状态为 x=[θ, θ̇]^T，不含陀螺仪偏置 b_g。
# 陀螺仪直接测量角速度 θ̇，编码器测量角度 θ。

# %%
dt_f = 0.01; N_f = 300
# 状态: [θ, θ̇] — 单关节角度和速度
A_f = np.array([[1, dt_f], [0, 1]])
Q_f = np.diag([1e-6, 0.01])  # 过程噪声

# 传感器
C_enc = np.array([[1, 0]])    # 编码器测角度
R_enc = np.array([[0.001]])   # 编码器噪声(小)
C_gyro = np.array([[0, 1]])    # 陀螺仪：直接测量角速度
R_gyro = np.array([[0.5]])     # 陀螺仪噪声

# 真实轨迹
true_q = np.zeros(N_f); true_qd = np.zeros(N_f)
true_qd[0] = 2.0
mu_f = np.zeros((N_f, 2))
sigma_f = np.zeros((N_f, 2))

kf_f = KalmanFilter(A_f, np.zeros((2,1)), C_enc, Q_f, R_enc,
                    mu=np.array([0, 0]), Sigma=np.eye(2)*1.0)

for t in range(1, N_f):
    true_q[t] = true_q[t-1] + true_qd[t-1]*dt_f
    true_qd[t] = true_qd[t-1] + rng.normal(0, np.sqrt(0.01))

    # 编码器（100Hz — 每次都更新）
    z_enc = true_q[t] + rng.normal(0, 0.03)
    kf_f.predict()
    kf_f.update(z_enc)  # 注意：这里用的是 C_enc

    # IMU（50Hz — 每隔一次）
    if t % 2 == 0:
        z_imu = true_qd[t] + rng.normal(0, np.sqrt(0.5))
        # 临时切换到 IMU 观测矩阵
        kf_f_tmp = KalmanFilter(A_f, np.zeros((2,1)), C_gyro, Q_f, R_gyro,
                                mu=kf_f.mu, Sigma=kf_f.Sigma)
        kf_f_tmp.update(np.array([z_imu]))
        kf_f.mu, kf_f.Sigma = kf_f_tmp.mu, kf_f_tmp.Sigma

    mu_f[t] = kf_f.mu
    sigma_f[t] = np.sqrt(np.diag(kf_f.Sigma))

t_f = np.arange(N_f)*dt_f
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t_f, true_q, 'k-', linewidth=2, label='True θ')
axes[0].plot(t_f, mu_f[:, 0], 'b-', linewidth=1.5, label='Estimate θ̂')
axes[0].fill_between(t_f, mu_f[:,0]-2*sigma_f[:,0], mu_f[:,0]+2*sigma_f[:,0], color='blue', alpha=0.15)
axes[0].set_ylabel('θ (rad)'); axes[0].set_title('Encoder + IMU Fusion — Position')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_f, true_qd, 'k-', linewidth=1, label='True θ̇')
axes[1].plot(t_f, mu_f[:, 1], 'r-', linewidth=1.5, label='Estimate θ̇̂')
axes[1].fill_between(t_f, mu_f[:,1]-2*sigma_f[:,1], mu_f[:,1]+2*sigma_f[:,1], color='red', alpha=0.15)
axes[1].set_ylabel('θ̇ (rad/s)'); axes[1].set_xlabel('t (s)')
axes[1].set_title('Encoder + IMU Fusion — Velocity')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/24_sensor_fusion.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.2 可观测性分析

# %%
# 可观测性矩阵 O = [C; CA; CA²; ...]
O_enc_only = C_enc
for k in range(1, 2):
    O_enc_only = np.vstack([O_enc_only, C_enc @ np.linalg.matrix_power(A_f, k)])

O_both = np.vstack([C_enc, C_gyro])
for k in range(1, 2):
    O_both = np.vstack([O_both, C_enc @ np.linalg.matrix_power(A_f, k), C_gyro @ np.linalg.matrix_power(A_f, k)])

rank_enc = np.linalg.matrix_rank(O_enc_only)
rank_both = np.linalg.matrix_rank(O_both)
print(f"仅编码器: rank(O) = {rank_enc} (满秩=2 则完全可观测)")
print(f"编码器+IMU: rank(O) = {rank_both}")
# 说明: 在恒速度模型 x=[q, q̇]^T 下，编码器测量 q 本身已使系统可观测
# q̇ 可通过差分推断。IMU 直接测量 q̇，可改善估计带宽和数值条件，
# 但并非从不可观测变为可观测。真正需要 IMU 的场景是当 q̇ 的动态
# 存在模型误差（如未知扰动）或需要估计偏置时。
if rank_enc == 2:
    print("注意：恒速度模型下，编码器角度序列已使 [q, q̇] 可观测。")
    print("陀螺仪直接测量角速度 → 改善估计带宽、数值条件和动态响应，")
    print("但不是从不可观测变成可观测。")
print("本模型为无偏置简化 (x=[q,q̇] 不含 b_g)。带偏置模型需扩展 x=[q,q̇,b_g]^T。")

# %% [markdown]
# ## 6. 练习题
#
# ### 概念题
# 1. 加速度计测量的是加速度还是比力？两者的区别？
# 2. 多传感器异步融合时，时间戳不同怎么处理？
#
# ### 编程题
# 1. 实现带偏置估计的 IMU+编码器 EKF。
# 2. 加入视觉重投影误差观测，实现完整的 VIO 简化版。
#
# > 答案见 `solutions/24_solutions.ipynb`

# %% [markdown]
# ## 7. 本节总结
#
# | 传感器 | 测量 | 噪声特性 | 频率 | 融合角色 |
# |--------|------|:--------:|:----:|----------|
# | 编码器 | 关节角 q | 低噪声、无漂移 | 高 (1kHz) | 基础位置 |
# | 加速度计 | 比力 a-g | 高噪声、低频漂移 | 高 (200Hz) | 姿态修正 |
# | 陀螺仪 | 角速度 ω (不含偏置，本简化模型) | 低噪声但有漂移 | 高 (200Hz) | 短时姿态/改善带宽 |
# | 相机 | 像素 [u,v] | 中噪声 | 低 (30Hz) | 绝对参考 |
| 激光雷达 | 距离+方位 | 低噪声 | 中 (10Hz) | 精确距离 |
