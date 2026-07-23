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
# # Notebook 22：卡尔曼滤波（Kalman Filter）
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB21 概率基础 ──→ NB22 卡尔曼滤波 ──→ NB23 EKF/粒子滤波
#                       │
#                       └── 线性高斯系统的最优估计器
# ```
#
# 卡尔曼滤波是机器人学中最重要的状态估计算法。它在**线性高斯**假设下是最优的（最小方差无偏估计），且计算效率高——只需维护 $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ 两个量。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 掌握 KF 预测步和更新步的完整公式
# - ⭐ 理解卡尔曼增益 $\mathbf{K}$ 的物理直觉（Q 小→信预测，R 小→信观测）
# - ⭐ 理解 $\boldsymbol{\Sigma}$ 不依赖观测 $\mathbf{z}$（可离线计算）
# - ⭐ 理解可观测性分析的 rank 条件
# - 📖 信息滤波与稳态卡尔曼增益

# %% [markdown]
# ## 3. KF 算法 ⭐

# %% [markdown]
# ### 3.1 系统模型
#
# **过程模型**：$\mathbf{x}_t = \mathbf{A}_t \mathbf{x}_{t-1} + \mathbf{B}_t \mathbf{u}_t + \boldsymbol{\epsilon}_t$，$\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)$
# **观测模型**：$\mathbf{z}_t = \mathbf{C}_t \mathbf{x}_t + \boldsymbol{\delta}_t$，$\boldsymbol{\delta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)$

# %% [markdown]
# ### 3.2 预测步（Prior）
#
# $$\hat{\boldsymbol{\mu}}_t = \mathbf{A}_t \boldsymbol{\mu}_{t-1} + \mathbf{B}_t \mathbf{u}_t$$
# $$\hat{\boldsymbol{\Sigma}}_t = \mathbf{A}_t \boldsymbol{\Sigma}_{t-1} \mathbf{A}_t^T + \mathbf{Q}_t$$

# %% [markdown]
# ### 3.3 更新步（Posterior）
#
# **卡尔曼增益**：
# $$\mathbf{K}_t = \hat{\boldsymbol{\Sigma}}_t \mathbf{C}_t^T (\mathbf{C}_t \hat{\boldsymbol{\Sigma}}_t \mathbf{C}_t^T + \mathbf{R}_t)^{-1}$$
#
# **后验均值与协方差**：
# $$\boldsymbol{\mu}_t = \hat{\boldsymbol{\mu}}_t + \mathbf{K}_t (\mathbf{z}_t - \mathbf{C}_t \hat{\boldsymbol{\mu}}_t)$$
# $$\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{C}_t) \hat{\boldsymbol{\Sigma}}_t$$

# %% [markdown]
# ### 3.4 卡尔曼增益的直觉 ⭐
#
# $$K = \frac{\sigma^2_{prior}}{\sigma^2_{prior} + \sigma^2_{obs}} \in [0, 1]$$
#
# - 观测噪声大（$\mathbf{R}$ 大）→ $\mathbf{K} \to \mathbf{0}$ → 更信预测
# - 过程噪声大（$\mathbf{Q}$ 大）→ $\hat{\boldsymbol{\Sigma}}$ 大 → $\mathbf{K} \to \mathbf{I}$ → 更信观测
# - 创新（Innovation）$\mathbf{z}_t - \mathbf{C}\hat{\boldsymbol{\mu}}_t$ 是预测观测与实际观测的差异

# %% [markdown]
# ## 4. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.estimation import KalmanFilter
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 4.1 2D 位置跟踪（恒速模型）

# %%
dt_kf = 0.1; N_kf = 150

# 状态 [x, y, vx, vy]
A = np.array([[1,0,dt_kf,0],[0,1,0,dt_kf],[0,0,1,0],[0,0,0,1]])
B = np.zeros((4,1))
C = np.array([[1,0,0,0],[0,1,0,0]])  # 只观测位置
Q = np.diag([0.001, 0.001, 0.01, 0.01])  # 过程噪声（加速度不确定性）
R = np.diag([0.5, 0.5])  # 观测噪声

kf = KalmanFilter(A, B, C, Q, R, mu=np.array([0,0,1.0,0.5]), Sigma=np.eye(4)*0.5)

# 仿真
true_x = np.zeros((N_kf, 4)); true_x[0] = [0,0,1.0,0.5]
z_hist = np.zeros((N_kf, 2))
mu_hist = np.zeros((N_kf, 4))
sigma_hist = np.zeros((N_kf, 4))

for t in range(1, N_kf):
    # 真值（恒速 + 随机加速扰动）
    true_x[t] = A @ true_x[t-1] + rng.multivariate_normal(np.zeros(4), Q)
    z_hist[t] = C @ true_x[t] + rng.multivariate_normal(np.zeros(2), R)
    mu_hist[t], Sigma_t, _ = kf.step(z_hist[t])
    sigma_hist[t] = np.sqrt(np.diag(Sigma_t))
mu_hist[0] = kf.mu; sigma_hist[0] = np.sqrt(np.diag(kf.Sigma))

t_kf = np.arange(N_kf)*dt_kf
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for dim, (ax1, ax2, d_label) in enumerate(zip(axes[0], axes[1], ['X', 'Y'])):
    ax1.plot(t_kf, true_x[:, dim], 'k-', linewidth=2, label='True')
    ax1.plot(t_kf, z_hist[:, dim], 'r.', markersize=3, alpha=0.5, label='Measured')
    ax1.plot(t_kf, mu_hist[:, dim], 'b-', linewidth=2, label='KF Estimate')
    ax1.fill_between(t_kf, mu_hist[:,dim]-2*sigma_hist[:,dim], mu_hist[:,dim]+2*sigma_hist[:,dim], color='blue', alpha=0.15)
    ax1.set_ylabel(f'{d_label} (m)'); ax1.set_title(f'{d_label} Position'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(t_kf, mu_hist[:, dim+2], 'b-', linewidth=2, label=f'v_{d_label.lower()} estimate')
    ax2.set_ylabel(f'{d_label} vel (m/s)'); ax2.set_xlabel('t (s)'); ax2.set_title(f'{d_label} Velocity (unobserved!)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/22_kf_2d_tracking.png', dpi=100, bbox_inches='tight')
plt.show()

print("虽然传感器只测位置不测速度，KF 仍然可以估计速度！")
print("这是 KF 的核心价值之一：利用模型从间接测量中推断不可观测状态。")

# %% [markdown]
# ### 4.2 卡尔曼增益收敛

# %%
fig, ax = plt.subplots(figsize=(10, 5))
K_trace = np.zeros((N_kf, 4, 2))
kf2 = KalmanFilter(A, B, C, Q, R, mu=np.array([0,0,1,0.5]), Sigma=np.eye(4)*0.5)
for t in range(N_kf):
    kf2.predict()
    K_trace[t] = kf2.Sigma @ C.T @ np.linalg.inv(C @ kf2.Sigma @ C.T + R)
    kf2.update(z_hist[t] if t < len(z_hist) else np.zeros(2))

for i in range(4):
    ax.plot(t_kf, K_trace[:, i, 0], linewidth=1.5, label=f'K[{i},0] (x-pos gain)')
ax.set_xlabel('t (s)'); ax.set_ylabel('Kalman Gain');
ax.set_title('Kalman Gain Convergence (independent of measurements!)')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/22_kalman_gain.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.3 不同 Q/R 的影响

# %%
configs = [('Q large, R small', 0.1, 0.01), ('Balanced', 0.01, 0.1), ('Q small, R large', 0.001, 1.0)]
fig, ax = plt.subplots(figsize=(10, 5))
t_short = np.arange(100)*dt_kf

for label, Q_s, R_s in configs:
    A1d = np.array([[1, dt_kf], [0, 1]]); C1d = np.array([[1, 0]])
    kf1d = KalmanFilter(A1d, np.zeros((2,1)), C1d, np.diag([0, Q_s]), np.array([[R_s]]),
                        mu=np.array([0, 1.0]), Sigma=np.eye(2)*0.5)
    mu1d = np.zeros((100, 2))
    for t in range(1, 100):
        mu1d[t], _, _ = kf1d.step(true_x[t, 0] + rng.normal(0, np.sqrt(R_s)))
    ax.plot(t_short, mu1d[:, 0], linewidth=1.5, label=f'{label}')

ax.plot(t_short, true_x[:100, 0], 'k-', linewidth=2, alpha=0.7, label='True')
ax.set_xlabel('t (s)'); ax.set_ylabel('x (m)')
ax.set_title('KF Behavior with Different Q/R Ratios')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/22_qr_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. 练习题
#
# ### 概念题
# 1. 卡尔曼增益 K 的公式中每项的含义？极端情况下（R→∞ 或 Q→∞）K 趋向什么？
# 2. 为什么 KF 的 Σ_t 不依赖观测 z_t？有什么好处？
#
# ### 编程题
# 1. 实现信息滤波（Information Filter）版本。
# 2. 对 2R 臂的关节速度估计问题应用 KF。
#
# > 答案见 `solutions/22_solutions.ipynb`

# %% [markdown]
# ## 6. 本节总结
#
# | 步骤 | 公式 | 计算 |
# |------|------|------|
# | 预测均值 | $\hat{\boldsymbol{\mu}} = \mathbf{A}\boldsymbol{\mu} + \mathbf{B}\mathbf{u}$ | O(n²) |
# | 预测协方差 | $\hat{\boldsymbol{\Sigma}} = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T + \mathbf{Q}$ | O(n³) |
# | 卡尔曼增益 | $\mathbf{K} = \hat{\boldsymbol{\Sigma}}\mathbf{C}^T(\mathbf{C}\hat{\boldsymbol{\Sigma}}\mathbf{C}^T + \mathbf{R})^{-1}$ | O(m³) |
# | 更新均值 | $\boldsymbol{\mu} = \hat{\boldsymbol{\mu}} + \mathbf{K}(\mathbf{z} - \mathbf{C}\hat{\boldsymbol{\mu}})$ | O(nm) |
# | 更新协方差 | $\boldsymbol{\Sigma} = (\mathbf{I} - \mathbf{K}\mathbf{C})\hat{\boldsymbol{\Sigma}}$ | O(n²m) |
