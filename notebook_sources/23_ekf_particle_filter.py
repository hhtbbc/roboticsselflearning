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
# # Notebook 23：扩展卡尔曼滤波与粒子滤波
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB22 KF ──→ NB23 EKF + PF ──→ NB24 传感器融合
# (线性)     (非线性系统)      (多传感器)
# ```
#
# 现实中的机器人系统几乎都是**非线性**的。EKF 通过一阶泰勒展开来近似非线性，粒子滤波则避开线性/高斯假设。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 掌握 EKF 的线性化方法（状态转移雅可比 $\mathbf{A}_t$、观测雅可比 $\mathbf{C}_t$）
# - ⭐ 理解 EKF 的局限性（线性化误差、不一致性）
# - ⭐ 掌握 SIR 粒子滤波的完整流程（采样→加权→重采样）
# - ⭐ 理解 $N_{eff}$ 和重采样的必要性
# - 📖 UKF 的基本思想

# %% [markdown]
# ## 3. EKF ⭐

# %% [markdown]
# ### 3.1 线性化
#
# 非线性系统：
# $$\mathbf{x}_t = \mathbf{f}(\mathbf{x}_{t-1}, \mathbf{u}_t) + \boldsymbol{\epsilon}_t$$
# $$\mathbf{z}_t = \mathbf{h}(\mathbf{x}_t) + \boldsymbol{\delta}_t$$
#
# 在估计均值处一阶泰勒展开：
# $$\mathbf{A}_t = \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right|_{\boldsymbol{\mu}_{t-1}, \mathbf{u}_t}, \quad \mathbf{C}_t = \left.\frac{\partial \mathbf{h}}{\partial \mathbf{x}}\right|_{\hat{\boldsymbol{\mu}}_t}$$

# %% [markdown]
# ### 3.2 EKF vs KF 的区别
#
# | 步骤 | KF | EKF |
# |------|----|-----|
# | 预测均值 | $\mathbf{A}\boldsymbol{\mu} + \mathbf{B}\mathbf{u}$ | $\mathbf{f}(\boldsymbol{\mu}, \mathbf{u})$（非线性函数） |
# | 预测协方差 | $\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T + \mathbf{Q}$ | 同（但 $\mathbf{A}$ 是雅可比） |
# | 更新均值 | $\hat{\boldsymbol{\mu}} + \mathbf{K}(\mathbf{z} - \mathbf{C}\hat{\boldsymbol{\mu}})$ | $\hat{\boldsymbol{\mu}} + \mathbf{K}(\mathbf{z} - \mathbf{h}(\hat{\boldsymbol{\mu}}))$ |
#
# **关键**：均值传播用真实的非线性函数 $\mathbf{f}$ 和 $\mathbf{h}$；协方差传播用雅可比。

# %% [markdown]
# ### 3.3 EKF 的局限性
#
# - **线性化误差**：在高度非线性区域，一阶近似很差
# - **不一致性（Inconsistency）**：EKF 倾向于低估协方差（过于自信）
# - **不适用于多峰分布**：高斯假设意味着单峰

# %% [markdown]
# ## 4. 粒子滤波 ⭐

# %% [markdown]
# ### 4.1 核心思想
#
# 用 $M$ 个加权粒子 $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^M$ 近似任意分布：
# $$p(\mathbf{x}) \approx \sum_{i=1}^M w^{(i)} \delta(\mathbf{x} - \mathbf{x}^{(i)})$$
#
# 不需要高斯假设，可以表示多峰分布（如全局定位中"门可能在左边也可能在右边"）。

# %% [markdown]
# ### 4.2 SIR 流程
#
# 1. **预测**：从提议分布采样 $\mathbf{x}^{(i)}_t \sim q(\mathbf{x}_t \mid \mathbf{x}^{(i)}_{t-1}, \mathbf{u}_t, \mathbf{z}_t)$
# 2. **更新权重**：$w^{(i)}_t \propto w^{(i)}_{t-1} \frac{p(\mathbf{z}_t \mid \mathbf{x}^{(i)}_t) p(\mathbf{x}^{(i)}_t \mid \mathbf{x}^{(i)}_{t-1}, \mathbf{u}_t)}{q(\mathbf{x}^{(i)}_t \mid \dots)}$
# 3. **重采样**：当 $N_{eff} = 1/\sum(w^{(i)})^2 < N/2$ 时，重采样粒子

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.estimation import ExtendedKalmanFilter, ParticleFilter
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 EKF 移动机器人定位

# %%
dt_ekf = 0.1; N_ekf = 200
# 非线性运动模型：独轮车
# x_{t+1} = x_t + v*cos(θ)*dt, y_{t+1} = y_t + v*sin(θ)*dt, θ_{t+1} = θ_t + ω*dt
def f_ekf(x, u):
    v, omega = u[0], u[1]
    return np.array([x[0]+v*np.cos(x[2])*dt_ekf, x[1]+v*np.sin(x[2])*dt_ekf, x[2]+omega*dt_ekf])

def A_func(x, u):
    v = u[0]
    return np.array([[1,0,-v*np.sin(x[2])*dt_ekf],[0,1,v*np.cos(x[2])*dt_ekf],[0,0,1]])

# 观测模型：距离+方位到两个地标
landmarks = np.array([[5,5],[8,2],[2,8]])
def h_ekf(x):
    z = []
    for lm in landmarks:
        dx, dy = lm[0]-x[0], lm[1]-x[1]
        r = np.sqrt(dx**2+dy**2)
        b = np.arctan2(dy, dx) - x[2]
        b = np.arctan2(np.sin(b), np.cos(b))
        z.extend([r, b])
    return np.array(z)

def C_func(x):
    C = np.zeros((6,3))
    for i, lm in enumerate(landmarks):
        dx, dy = lm[0]-x[0], lm[1]-x[1]
        r_sq = dx**2+dy**2; r = np.sqrt(r_sq)
        C[2*i] = [-dx/r, -dy/r, 0]
        C[2*i+1] = [dy/r_sq, -dx/r_sq, -1]
    return C

Q_ekf = np.diag([0.01, 0.01, 0.005])
R_ekf = np.diag([0.3, 0.05]*3)

ekf = ExtendedKalmanFilter(f_ekf, h_ekf, A_func, C_func, Q_ekf, R_ekf,
                           mu=np.array([1,1,np.pi/4]), Sigma=np.eye(3)*0.5)

true_pose = np.zeros((N_ekf, 3)); true_pose[0] = [1,1,np.pi/4]
mu_efk = np.zeros((N_ekf, 3)); mu_efk[0] = ekf.mu
mu_efk = np.zeros((N_ekf, 3))

for t in range(1, N_ekf):
    u = np.array([0.5, 0.1*np.sin(t*dt_ekf)])
    true_pose[t] = f_ekf(true_pose[t-1], u) + rng.multivariate_normal(np.zeros(3), Q_ekf)
    z_true = h_ekf(true_pose[t])
    z_noisy = z_true + rng.multivariate_normal(np.zeros(6), R_ekf)
    mu_efk[t], _, _ = ekf.step(z_noisy, u)
mu_efk[0] = ekf.mu

fig, ax = plt.subplots(figsize=(10, 9))
ax.plot(true_pose[:,0], true_pose[:,1], 'k-', linewidth=2, label='True Trajectory')
ax.plot(mu_efk[:,0], mu_efk[:,1], 'b--', linewidth=2, label='EKF Estimate')
ax.scatter(landmarks[:,0], landmarks[:,1], c='red', s=100, marker='s', label='Landmarks')
for i, (tx, te) in enumerate(zip(true_pose[::20], mu_efk[::20])):
    cov_xy = ekf.Sigma[:2,:2]
    ev, evec = np.linalg.eigh(cov_xy)
    ell = plt.matplotlib.patches.Ellipse(te[:2], 2*np.sqrt(ev[0]), 2*np.sqrt(ev[1]),
           angle=np.degrees(np.arctan2(evec[1,0],evec[0,0])), facecolor='blue', alpha=0.08, edgecolor='blue', lw=0.5)
    ax.add_patch(ell)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title(f'EKF Robot Localization (RMSE: {np.sqrt(np.mean(np.sum((true_pose[:,:2]-mu_efk[:,:2])**2, axis=1))):.3f}m)')
ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/23_ekf_localization.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.2 粒子滤波 — 全局重定位

# %%
# 简化：2D 位置 (x, y)，速度控制 + 距离测量
def f_pf(x, u):
    return x + u * dt_ekf + rng.normal(0, 0.05, 2)

def h_pf(x):
    z = []
    for lm in landmarks[:,:2]:
        z.append(np.linalg.norm(x[:2] - lm))
    return np.array(z)

pf = ParticleFilter(200, 2, f_pf, h_pf, np.array([0.05,0.05]), np.array([0.3,0.3,0.3]),
                    bounds=np.array([[0,10],[0,10]]), rng=rng)
pf.initialize_uniform(np.array([[0,10],[0,10]]))

pf_hist = np.zeros((N_ekf, 2))
for t in range(N_ekf):
    u = np.array([0.5*np.cos(true_pose[t,2]), 0.5*np.sin(true_pose[t,2])])
    z = h_pf(true_pose[t,:2]) + rng.normal(0, 0.3, 3)
    pf.step(z, u)
    pf_hist[t], _ = pf.estimate()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 初始：扩散的粒子
ax1.scatter(pf.particles[:,0], pf.particles[:,1], s=5, c='blue', alpha=0.3)
ax1.scatter(*true_pose[0,:2], c='green', s=200, marker='*', label='True Start', zorder=5)
ax1.scatter(landmarks[:,0], landmarks[:,1], c='red', s=80, marker='s', label='Landmarks')
ax1.set_title(f'Particle Filter — t=0 (Uniform, N_eff={pf.neff():.0f})')
ax1.set_xlim([0,10]); ax1.set_ylim([0,10]); ax1.set_aspect('equal'); ax1.legend()

# 最终：收敛的粒子
ax2.scatter(pf.particles[:,0], pf.particles[:,1], s=15, c='blue', alpha=0.5)
ax2.scatter(*true_pose[N_ekf-1,:2], c='green', s=200, marker='*', label='True End', zorder=5)
ax2.scatter(landmarks[:,0], landmarks[:,1], c='red', s=80, marker='s')
ax2.scatter(*pf_hist[N_ekf-1], c='orange', s=150, marker='x', label='PF Estimate', zorder=5, linewidths=2)
ax2.set_title(f'Particle Filter — t={N_ekf*dt_ekf:.0f}s (Converged, N_eff={pf.neff():.0f})')
ax2.set_xlim([0,10]); ax2.set_ylim([0,10]); ax2.set_aspect('equal'); ax2.legend()

plt.tight_layout()
plt.savefig('../outputs/23_particle_filter.png', dpi=100, bbox_inches='tight')
plt.show()

print("粒子滤波成功：从完全未知（均匀分布）收敛到目标位置！")
print(f"最终估计误差: {np.linalg.norm(pf_hist[-1] - true_pose[-1,:2]):.3f}m")

# %% [markdown]
# ## 6. 练习题
#
# ### 概念题
# 1. EKF 在什么情况下会发散？
# 2. 粒子滤波的 $N_{eff}$ 为什么需要监控？重采样频率的权衡？
#
# ### 编程题
# 1. 实现 UKF 并对 EKF/UKF/PF 在强非线性场景下做详细对比。
# 2. 对 2R 臂的关节角度+速度做 EKF 估计。
#
# > 答案见 `solutions/23_solutions.ipynb`

# %% [markdown]
# ## 7. 本节总结
#
# | 算法 | 分布假设 | 线性假设 | 计算量 | 适合场景 |
# |------|:--------:|:--------:|:------:|----------|
# | KF | 高斯 | 线性 | 低 | 线性系统 |
# | EKF | 高斯 | 局部线性 | 中 | 适度非线性 |
# | UKF | 高斯 | 无 | 中高 | 强非线性 |
# | 粒子滤波 | 任意 | 无 | 高 | 多峰、全局定位 |
