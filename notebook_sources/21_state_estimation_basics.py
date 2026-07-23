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
# # Notebook 21：概率与状态估计基础
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB01 线性代数 ──→ NB21 概率基础 ──→ NB22 卡尔曼滤波
#                       │
#                       └── 贝叶斯滤波框架 ──→ NB23 EKF/PF
# ```
#
# 机器人永远活在不确定性中——传感器有噪声、模型不完备、环境不可预测。状态估计用概率论的数学工具来"从不确定信息中提取最可能的真相"。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解高斯分布及其协方差矩阵的几何含义
# - ⭐ 掌握贝叶斯滤波的"预测—更新"递归框架
# - ⭐ 理解传感器模型和运动模型的概率表示
# - ⭐ 理解协方差传播（不确定性传播）
# - 📖 信息形式（信息矩阵 Ω、信息向量 ξ）

# %% [markdown]
# ## 3. 高斯分布 ⭐

# %% [markdown]
# ### 3.1 多元高斯
#
# $$p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^n|\boldsymbol{\Sigma}|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$
#
# - $\boldsymbol{\mu} \in \mathbb{R}^n$：均值
# - $\boldsymbol{\Sigma} \in \mathbb{R}^{n\times n}$：协方差矩阵（对称正定）
# - 等概率面是椭球 $(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) = \text{const}$

# %% [markdown]
# ### 3.2 高斯的线性变换
#
# 若 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则 $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$：
# $$\mathbf{y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$
#
# 这是卡尔曼滤波中协方差传播的基础！

# %% [markdown]
# ### 3.3 信息形式
#
# - 信息矩阵：$\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$
# - 信息向量：$\boldsymbol{\xi} = \boldsymbol{\Omega}\boldsymbol{\mu}$
#
# - 完全不确定（$\boldsymbol{\Sigma} \to \infty$）→ $\boldsymbol{\Omega} \to \mathbf{0}$
# - 信息滤波在循环闭包（loop closure）检测中数值上更高效

# %% [markdown]
# ## 4. 贝叶斯滤波框架 ⭐

# %% [markdown]
# ### 4.1 问题设定
#
# - **状态** $\mathbf{x}_t$：机器人想知道的量（如位置、速度）
# - **观测** $\mathbf{z}_t$：传感器测量值（有噪声）
# - **控制** $\mathbf{u}_t$：机器人施加的动作（有噪声）
#
# 目标：递归计算 $p(\mathbf{x}_t \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$——给定所有历史测量和控制，当前状态的概率分布。

# %% [markdown]
# ### 4.2 两步递推
#
# **预测步（Prediction / Prior）**——利用运动模型猜测下一个状态：
# $$p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t}) = \int p(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{u}_t) \, p(\mathbf{x}_{t-1} \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t-1}) \, d\mathbf{x}_{t-1}$$
#
# **更新步（Update / Posterior）**——用新观测修正预测：
# $$p(\mathbf{x}_t \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) \propto p(\mathbf{z}_t \mid \mathbf{x}_t) \, p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})$$
#
# - **先验** × **似然** ∝ **后验**
# - 如果 $p(\mathbf{z}_t \mid \mathbf{x}_t)$ 很准（低噪声）→ 后验更接近观测 → 不确定性缩小

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
sys.path.insert(0, '..')
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 2D 高斯分布可视化

# %%
mu = np.array([2.0, 3.0])
Sigma = np.array([[1.5, 0.8], [0.8, 1.0]])

# 生成采样
rng = np.random.RandomState(42)
samples = rng.multivariate_normal(mu, Sigma, 500)

# 等概率椭圆
eigvals, eigvecs = np.linalg.eigh(Sigma)
theta = np.linspace(0, 2*np.pi, 200)
ellipse_1sigma = np.array([np.cos(theta), np.sin(theta)]).T @ np.diag(np.sqrt(eigvals)) @ eigvecs.T + mu
ellipse_2sigma = ellipse_1sigma.copy()
ellipse_2sigma = np.array([2*np.cos(theta), 2*np.sin(theta)]).T @ np.diag(np.sqrt(eigvals)) @ eigvecs.T + mu

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.4, label='Samples')
ax.plot(ellipse_1sigma[:, 0], ellipse_1sigma[:, 1], 'b-', linewidth=2, label='1σ (68%)')
ax.plot(ellipse_2sigma[:, 0], ellipse_2sigma[:, 1], 'b--', linewidth=2, label='2σ (95%)')
ax.scatter(*mu, c='red', s=100, marker='x', linewidths=2, label='μ')
# 画特征向量
for i in range(2):
    ax.arrow(mu[0], mu[1], 2*np.sqrt(eigvals[i])*eigvecs[0, i], 2*np.sqrt(eigvals[i])*eigvecs[1, i],
             head_width=0.15, fc='darkred', ec='darkred', linewidth=2)

ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
ax.set_title('2D Gaussian: μ=[2,3], Σ=[[1.5,0.8],[0.8,1.0]]')
ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/21_gaussian_2d.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.2 协方差传播演示

# %%
A = np.array([[1.5, 0.3], [-0.2, 0.8]])
b = np.array([1.0, -0.5])
Sigma_y = A @ Sigma @ A.T
mu_y = A @ mu + b

samples_y = (A @ samples.T).T + b

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pts, m, s, title in [
    (axes[0], samples, mu, Sigma, 'Input: x ~ N(μ, Σ)'),
    (axes[1], samples_y, mu_y, Sigma_y, 'Output: y = Ax + b ~ N(Aμ+b, AΣA^T)')
]:
    ev, evec = np.linalg.eigh(s)
    ell = np.array([np.cos(theta), np.sin(theta)]).T @ np.diag(np.sqrt(ev)) @ evec.T + m
    ax.scatter(pts[:,0], pts[:,1], s=5, alpha=0.4)
    ax.plot(ell[:,0], ell[:,1], 'r-', linewidth=2, label='1σ')
    ax.scatter(*m, c='red', s=80, marker='x')
    ax.set_title(title); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/21_covariance_propagation.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.3 一维贝叶斯滤波仿真

# %%
# 机器人沿 1D 直线运动，每隔一步用噪声传感器测量位置
dt_bf = 1.0; N = 20
true_x = np.zeros(N); z = np.zeros(N)
x_bf = 0.0; true_x[0] = x_bf
mu_bf = np.zeros(N); sigma_bf = np.zeros(N)
mu_bf[0] = 0.0; sigma_bf[0] = 2.0  # 初始不确定性大

# 噪声参数
Q_proc = 0.5  # 过程噪声（运动不确定性）
R_obs = 0.8   # 观测噪声（传感器不确定性）

for t in range(1, N):
    # 真值（确定性运动 + 随机噪声）
    true_x[t] = true_x[t-1] + 1.0 + np.sqrt(Q_proc)*rng.randn()
    z[t] = true_x[t] + np.sqrt(R_obs)*rng.randn()

    # 预测：x_t = x_{t-1} + 1 + noise
    mu_pred = mu_bf[t-1] + 1.0
    sigma_pred = sigma_bf[t-1] + Q_proc

    # 更新：z_t = x_t + noise
    K = sigma_pred / (sigma_pred + R_obs)
    mu_bf[t] = mu_pred + K * (z[t] - mu_pred)
    sigma_bf[t] = (1 - K) * sigma_pred

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(N), true_x, 'k-o', linewidth=2, label='True', markersize=5)
ax.plot(range(N), z, 'r.', markersize=8, alpha=0.6, label='Measurement')
ax.plot(range(N), mu_bf, 'b-o', linewidth=2, label='Estimate', markersize=5)
ax.fill_between(range(N), mu_bf-2*np.sqrt(sigma_bf), mu_bf+2*np.sqrt(sigma_bf),
                color='blue', alpha=0.15, label='±2σ')
ax.set_xlabel('Time step'); ax.set_ylabel('Position')
ax.set_title('1D Bayesian Filtering (Prior × Likelihood = Posterior)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/21_bayes_filter_1d.png', dpi=100, bbox_inches='tight')
plt.show()

print("注意不确定性 ±2σ 随时间变窄（观测帮助缩小协方差）")

# %% [markdown]
# ## 6. 练习题
#
# ### 概念题
# 1. 贝叶斯滤波的"预测—更新"两步各用什么信息？
# 2. 为什么协方差矩阵必须对称正定？
#
# ### 编程题
# 1. 实现二维贝叶斯定位器（用离散网格近似）。
# 2. 对比不同过程噪声和观测噪声对滤波精度的影响。
#
# > 答案见 `solutions/21_solutions.ipynb`

# %% [markdown]
# ## 7. 本节总结
#
# | 概念 | 公式/符号 | 含义 |
# |------|-----------|------|
# | 高斯分布 | $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | 均值和协方差完全定义 |
# | 协方差传播 | $\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$ | 线性变换后的不确定性 |
# | 预测步 | $p(\mathbf{x}_t\|\mathbf{z}_{1:t-1})$ | 用运动模型外推 |
# | 更新步 | $p(\mathbf{x}_t\|\mathbf{z}_{1:t})$ | 用观测修正 |
# | 信息矩阵 | $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$ | 完全不确定→0 |
