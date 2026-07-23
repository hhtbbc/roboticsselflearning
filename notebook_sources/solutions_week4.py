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

# %%
import numpy as np; import sys; sys.path.insert(0, '..')
from src.robotics_learning.estimation import KalmanFilter
from src.robotics_learning.simulation import simulate_encoder, simulate_accelerometer

# %% [markdown]
# # 第 4 周练习解答 — 操作空间、力控制与状态估计 (NB19-26)

# %% [markdown]
# ## NB19 — 零空间投影不影响末端运动
# J(I-J⁺J) = J-JJ⁺J = J-J = 0（Penrose 条件 1）

# %%
J_test=np.random.randn(2,3); Jp=np.linalg.pinv(J_test)
N=np.eye(3)-Jp@J_test
qdot_null=N@np.array([0.5,-0.3,0.2])
print(f"J·q̇_null = {np.linalg.norm(J_test@qdot_null):.2e}")

# %% [markdown]
# ## NB21 — 一维贝叶斯更新演示
# Prior: μ=0, σ²=2; Obs: z=1.5, σ²_obs=0.5

# %%
mu_prior,sigma_prior=0.0,2.0; z,sigma_obs=1.5,0.5
K=sigma_prior/(sigma_prior+sigma_obs)
mu_post=mu_prior+K*(z-mu_prior)
sigma_post=(1-K)*sigma_prior
print(f"Posterior: μ={mu_post:.3f}, σ²={sigma_post:.3f}")

# %% [markdown]
# ## NB22 — Σ_t 不依赖 z_t 的验证
# K = Σ̂C^T(CΣ̂C^T+R)⁻¹ 不包含 z

# %%
A=np.array([[1,0.1],[0,1]]); C=np.array([[1,0]])
kf=KalmanFilter(A,np.zeros((2,1)),C,np.diag([0.001,0.01]),np.array([[0.5]]),
                mu=np.array([0.,1.]),Sigma=np.eye(2)*0.5)
for t in range(20): kf.predict()
print(f"Σ (no obs, converged): {np.round(kf.Sigma,3)}")

# %% [markdown]
# ## NB23 — N_eff 退化检查

# %%
w=np.array([0.4,0.3,0.15,0.1,0.04,0.01])
N_eff=1/np.sum(w**2)
print(f"N_eff={N_eff:.1f}/6 → {'退化,需重采样' if N_eff<3 else 'OK'}")

# %% [markdown]
# ## NB24 — 编码器 vs 加速度计测量演示

# %%
rng=np.random.RandomState(42)
z_enc=simulate_encoder(np.array([0.5,-0.3]),std=0.001,rng=rng)
z_acc=simulate_accelerometer(np.array([0.,0.,-9.81]),std=0.01,rng=rng)
print(f"Encoder: {np.round(z_enc,4)}")
print(f"Accel (at rest): {np.round(z_acc,2)} ≈ [0,0,+g]")

# %% [markdown]
# ## 系统设计回顾
# 闭环: RRT规划→样条轨迹→CTC控制(用EKF估计状态)→RK4动力学→编码器噪声→EKF→反馈
# 各模块接口见 NB25 综合项目。
