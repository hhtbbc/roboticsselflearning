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
# # Notebook 17b：状态空间、线性化与 LQR

# ## 1. 定位
# NB17-18 讨论了非线性控制器（重力补偿 PD, CTC）。本节建立线性系统理论——状态空间表示、沿轨迹线性化、LQR 最优控制。这是 MPC 和 LQG 的基础。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ 连续和离散状态空间表示
# - ⭐ 平衡点线性化与沿轨迹线性化
# - ⭐ LQR 推导：Riccati 方程
# - ⭐ LQR 与重力补偿 PD 的联系
# - 📖 可控性与可观测性

# %% [markdown]
# ## 3. 状态空间表示 ⭐

# %% [markdown]
# 连续非线性：$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$
#
# 2R 臂的状态空间（$\mathbf{x} = [\mathbf{q}; \dot{\mathbf{q}}]$）：
# $$\dot{\mathbf{x}} = \begin{bmatrix} \dot{\mathbf{q}} \\ \mathbf{M}^{-1}(\mathbf{q})(\boldsymbol{\tau} - \mathbf{C}\dot{\mathbf{q}} - \mathbf{g}) \end{bmatrix}$$

# %% [markdown]
# ### 3.1 平衡点线性化
# 在 $(\mathbf{x}_{eq}, \mathbf{u}_{eq})$ 附近：
# $$\delta\dot{\mathbf{x}} = \mathbf{A}\delta\mathbf{x} + \mathbf{B}\delta\mathbf{u}$$
# $$\mathbf{A} = \left.\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\right|_{eq}, \quad \mathbf{B} = \left.\frac{\partial\mathbf{f}}{\partial\mathbf{u}}\right|_{eq}$$

# %% [markdown]
# ## 4. LQR ⭐

# %% [markdown]
# 最优控制问题：最小化 $J = \int_0^\infty (\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{u}^T\mathbf{R}\mathbf{u}) dt$，受 $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$
#
# 解：$\mathbf{u}^* = -\mathbf{K}\mathbf{x}$，其中 $\mathbf{K} = \mathbf{R}^{-1}\mathbf{B}^T\mathbf{P}$
#
# $\mathbf{P}$ 是代数 Riccati 方程的解：
# $$\mathbf{A}^T\mathbf{P} + \mathbf{P}\mathbf{A} - \mathbf{P}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{P} + \mathbf{Q} = \mathbf{0}$$

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import sys; sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 倒立摆线性化 + LQR

# %%
# 倒立摆: θ̈ = (g/l)sin(θ) + τ/(ml²)
g, l_pend, m_pend = 9.81, 1.0, 1.0
# 状态 x = [θ, θ̇], 平衡点 (0,0) (直立)
# A = [[0, 1], [g/l, 0]], B = [[0], [1/(ml²)]]
A_pend = np.array([[0, 1], [g/l_pend, 0]])
B_pend = np.array([[0], [1/(m_pend*l_pend**2)]])

Q_pend = np.diag([100, 1])  # 重视角度偏差
R_pend = np.array([[0.1]])   # 控制代价

P_pend = solve_continuous_are(A_pend, B_pend, Q_pend, R_pend)
K_pend = np.linalg.solve(R_pend, B_pend.T @ P_pend)
print(f"LQR 增益 K = {np.round(K_pend, 4)}")

# 仿真闭环
dt_sim = 0.01; T_sim = 3.0; N = int(T_sim/dt_sim)
x = np.array([0.3, 0.0])  # 初始偏离 0.3 rad
x_hist = [x.copy()]; u_hist = []

for _ in range(N):
    u = -K_pend @ x
    u_hist.append(u[0])
    xdot = A_pend @ x + B_pend.flatten() * u
    x = x + xdot * dt_sim
    x_hist.append(x.copy())
x_hist = np.array(x_hist)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t = np.linspace(0, T_sim, N+1)
axes[0].plot(t, x_hist[:,0], 'b-', linewidth=2)
axes[0].set_ylabel('θ (rad)'); axes[0].set_title('LQR Stabilization — Inverted Pendulum'); axes[0].grid(True, alpha=0.3)
axes[1].plot(t[:-1], u_hist, 'r-', linewidth=2)
axes[1].set_ylabel('u (Nm)'); axes[1].set_xlabel('t (s)'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/17b_lqr_pendulum.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.2 2R 臂沿轨迹 LQR

# %%
dyn = TwoLinkArmDynamics(m1=1., m2=1., l1=1., l2=0.8, g=9.81)

# 在平衡点 q_eq = [π/4, π/6] 线性化
q_eq = np.array([np.pi/4, np.pi/6])
tau_eq = dyn.gravity_vector(q_eq)  # 平衡力矩抵消重力

# 数值线性化 A (4×4), B (4×2)
eps = 1e-6
def f_closed(x, tau):
    q, qd = x[:2], x[2:]
    qdd = dyn.forward_dynamics(q, qd, tau)
    return np.concatenate([qd, qdd])

x_eq = np.concatenate([q_eq, np.zeros(2)])
A_2r = np.zeros((4, 4)); B_2r = np.zeros((4, 2))
f0 = f_closed(x_eq, tau_eq)

for i in range(4):
    xp = x_eq.copy(); xp[i] += eps
    A_2r[:, i] = (f_closed(xp, tau_eq) - f0) / eps
for i in range(2):
    up = tau_eq.copy(); up[i] += eps
    B_2r[:, i] = (f_closed(x_eq, up) - f0) / eps

Q_2r = np.diag([100, 80, 10, 8]); R_2r = np.diag([0.1, 0.1])
P_2r = solve_continuous_are(A_2r, B_2r, Q_2r, R_2r)
K_2r = np.linalg.solve(R_2r, B_2r.T @ P_2r)

# LQR 闭环仿真
x_lqr = np.concatenate([q_eq + np.array([0.2, -0.15]), np.zeros(2)])
x_lqr_hist = [x_lqr.copy()]
for _ in range(500):
    dx = x_lqr - x_eq; u_lqr = tau_eq - K_2r @ dx
    x_lqr = x_lqr + f_closed(x_lqr, u_lqr) * 0.005
    x_lqr_hist.append(x_lqr.copy())
x_lqr_hist = np.array(x_lqr_hist)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
t2 = np.linspace(0, 2.5, 501)
for j in range(2):
    axes[0].plot(t2, x_lqr_hist[:, j], label=f'q{j+1}')
axes[0].axhline(y=q_eq[0], c='b', ls='--', alpha=0.3)
axes[0].axhline(y=q_eq[1], c='r', ls='--', alpha=0.3)
axes[0].set_ylabel('q (rad)'); axes[0].set_title('2R Arm LQR Regulation'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(t2, x_lqr_hist[:, 2:])
axes[1].set_ylabel('q̇ (rad/s)'); axes[1].set_xlabel('t (s)'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/17b_lqr_2r.png', dpi=100, bbox_inches='tight')
plt.show()
print("LQR 成功将扰动的 2R 臂稳定回平衡点。")

# %% [markdown]
# ## 6. 练习题
# 1. LQR 的 Q 和 R 各控制什么？增大 Q 或减小 R 的效果？
# 2. 代数 Riccati 方程与李雅普诺夫方程的关系？
# 3. LQR 和重力补偿 PD 的数学联系是什么？
