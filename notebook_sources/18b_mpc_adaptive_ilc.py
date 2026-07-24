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
# # Notebook 18b：MPC、自适应控制与迭代学习控制

# ## 1. 定位
# 将 NB17-18 的非线性控制扩展到：预测未来（MPC）、在线学习参数（自适应）、从重复经验中改进（ILC）。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ MPC：receding horizon, 约束处理, QP 形式
# - ⭐ 自适应控制：$\boldsymbol{\tau} = \mathbf{Y}\hat{\boldsymbol{\theta}} - \mathbf{K}_D\mathbf{s}$, $\dot{\hat{\boldsymbol{\theta}}} = -\boldsymbol{\Gamma}^{-1}\mathbf{Y}^T\mathbf{s}$
# - ⭐ ILC：$\mathbf{u}_{k+1} = \mathbf{u}_k + \mathbf{L}\mathbf{e}_k$, Q-filter
# - 📖 持续激励、参数投影、收敛条件

# %% [markdown]
# ## 3. MPC ⭐

# %% [markdown]
# ### 3.1 核心思想
# 在每个时间步，求解一个有限时域的最优控制问题，只执行第一步控制，然后滚动向前。
#
# $$\min_{\mathbf{u}_{0:H-1}} \sum_{k=0}^{H-1} (\mathbf{x}_k^T\mathbf{Q}\mathbf{x}_k + \mathbf{u}_k^T\mathbf{R}\mathbf{u}_k) + \mathbf{x}_H^T\mathbf{Q}_f\mathbf{x}_H$$
# $$\text{s.t. } \mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k, \quad \mathbf{u}_{min} \leq \mathbf{u}_k \leq \mathbf{u}_{max}$$

# %% [markdown]
# ## 4. 自适应控制 ⭐

# %% [markdown]
# 当动力学参数 $\boldsymbol{\theta}$ 不完全已知时，在线更新估计 $\hat{\boldsymbol{\theta}}$：
#
# 控制律：$\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q}, \dot{\mathbf{q}}, \dot{\mathbf{q}}_r, \ddot{\mathbf{q}}_r)\hat{\boldsymbol{\theta}} - \mathbf{K}_D\mathbf{s}$
#
# 自适应律：$\dot{\hat{\boldsymbol{\theta}}} = -\boldsymbol{\Gamma}^{-1}\mathbf{Y}^T\mathbf{s}$
#
# 稳定性依赖于 $\dot{\mathbf{M}} - 2\mathbf{C}$ 反对称性。
# 关键区分：**参数收敛** (需要 persistent excitation) ≠ **跟踪收敛** (只需自适应律)。

# %% [markdown]
# ## 5. ILC

# %% [markdown]
# 对于重复执行同一轨迹的系统：
# $$\mathbf{u}_{k+1}(t) = Q(\mathbf{u}_k(t) + \mathbf{L}\mathbf{e}_k(t))$$
# - $Q$: Q-filter（低通，抑制高频学习）
# - $\mathbf{L}$: 学习增益
# - $\mathbf{e}_k = \mathbf{x}_d - \mathbf{x}_k$: 第 k 次试验的误差

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics
from src.robotics_learning.control import PIDController
from scipy.linalg import solve_continuous_are
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 MPC — 倒立摆带输入约束

# %%
A_p = np.array([[0, 1], [9.81, 0]]); B_p = np.array([[0], [1.0]])
dt_mpc = 0.05; H = 10  # 预测时域 10 步
# 离散化 (前向欧拉)
Ad = np.eye(2) + dt_mpc * A_p; Bd = dt_mpc * B_p

Q_mpc = np.diag([100, 1]); R_mpc = np.array([[0.1]])
# Terminal cost: LQR P∞
P_inf = solve_continuous_are(A_p, B_p, Q_mpc, R_mpc)

x_mpc = np.array([0.3, 0.0]); x_mpc_hist = [x_mpc.copy()]; u_mpc_hist = []
u_max = 5.0

for _ in range(100):
    # 简化 MPC: 无约束解析解 → 用 LQR 的 K (带输入限幅)
    K_inf = np.linalg.solve(R_mpc, Bd.T @ P_inf @ Ad)
    u_opt = -K_inf @ x_mpc
    u_opt = np.clip(u_opt, -u_max, u_max)  # 输入饱和
    x_mpc = Ad @ x_mpc + Bd.flatten() * u_opt
    x_mpc_hist.append(x_mpc.copy()); u_mpc_hist.append(u_opt)

x_mpc_hist = np.array(x_mpc_hist)
t_mpc = np.linspace(0, 100*dt_mpc, 101)

fig, axes = plt.subplots(2,1,figsize=(10,6))
axes[0].plot(t_mpc, x_mpc_hist[:,0], 'b-', linewidth=2); axes[0].set_ylabel('θ'); axes[0].grid(True, alpha=0.3)
axes[0].set_title('MPC (Receding Horizon LQR + Input Saturation)')
axes[1].plot(t_mpc[:-1], u_mpc_hist, 'r-', linewidth=2)
axes[1].axhline(y=u_max, c='k', ls='--', alpha=0.3); axes[1].axhline(y=-u_max, c='k', ls='--', alpha=0.3)
axes[1].set_ylabel('u (clipped)'); axes[1].set_xlabel('t (s)'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/18b_mpc_pendulum.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.2 自适应控制仿真

# %%
dyn_true = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=1.0, l2=0.8, g=9.81)
dyn_nom = TwoLinkArmDynamics(m1=0.7, m2=0.8, l1=1.0, l2=0.8, g=9.81)  # ~25% 参数误差

# 复合参数: θ = [α, β, δ, g1_coeff, g2_coeff]
theta_true = np.array([dyn_true.alpha, dyn_true.beta, dyn_true.delta,
                       dyn_true.m1*dyn_true.lc1*9.81+dyn_true.m2*dyn_true.l1*9.81,
                       dyn_true.m2*dyn_true.lc2*9.81])
theta_hat = np.array([dyn_nom.alpha, dyn_nom.beta, dyn_nom.delta,
                      dyn_nom.m1*dyn_nom.lc1*9.81+dyn_nom.m2*dyn_nom.l1*9.81,
                      dyn_nom.m2*dyn_nom.lc2*9.81])

Gamma_inv = np.diag([0.5]*5)  # 自适应增益
Kd = np.diag([50, 40])
Lambda_adapt = np.diag([10, 8])

dt_ad = 0.002; N_ad = 2000
qd_des = np.array([np.pi/3, np.pi/6])  # 定常目标
q = np.array([0.2, 0.1]); q_dot = np.zeros(2)
theta_hist = [theta_hat.copy()]; e_hist = []

for _ in range(N_ad):
    e = qd_des - q; s = Lambda_adapt @ e + (np.zeros(2) - q_dot)  # q̇_d = 0

    # 回归矩阵 Y (NB12)
    c2 = np.cos(q[1]); s2 = np.sin(q[1]); c1 = np.cos(q[0]); c12 = np.cos(q[0]+q[1])
    qr_ddot = np.zeros(2)  # 定常目标 → q̈_r = 0 (简化)
    Y = np.zeros((2, 5))
    Y[0] = [qr_ddot[0], 2*c2*qr_ddot[0]+c2*qr_ddot[1]-s2*q_dot[1]*(2*q_dot[0]+q_dot[1]), qr_ddot[1], c1, c12]
    Y[1] = [0, c2*qr_ddot[0]+s2*q_dot[0]**2, qr_ddot[0]+qr_ddot[1], 0, c12]
    # 简化: 使用 PID-type reference
    Y_simple = np.zeros((2, 5))
    Y_simple[0] = [0, 0, 0, c1, c12]
    Y_simple[1] = [0, 0, 0, 0, c12]

    tau = Y_simple @ theta_hat + Kd @ s
    qdd = dyn_true.forward_dynamics(q, q_dot, tau)
    q_dot += qdd * dt_ad; q += q_dot * dt_ad

    # 自适应律
    theta_hat = theta_hat + Gamma_inv @ Y_simple.T @ s * dt_ad
    theta_hist.append(theta_hat.copy()); e_hist.append(np.linalg.norm(e))

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
t_ad = np.linspace(0, N_ad*dt_ad, N_ad)
axes[0].semilogy(t_ad, e_hist, 'b-', linewidth=2)
axes[0].set_ylabel('||e||'); axes[0].set_title('Adaptive Control — Tracking Error Convergence'); axes[0].grid(True, alpha=0.3)
for i in range(5):
    axes[1].plot(np.linspace(0, N_ad*dt_ad, len(theta_hist)), np.array(theta_hist)[:,i], linewidth=1.5, label=f'θ_hat[{i}]')
for i, tv in enumerate(theta_true):
    axes[1].axhline(y=tv, ls='--', alpha=0.3)
axes[1].set_xlabel('t (s)'); axes[1].set_ylabel('θ_hat'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/18b_adaptive_control.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.3 ILC 演示

# %%
# 1D 系统: ẍ = u, 重复跟踪 sin 轨迹
T_ilc = 2.0; dt_ilc = 0.01; N_ilc = int(T_ilc/dt_ilc)
t_ilc = np.linspace(0, T_ilc, N_ilc)
xd = np.sin(2*np.pi*t_ilc/T_ilc); vd = np.gradient(xd, dt_ilc)

L_ilc = 0.3; Q_cutoff = 5  # Q-filter 截止频率 (Hz)
n_trials = 10
errors_rms = []

# 简化 Q-filter: moving average
def q_filter(u, window=3):
    u_f = u.copy()
    for i in range(window, len(u)-window):
        u_f[i] = np.mean(u[i-window:i+window+1])
    return u_f

u_ilc = np.zeros(N_ilc)
for trial in range(n_trials):
    x = 0.0; v = 0.0; x_hist = []
    for i in range(N_ilc):
        x_hist.append(x)
        a = u_ilc[i]; v += a*dt_ilc; x += v*dt_ilc
    e = xd - np.array(x_hist)
    errors_rms.append(np.sqrt(np.mean(e**2)))
    u_ilc = q_filter(u_ilc + L_ilc * e)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(t_ilc, xd, 'k-', linewidth=2, label='Desired')
for trial in [0, 3, n_trials-1]:
    x_test = 0.0; v_test = 0.0; xh = []
    u_test = q_filter(np.zeros(N_ilc))
    for _ in range(trial+1):
        u_test = q_filter(u_test + L_ilc * (xd - np.array(xh))) if len(xh)==N_ilc else np.zeros(N_ilc)
    for i in range(N_ilc):
        a = u_test[i]; v_test += a*dt_ilc; xh.append(x_test); x_test += v_test*dt_ilc
    axes[0].plot(t_ilc, xh, linewidth=1.5, alpha=0.7, label=f'Trial {trial+1}')
axes[0].set_ylabel('x'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(range(1, n_trials+1), errors_rms, 'b-o', linewidth=2)
axes[1].set_xlabel('Trial'); axes[1].set_ylabel('RMS Error'); axes[1].grid(True, alpha=0.3)
axes[1].set_title('ILC Convergence')
plt.suptitle('Iterative Learning Control — 1D Point Mass')
plt.tight_layout()
plt.savefig('../outputs/18b_ilc.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 练习题
# 1. MPC 的 receding horizon 与 terminal cost 的作用？
# 2. 自适应控制中参数收敛和跟踪收敛的区别？
# 3. ILC 的 Q-filter 为什么要做低通滤波？
