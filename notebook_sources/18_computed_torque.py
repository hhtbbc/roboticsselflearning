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
# # Notebook 18：计算力矩控制与反馈线性化
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB09-10 动力学(M/C/g) ──→ NB18 计算力矩控制 ──→ NB19 操作空间控制
# NB17 基础控制 ──→ (模型前馈 + 反馈校正)
# ```
#
# 当动力学模型已知时，计算力矩控制（CTC）利用 $\mathbf{M}(\mathbf{q}), \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}), \mathbf{g}(\mathbf{q})$ 将非线性机器人系统**精确线性化**为一个解耦的二阶线性系统。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解反馈线性化的核心思想：用模型抵消非线性
# - ⭐ 掌握 CTC 控制律的推导和闭环误差动力学
# - ⭐ 分析无模型误差时的指数稳定性
# - ⭐ 理解模型误差对 CTC 性能的影响
# - 📖 滑模控制（SMC）基础
# - 📚 自适应控制与迭代学习控制入门

# %% [markdown]
# ## 3. 反馈线性化与 CTC ⭐

# %% [markdown]
# ### 3.1 核心思想
#
# 动力学方程：$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
#
# 如果我们可以选择 $\boldsymbol{\tau}$ 来**抵消** $\mathbf{C}\dot{\mathbf{q}} + \mathbf{g}$ 并**塑造** $\mathbf{M}$ 的输入，就能将非线性系统变为线性系统。

# %% [markdown]
# ### 3.2 控制律
#
# $$\boxed{\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})[\ddot{\mathbf{q}}_d + \mathbf{K}_v(\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + \mathbf{K}_p(\mathbf{q}_d - \mathbf{q})] + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})}$$
#
# - **前馈部分**（基于模型）：$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}_d + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g}$
# - **反馈部分**（误差纠正）：$\mathbf{M}(\mathbf{q})[\mathbf{K}_v\dot{\mathbf{e}} + \mathbf{K}_p\mathbf{e}]$

# %% [markdown]
# ### 3.3 闭环误差动力学
#
# 将控制律代入动力学方程，$\mathbf{M}\ddot{\mathbf{q}}$ 消去，$\mathbf{C}\dot{\mathbf{q}}$ 消去，$\mathbf{g}$ 消去：
#
# $$\ddot{\mathbf{e}} + \mathbf{K}_v\dot{\mathbf{e}} + \mathbf{K}_p\mathbf{e} = \mathbf{0}$$
#
# 这是一个**解耦的二阶线性系统**！选择 $\mathbf{K}_p = \omega_n^2\mathbf{I}, \mathbf{K}_v = 2\zeta\omega_n\mathbf{I}$。$\omega_n$ 控制响应速度，$\zeta$ 控制阻尼。

# %% [markdown]
# ### 3.4 模型不准确时
#
# 使用带有误差的名义模型 $\hat{\mathbf{M}}, \hat{\mathbf{C}}, \hat{\mathbf{g}}$：
# $$\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g} = \hat{\mathbf{M}}(\ddot{\mathbf{q}}_d + \mathbf{K}_v\dot{\mathbf{e}} + \mathbf{K}_p\mathbf{e}) + \hat{\mathbf{C}}\dot{\mathbf{q}} + \hat{\mathbf{g}}$$
#
# 整理后：
# $$\ddot{\mathbf{e}} + \mathbf{K}_v\dot{\mathbf{e}} + \mathbf{K}_p\mathbf{e} = \mathbf{M}^{-1}[(\mathbf{M} - \hat{\mathbf{M}})\ddot{\mathbf{q}} + (\mathbf{C} - \hat{\mathbf{C}})\dot{\mathbf{q}} + (\mathbf{g} - \hat{\mathbf{g}})]$$
#
# 右边是**模型误差项**——如果模型不准，误差不会收敛到零，而会收敛到有界误差。误差界的大小与模型误差 $\propto \omega_n$ 有关。

# %% [markdown]
# ## 4. 滑模控制基础

# %% [markdown]
# ### 4.1 滑模面
#
# $$\mathbf{s} = \dot{\mathbf{e}} + \boldsymbol{\Lambda}\mathbf{e}$$
#
# $\mathbf{s} = \mathbf{0}$ 是滑模面（sliding surface）。当系统在滑模面上时，误差以 $\dot{\mathbf{e}} = -\boldsymbol{\Lambda}\mathbf{e}$ 的速率指数收敛。

# %% [markdown]
# ### 4.2 控制律
#
# $$\boldsymbol{\tau} = \hat{\mathbf{M}}(\ddot{\mathbf{q}}_d + \boldsymbol{\Lambda}\dot{\mathbf{e}}) + \hat{\mathbf{C}}\dot{\mathbf{q}} + \hat{\mathbf{g}} - \mathbf{K}\,\text{sgn}(\mathbf{s})$$
#
# 其中 $\text{sgn}(\mathbf{s})$ 是切换项——使系统状态向滑模面趋近。为减少抖振，用饱和函数 $\text{sat}(\mathbf{s}/\Phi)$ 替代 sign。

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics, simulate_dynamics
from src.robotics_learning.control import computed_torque_control, sliding_mode_control
from src.robotics_learning.trajectory import quintic_trajectory
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 CTC vs 重力补偿 PD 轨迹跟踪

# %%
dyn = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=1.0, l2=0.8, g=9.81)
# 用带误差的名义模型模拟模型不确定性
dyn_nominal = TwoLinkArmDynamics(m1=0.9, m2=0.95, l1=1.0, l2=0.82, g=9.81)  # ~10% 误差

dt = 0.002; T_sim = 3.0
_, q_d_traj, qd_d_traj, qdd_d_traj, _ = quintic_trajectory(
    q0=0.0, qf=np.pi/2, v0=0, vf=0, a0=0, af=0, T=T_sim, dt=dt)

def run_ctc(name, use_true_model=True, Kp_vec=None, Kd_vec=None):
    if Kp_vec is None: Kp_vec = np.array([400, 300])
    if Kd_vec is None: Kd_vec = np.array([40, 30])
    q = np.zeros(2); q_dot = np.zeros(2)
    q_hist = [q.copy()]; tau_hist = [np.zeros(2)]
    Md = use_true_model and dyn or dyn_nominal

    for i in range(len(q_d_traj)-1):
        q_d = np.array([q_d_traj[i], q_d_traj[i]*0.4])
        qd_d = np.array([qd_d_traj[i], qd_d_traj[i]*0.4])
        qdd_d = np.array([qdd_d_traj[i], qdd_d_traj[i]*0.4])

        tau = computed_torque_control(q_d, qd_d, qdd_d, q, q_dot, Kp_vec, Kd_vec,
                                       Md.mass_matrix, Md.coriolis_matrix, Md.gravity_vector)
        q_ddot = dyn.forward_dynamics(q, q_dot, tau)
        q_dot = q_dot + q_ddot * dt; q = q + q_dot * dt
        q_hist.append(q.copy()); tau_hist.append(tau)
    return np.array(q_hist), np.array(tau_hist)

q_ctc_true, tau_ctc_true = run_ctc("CTC-True", True)
q_ctc_nom, tau_ctc_nom = run_ctc("CTC-Nom", False)

# 重力补偿 PD 基准
q_gc, _ = (lambda qh, th:
    (np.array(qh), np.array(th)))(*([np.zeros(2)], [np.zeros(2)]))
q_gc_list, q, q_dot = [np.zeros(2)], np.zeros(2), np.zeros(2)
for i in range(len(q_d_traj)-1):
    q_d = np.array([q_d_traj[i], q_d_traj[i]*0.4])
    qd_d = np.array([qd_d_traj[i], qd_d_traj[i]*0.4])
    tau = np.diag([300, 200]) @ (q_d - q) + np.diag([30, 20]) @ (qd_d - q_dot) + dyn.gravity_vector(q)
    q_ddot = dyn.forward_dynamics(q, q_dot, tau)
    q_dot += q_ddot * dt; q += q_dot * dt
    q_gc_list.append(q.copy())
q_gc = np.array(q_gc_list)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
t = np.linspace(0, T_sim, len(q_ctc_true))
labels = ['CTC (True Model)', 'CTC (~10% Error)', 'GC-PD']
colors = ['blue', 'green', 'red']
for idx, (q_hist, c, l) in enumerate(zip([q_ctc_true, q_ctc_nom, q_gc], colors, labels)):
    e = q_d_traj[:len(q_hist)] - q_hist[:, 0]
    axes[0,0].plot(t, q_hist[:, 0], c, linewidth=1.5, alpha=0.7)
    axes[1,0].plot(t, e, c, linewidth=1.5, label=l)

axes[0,0].plot(t, q_d_traj[:len(q_ctc_true)], 'k--', linewidth=1, alpha=0.5, label='Desired')
axes[0,0].set_ylabel('q₁ (rad)'); axes[0,0].set_title('Joint 1 Trajectory Tracking'); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)
axes[1,0].set_ylabel('e₁ (rad)'); axes[1,0].set_title('Tracking Error'); axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

for idx, (q_hist, c, l) in enumerate(zip([q_ctc_true, q_ctc_nom, q_gc], colors, labels)):
    e = q_d_traj[:len(q_hist)]*0.4 - q_hist[:, 1]
    axes[0,1].plot(t, q_hist[:, 1], c, linewidth=1.5, alpha=0.7)
    axes[1,1].plot(t, e, c, linewidth=1.5)
axes[0,1].plot(t, q_d_traj[:len(q_ctc_true)]*0.4, 'k--', linewidth=1, alpha=0.5)
axes[0,1].set_ylabel('q₂ (rad)'); axes[0,1].set_title('Joint 2'); axes[0,1].grid(True, alpha=0.3)
axes[1,1].set_ylabel('e₂ (rad)'); axes[1,1].set_xlabel('t (s)'); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/18_ctc_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

rms_true = np.sqrt(np.mean((q_d_traj[:len(q_ctc_true)] - q_ctc_true[:, 0])**2))
rms_nom = np.sqrt(np.mean((q_d_traj[:len(q_ctc_nom)] - q_ctc_nom[:, 0])**2))
rms_gc = np.sqrt(np.mean((q_d_traj[:len(q_gc)] - q_gc[:, 0])**2))
print(f"RMS 跟踪误差: CTC-True={rms_true:.4f}, CTC-Nom={rms_nom:.4f}, GC-PD={rms_gc:.4f}")

# %% [markdown]
# ### 5.2 滑模控制演示

# %%
dt_smc = 0.001
_, q_d_smc, qd_d_smc, qdd_d_smc, _ = quintic_trajectory(q0=0.0, qf=1.5, v0=0, vf=0, a0=0, af=0, T=2.0, dt=dt_smc)

q_smc = np.zeros(2); q_dot_smc = np.zeros(2)
q_smc_hist = [q_smc.copy()]; s_hist = []

for i in range(len(q_d_smc)-1):
    q_d = np.array([q_d_smc[i], q_d_smc[i]*0.3])
    qd_d = np.array([qd_d_smc[i], qd_d_smc[i]*0.3])
    qdd_d = np.array([qdd_d_smc[i], qdd_d_smc[i]*0.3])
    tau = sliding_mode_control(q_d, qd_d, qdd_d, q_smc, q_dot_smc,
                                np.array([10, 8]), np.array([15, 12]),
                                dyn.mass_matrix, dyn.coriolis_matrix, dyn.gravity_vector,
                                boundary=0.05)
    q_ddot = dyn.forward_dynamics(q_smc, q_dot_smc, tau)
    q_dot_smc += q_ddot * dt_smc; q_smc += q_dot_smc * dt_smc
    q_smc_hist.append(q_smc.copy())
    s = (qd_d - q_dot_smc) + np.diag([10, 8]) @ (q_d - q_smc)
    s_hist.append(s)

q_smc_hist = np.array(q_smc_hist)
s_hist = np.array(s_hist)
t_smc = np.linspace(0, 2.0, len(s_hist))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t_smc, q_d_smc[:len(s_hist)], 'k--', linewidth=1, alpha=0.5, label='Desired')
axes[0].plot(t_smc, q_smc_hist[:len(s_hist), 0], 'b-', linewidth=1.5, label='q₁ (SMC)')
axes[0].set_ylabel('q₁ (rad)'); axes[0].set_title('Sliding Mode Control Tracking'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_smc, s_hist[:, 0], 'r-', linewidth=1.5, label='s₁ = ė₁ + λe₁')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Sliding Variable s'); axes[1].set_xlabel('t (s)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title('Convergence to Sliding Surface (s → 0)')
plt.tight_layout()
plt.savefig('../outputs/18_sliding_mode.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. 练习题
#
# ### 概念题
# 1. CTC 如何将非线性系统转化为线性系统？写出关键消去步骤。
# 2. 模型误差对 CTC 的影响用什么方程描述？
#
# ### 编程题
# 1. 对比 CTC、SMC 和重力补偿 PD 在不同模型误差水平下的跟踪性能。
# 2. 实现自适应控制器：在线更新 $\hat{\boldsymbol{\theta}}$。
#
# > 答案见 `solutions/18_solutions.ipynb`

# %% [markdown]
# ## 7. 本节总结
#
# | 控制器 | 依赖模型 | 鲁棒性 | 需要 |
# |--------|:--------:|:------:|------|
# | PD | 无 | 较差 | 仅 q, q̇ |
# | GC-PD | g(q) | 中 | g(q) |
# | CTC | M, C, g（完整） | 模型准确时最优 | M, C, g 全部 |
# | SMC | M, C, g（名义） | 强鲁棒 | 需设计 K 和边界层 |
| 自适应 | 部分结构 | 在线学习 | 参数线性化 Yθ |
