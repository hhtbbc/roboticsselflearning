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
# # Notebook 17：关节空间控制 — PID、重力补偿与李雅普诺夫稳定性
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB09-10 动力学 ──→ NB17 关节空间控制 ──→ NB18 计算力矩控制
# NB13 轨迹生成 ──→ (q_d, q̇_d 作为参考输入)
# ```
#
# 控制器是机器人的"肌肉"——给定参考轨迹和当前状态，计算出每个关节需要多少力矩。从最简单的独立关节 PID 到基于模型的重力补偿 PD，逐步建立控制理论直觉。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 掌握独立关节 PID 控制的结构和局限性
# - ⭐ 理解重力补偿 PD 如何消除稳态误差
# - ⭐ **能用李雅普诺夫方法严格证明重力补偿 PD 的全局渐近稳定性**
# - ⭐ 理解 $(\dot{\mathbf{M}} - 2\mathbf{C})$ 反对称性在稳定性证明中的关键作用
# - 📖 PID 积分抗饱和策略
# - 📚 级联控制（Cascade Control）

# %% [markdown]
# ## 3. 独立关节 PID ⭐
#
# 将每个关节视为独立 SISO 系统——忽略所有耦合项：
#
# $$\tau_i = K_{p,i} e_i + K_{d,i} \dot{e}_i + K_{i,i} \int e_i \,dt$$
#
# 其中 $e_i = q_{d,i} - q_i$。
#
# **问题**：耦合项 $\mathbf{C}\dot{\mathbf{q}} + \mathbf{g}$ 被当作"未知干扰"——在高速/高精度场景下跟踪性能差，有稳态误差。

# %% [markdown]
# ## 4. 重力补偿 PD ⭐
#
# ### 4.1 控制律
#
# $$\boldsymbol{\tau} = \mathbf{K}_p \mathbf{e} + \mathbf{K}_d \dot{\mathbf{e}} + \mathbf{g}(\mathbf{q})$$
#
# 加入了重力前馈项 $\mathbf{g}(\mathbf{q})$ 来抵消重力力矩。这是"基于部分模型"的控制——只用了 $\mathbf{g}(\mathbf{q})$，不需要 $\mathbf{M}(\mathbf{q})$ 或 $\mathbf{C}$。

# %% [markdown]
# ### 4.2 李雅普诺夫稳定性证明 ⭐ 面试必会！
#
# 设 $V(\mathbf{e}, \dot{\mathbf{q}})$ 为候选李雅普诺夫函数（Lyapunov Function Candidate）：
#
# $$V = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q})\dot{\mathbf{q}} + \frac{1}{2}\mathbf{e}^T \mathbf{K}_p \mathbf{e}$$
#
# - **第一项**：系统动能 $\frac{1}{2}\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}}$ > 0
# - **第二项**：虚拟弹簧势能 $\frac{1}{2}\mathbf{e}^T\mathbf{K}_p\mathbf{e}$ > 0
# - 因此 $V > 0$（当 $(\mathbf{e}, \dot{\mathbf{q}}) \neq \mathbf{0}$ 时）
#
# 求导：
# $$\dot{V} = \dot{\mathbf{q}}^T\mathbf{M}\ddot{\mathbf{q}} + \frac{1}{2}\dot{\mathbf{q}}^T\dot{\mathbf{M}}\dot{\mathbf{q}} - \dot{\mathbf{q}}^T\mathbf{K}_p\mathbf{e}$$
#
# 注意 $\ddot{\mathbf{q}} = -\ddot{\mathbf{e}}$（因为 $\mathbf{q}_d$ 是定常参考 $\ddot{\mathbf{q}}_d = 0$）。
#
# 代入动力学方程 $\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g} = \boldsymbol{\tau} = \mathbf{K}_p\mathbf{e} + \mathbf{K}_d\dot{\mathbf{e}} + \mathbf{g}$：
# $$\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} = \mathbf{K}_p\mathbf{e} - \mathbf{K}_d\dot{\mathbf{q}}$$
#
# 将 $\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\mathbf{K}_p\mathbf{e} - \mathbf{K}_d\dot{\mathbf{q}} - \mathbf{C}\dot{\mathbf{q}})$ 代入 $\dot{V}$：
#
# $$\begin{aligned}
# \dot{V} &= \dot{\mathbf{q}}^T(\mathbf{K}_p\mathbf{e} - \mathbf{K}_d\dot{\mathbf{q}} - \mathbf{C}\dot{\mathbf{q}}) + \frac{1}{2}\dot{\mathbf{q}}^T\dot{\mathbf{M}}\dot{\mathbf{q}} - \dot{\mathbf{q}}^T\mathbf{K}_p\mathbf{e} \\
# &= -\dot{\mathbf{q}}^T\mathbf{K}_d\dot{\mathbf{q}} + \frac{1}{2}\dot{\mathbf{q}}^T(\dot{\mathbf{M}} - 2\mathbf{C})\dot{\mathbf{q}}
# \end{aligned}$$
#
# 由于 $\dot{\mathbf{M}} - 2\mathbf{C}$ 是**反对称矩阵**（NB09），二次型 $\dot{\mathbf{q}}^T(\dot{\mathbf{M}} - 2\mathbf{C})\dot{\mathbf{q}} = 0$。
#
# $$\therefore \dot{V} = -\dot{\mathbf{q}}^T \mathbf{K}_d \dot{\mathbf{q}} \leq 0$$
#
# $\dot{V}$ 是**半负定**的。通过 LaSalle 不变原理可以进一步证明全局渐近稳定性。
#
# 这个证明是机器人控制面试中**最高频的理论推导题**。

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics, simulate_dynamics
from src.robotics_learning.control import PIDController, gravity_compensation_pd
from src.robotics_learning.trajectory import quintic_trajectory
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 PD vs 重力补偿 PD 对比

# %%
dyn = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=1.0, l2=0.8, g=9.81)
dt = 0.002; T_sim = 3.0

# 期望轨迹：五次多项式从静止到目标
_, q_d_traj, qd_d_traj, _, _ = quintic_trajectory(q0=0.0, qf=np.pi/3, v0=0, vf=0, a0=0, af=0, T=T_sim, dt=dt)

def run_controller(name, tau_fn):
    q = np.zeros(2); q_dot = np.zeros(2)
    q_hist = [q.copy()]; q_dot_hist = [q_dot.copy()]; tau_hist = [np.zeros(2)]
    for i in range(len(q_d_traj)-1):
        q_d = np.array([q_d_traj[i], q_d_traj[i]*0.5])
        qd_d = np.array([qd_d_traj[i], qd_d_traj[i]*0.5])
        tau = tau_fn(q_d, qd_d, q, q_dot)
        q_ddot = dyn.forward_dynamics(q, q_dot, tau)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt
        q_hist.append(q.copy()); q_dot_hist.append(q_dot.copy()); tau_hist.append(tau)
    return np.array(q_hist), np.array(tau_hist)

# 纯 PD（无重力补偿）
q_pd, tau_pd = run_controller("PD", lambda qd,qdd,q,qdot: np.diag([50,30]) @ (qd-q) + np.diag([10,8]) @ (qdd-qdot))

# 重力补偿 PD
q_gc, tau_gc = run_controller("GC-PD",
    lambda qd,qdd,q,qdot: gravity_compensation_pd(qd, qdd, q, qdot, np.array([50,30]), np.array([10,8]), dyn.gravity_vector))

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
t = np.linspace(0, T_sim, len(q_pd))

for col, (q_hist, tau_hist, label) in enumerate([
    (q_pd, tau_pd, 'Pure PD'),
    (q_gc, tau_gc, 'Gravity Compensation PD')
]):
    for j, c in enumerate(['blue', 'red']):
        axes[0,col].plot(t, q_hist[:, j], color=c, linewidth=1.5, label=f'q{j+1}')
        axes[1,col].plot(t, q_d_traj[:len(q_hist)]*(0.5 if j==1 else 1) - q_hist[:, j], color=c, linewidth=1.5)
        axes[2,col].plot(t, tau_hist[:, j], color=c, linewidth=1.5)

    axes[0,col].set_title(f'{label}\nq_d (black dashed)'); axes[0,col].set_ylabel('q (rad)'); axes[0,col].grid(True, alpha=0.3)
    axes[1,col].set_ylabel('e (rad)'); axes[1,col].grid(True, alpha=0.3)
    axes[2,col].set_ylabel('τ (Nm)'); axes[2,col].set_xlabel('t (s)'); axes[2,col].grid(True, alpha=0.3)

for j in range(2):
    axes[0,j].plot(t, q_d_traj[:len(q_hist)], 'k--', linewidth=1, alpha=0.5)
    axes[0,j].plot(t, q_d_traj[:len(q_hist)]*0.5, 'k--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('../outputs/17_pd_vs_gc_pd.png', dpi=100, bbox_inches='tight')
plt.show()

# 稳态误差对比
e_pd_final = np.mean(np.abs(q_d_traj[-100:] - q_pd[-100:, 0]))
e_gc_final = np.mean(np.abs(q_d_traj[-100:] - q_gc[-100:, 0]))
print(f"纯 PD 末端稳态误差: {e_pd_final:.4f} rad")
print(f"重力补偿 PD 末端稳态误差: {e_gc_final:.4f} rad")
print(f"改善: {(1 - e_gc_final/(e_pd_final+1e-10))*100:.1f}%")

# %% [markdown]
# ### 5.2 不同增益的响应对比

# %%
Kp_values = [20, 60, 150]
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
dt_s = 0.005
q0_test, qd_test = 0.0, 1.0

for Kp in Kp_values:
    q, q_dot = np.array([q0_test, 0.0]), np.zeros(2)
    err_hist = []
    for _ in range(int(1.0/dt_s)):
        q_d = np.array([qd_test, 0.5])
        tau = gravity_compensation_pd(q_d, np.zeros(2), q, q_dot, np.array([Kp, Kp*0.6]), np.array([Kp*0.2, Kp*0.12]), dyn.gravity_vector)
        q_ddot = dyn.forward_dynamics(q, q_dot, tau)
        q_dot += q_ddot * dt_s; q += q_dot * dt_s
        err_hist.append(q_d[0] - q[0])

    t_s = np.linspace(0, 1.0, len(err_hist))
    axes[0].plot(t_s, qd_test - np.array(err_hist), linewidth=1.5, label=f'Kp={Kp}')
    axes[1].plot(t_s, err_hist, linewidth=1.5)

axes[0].axhline(y=qd_test, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('q₁ (rad)'); axes[0].set_title('Step Response with Different Kp'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_ylabel('Error e₁ (rad)'); axes[1].set_xlabel('t (s)'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/17_step_response.png', dpi=100, bbox_inches='tight')
plt.show()

print("Kp 越大 → 上升时间越短，但超调增加（Kd 不够大时可能振荡）")

# %% [markdown]
# ## 6. 练习题
#
# ### 概念题
# 1. 为什么纯 PD（无重力补偿）会有稳态误差？
# 2. 李雅普诺夫函数 $V$ 中每一项的物理含义是什么？
# 3. $(\dot{\mathbf{M}} - 2\mathbf{C})$ 反对称性在稳定性证明中如何"消掉"？
#
# ### 编程题
# 1. 实现带积分抗饱和的 PID 控制器，对比纯 PD。
# 2. 用 LaSalle 不变原理说明重力补偿 PD 的全局渐近稳定性。
#
# > 答案见 `solutions/17_solutions.ipynb`

# %% [markdown]
# ## 7. 本节总结
#
# | 控制器 | 控制律 | 优点 | 缺点 |
# |--------|--------|------|------|
# | PD | $\mathbf{K}_p\mathbf{e} + \mathbf{K}_d\dot{\mathbf{e}}$ | 简单、不需模型 | 有稳态误差 |
# | PID | 加积分项 | 消除稳态误差 | 积分饱和、慢 |
# | 重力补偿 PD | $+\mathbf{g}(\mathbf{q})$ | 消除重力导致的稳态误差 | 需要重力模型 |
# | 前馈 | $+\boldsymbol{\tau}_{des}$ | 减小跟踪延迟 | 需要期望力矩 |
