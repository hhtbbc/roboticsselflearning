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
# # Notebook 19：操作空间控制
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB18 关节空间控制 ──→ NB19 操作空间控制 ──→ NB20 力/阻抗控制
# NB07 雅可比 ──→ (利用 J 做关节↔任务映射)
# ```
#
# 许多任务在关节空间中表达不自然："末端向下移动 5cm"——这需要 IK 才能转化为关节指令。操作空间控制**直接在任务空间指定运动**，通过雅可比（或其逆）映射到关节执行。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解操作空间动力学：$\boldsymbol{\Lambda}(\mathbf{x}), \boldsymbol{\mu}(\mathbf{x},\dot{\mathbf{x}}), \mathbf{p}(\mathbf{x})$
# - ⭐ 掌握 Resolved Motion Rate Control (RMRC)
# - ⭐ 掌握冗余机器人的任务优先级框架（零空间投影）
# - 📖 全身控制 WBC 简介

# %% [markdown]
# ## 3. 操作空间动力学 ⭐

# %% [markdown]
# 关节空间动力学在操作空间的"投影"：
#
# $$\boldsymbol{\Lambda}(\mathbf{x})\ddot{\mathbf{x}} + \boldsymbol{\mu}(\mathbf{x},\dot{\mathbf{x}})\dot{\mathbf{x}} + \mathbf{p}(\mathbf{x}) = \mathbf{F}$$
#
# 其中：
# - $\boldsymbol{\Lambda}(\mathbf{x}) = (\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^T)^{-1}$ — 任务空间惯性矩阵
# - $\boldsymbol{\mu}(\mathbf{x},\dot{\mathbf{x}}) = \boldsymbol{\Lambda}(\mathbf{J}\mathbf{M}^{-1}\mathbf{C}\dot{\mathbf{q}} - \dot{\mathbf{J}}\dot{\mathbf{q}})$
# - $\mathbf{p}(\mathbf{x}) = \boldsymbol{\Lambda}\mathbf{J}\mathbf{M}^{-1}\mathbf{g}$
# - 关节力矩映射：$\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$

# %% [markdown]
# ## 4. Resolved Motion Rate Control ⭐

# %% [markdown]
# 给定末端期望速度 $\dot{\mathbf{x}}_d$：
# $$\dot{\mathbf{q}} = \mathbf{J}^{-1}(\mathbf{q})\,\dot{\mathbf{x}}_d$$
#
# 加上位置误差反馈（类似于 P 控制）：
# $$\dot{\mathbf{q}} = \mathbf{J}^{-1}(\mathbf{q})\,[\dot{\mathbf{x}}_d + \mathbf{K}(\mathbf{x}_d - \mathbf{x})]$$
#
# 注意：这里用 $\mathbf{J}^{-1}$ 是因为末端维度 = 关节维度。对于冗余臂 $(n > m)$，用伪逆 $\mathbf{J}^+$。

# %% [markdown]
# ## 5. 冗余机器人的任务优先级 ⭐

# %% [markdown]
# ### 5.1 零空间投影
#
# 当 $n > m$ 时，满足 $\dot{\mathbf{x}} = \mathbf{J}\dot{\mathbf{q}}$ 的解不唯一。通解：
#
# $$\dot{\mathbf{q}} = \mathbf{J}^+\dot{\mathbf{x}} + (\mathbf{I} - \mathbf{J}^+\mathbf{J})\dot{\mathbf{q}}_0$$
#
# - $\mathbf{J}^+\dot{\mathbf{x}}$：**特解**——产生期望末端速度的最小范数关节速度
# - $(\mathbf{I} - \mathbf{J}^+\mathbf{J})\dot{\mathbf{q}}_0$：**零空间分量**——不影响末端运动！可用于执行次任务

# %% [markdown]
# ### 5.2 多级任务堆叠
#
# 例如：主任务 = 末端跟踪轨迹；次任务 = 避免关节限位
# $$\dot{\mathbf{q}} = \mathbf{J}_1^+\dot{\mathbf{x}}_1 + \mathbf{N}_1[\mathbf{J}_2^+\dot{\mathbf{x}}_2 + \mathbf{N}_2\dot{\mathbf{q}}_0]$$
# 其中 $\mathbf{N}_1 = \mathbf{I} - \mathbf{J}_1^+\mathbf{J}_1$。

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.kinematics import forward_kinematics, compute_geometric_jacobian
from src.robotics_learning.dynamics import TwoLinkArmDynamics, simulate_dynamics
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 冗余 3R 臂零空间运动演示

# %%
# 3R 平面臂 (n=3, m=2 — 冗余!)
l1, l2, l3 = 1.0, 0.8, 0.5
dh_3r = np.array([[l1, 0, 0], [l2, 0, 0], [l3, 0, 0]])

q = np.array([np.pi/6, np.pi/4, -np.pi/3])
dh_full = np.column_stack([dh_3r, q])
_, transforms = forward_kinematics(dh_full)
p_ee_orig = transforms[-1][:2, 3]  # 原始末端位置

# 零空间运动：末端不动，但关节在动
dt_ns = 0.05
q_traj = [q.copy()]
p_ee_traj = [p_ee_orig]

for _ in range(100):
    # 次任务：最小化关节2的角度（让肘部向下）
    q0_dot = np.array([0.0, -0.5, 0.3])
    J = compute_geometric_jacobian(dh_3r, q)[:2, :]  # 只取 XY
    J_pinv = np.linalg.pinv(J)
    N = np.eye(3) - J_pinv @ J
    q_dot = N @ q0_dot  # 主任务 ẋ=0，所以只用零空间
    q = q + q_dot * dt_ns
    q_traj.append(q.copy())
    # 检查末端是否真没动
    dh_f = np.column_stack([dh_3r, q])
    _, tr = forward_kinematics(dh_f)
    p_ee_traj.append(tr[-1][:2, 3])

q_traj = np.array(q_traj)
p_ee_traj = np.array(p_ee_traj)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(np.degrees(q_traj), linewidth=1.5)
axes[0].set_xlabel('Step'); axes[0].set_ylabel('q (°)')
axes[0].set_title('Joint Motion in Null Space'); axes[0].legend(['q₁', 'q₂', 'q₃']); axes[0].grid(True, alpha=0.3)

axes[1].plot(p_ee_traj[:, 0], p_ee_traj[:, 1], 'b-o', markersize=3)
axes[1].scatter(*p_ee_orig, c='green', s=150, marker='o', label='Initial EE', zorder=5)
axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
axes[1].set_title('End-Effector Position (should NOT move)')
axes[1].set_aspect('equal'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].annotate(f'Δp = {np.linalg.norm(p_ee_traj[-1]-p_ee_traj[0]):.2e} m', (p_ee_orig[0]+0.1, p_ee_orig[1]))
plt.tight_layout()
plt.savefig('../outputs/19_null_space_motion.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"末端位移: {np.linalg.norm(p_ee_traj[-1] - p_ee_traj[0]):.2e} m ≈ 0")

# %% [markdown]
# ### 6.2 Resolved Motion Rate Control

# %%
# 给定圆形末端轨迹，用 RMRC 跟踪
n_steps = 300
t_rmrc = np.linspace(0, 2*np.pi, n_steps)
x_d_circle = 1.2 + 0.3*np.cos(t_rmrc)  # 圆心(1.2, 0)，半径0.3
y_d_circle = 0.3*np.sin(t_rmrc)
xy_des = np.column_stack([x_d_circle, y_d_circle])

q_rmrc = np.array([np.pi/3, -np.pi/4])  # 2R 臂初值
dh_2r = np.array([[1.0, 0, 0], [0.8, 0, 0]])
q_rmrc_hist = [q_rmrc.copy()]; ee_hist = []

K_rmrc = 20.0  # 反馈增益
dt_rmrc = 0.02

for i in range(n_steps):
    _, tr = forward_kinematics(np.column_stack([dh_2r, q_rmrc]))
    p_ee = tr[-1][:2, 3]; ee_hist.append(p_ee)

    # 期望速度 + 位置误差反馈
    xd_dot = (xy_des[i] - xy_des[max(i-1,0)]) / dt_rmrc if i > 0 else np.zeros(2)
    x_err = xy_des[i] - p_ee
    xd_cmd = xd_dot + K_rmrc * x_err

    J = compute_geometric_jacobian(dh_2r, q_rmrc)[:2, :]
    q_rmrc += np.linalg.solve(J, xd_cmd) * dt_rmrc  # J q̇ = ẋ_command
    q_rmrc_hist.append(q_rmrc.copy())

q_rmrc_hist = np.array(q_rmrc_hist); ee_hist = np.array(ee_hist)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(xy_des[:, 0], xy_des[:, 1], 'k--', linewidth=1, alpha=0.5, label='Desired Circle')
ax.plot(ee_hist[:, 0], ee_hist[:, 1], 'b-', linewidth=2, label='EE Trajectory')
ax.scatter(*ee_hist[0], c='green', s=100, label='Start')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('Resolved Motion Rate Control — Circle Tracking')
ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/19_rmrc_circle.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 练习题
#
# ### 概念题
# 1. RMRC 和逆运动学有何区别和联系？
# 2. 零空间投影 $\mathbf{I} - \mathbf{J}^+\mathbf{J}$ 为什么不影响末端运动？
#
# ### 编程题
# 1. 实现三级任务堆叠：位置 → 姿态 → 关节角中心化。
# 2. 在操作空间中实现 CTC（基于 $\boldsymbol{\Lambda}, \boldsymbol{\mu}, \mathbf{p}$）。
#
# > 答案见 `solutions/19_solutions.ipynb`

# %% [markdown]
# ## 8. 本节总结
#
# | 概念 | 公式 | 说明 |
# |------|------|------|
# | 任务空间质量 | $\boldsymbol{\Lambda} = (\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^T)^{-1}$ | 末端"有效惯性" |
# | RMRC | $\dot{\mathbf{q}} = \mathbf{J}^{-1}(\dot{\mathbf{x}}_d + \mathbf{K}\mathbf{e})$ | 速度级控制 |
# | 伪逆解 | $\dot{\mathbf{q}} = \mathbf{J}^+\dot{\mathbf{x}}$ | 最小范数解 |
# | 零空间 | $\dot{\mathbf{q}}_{null} = (\mathbf{I} - \mathbf{J}^+\mathbf{J})\dot{\mathbf{q}}_0$ | 次任务空间 |
