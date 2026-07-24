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
# # Notebook 16b：轨迹优化 — Direct Shooting, Collocation 与平滑

# %% [markdown]
# ## 1. 定位
#
# NB15-16 的 A*/RRT 回答"走哪条路"（几何路径），NB13-14 回答"多快走"（时间参数化）。轨迹优化将两者统一——直接优化含时间、动力学和约束的完整轨迹。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 区分几何路径规划 / 运动学约束规划 / 动力学可行规划 / 轨迹优化
# - ⭐ 理解 direct shooting and direct collocation
# - ⭐ 实现 RRT 路径的 shortcut 平滑
# - 📖 CHOMP / STOMP / TrajOpt 的基本思想

# %% [markdown]
# ## 3. 概念区分 ⭐

# %% [markdown]
# | 层次 | 输出 | 约束 | 方法 |
# |------|------|------|------|
# | 几何路径规划 | $\mathbf{q}(s)$ 无时间 | 仅碰撞 | A*, RRT |
# | 运动学约束规划 | $\mathbf{q}(s)$ | 碰撞 + 速度/加速度 | Kinodynamic RRT |
# | 动力学可行规划 | $\mathbf{q}(s)$ 或 $\mathbf{q}(t)$ | 碰撞 + 动力学 | TrajOpt, direct collocation |
# | 轨迹优化 | $\mathbf{q}(t), \dot{\mathbf{q}}(t), \ddot{\mathbf{q}}(t), \boldsymbol{\tau}(t)$ | 全约束 + 代价最小化 | Direct transcription |

# %% [markdown]
# ## 4. Path Shortcut 平滑 ⭐

# %% [markdown]
# RRT 产生的路径通常曲折、包含冗余节点。Shortcut 算法通过反复尝试连接路径上随机两点的直线段（若此段无碰撞则替代中间节点），逐步平滑路径。

# %% [markdown]
# ## 5. Direct Collocation 基本思想

# %% [markdown]
# 将连续轨迹离散为 $N$ 个节点 $\mathbf{x}_k = [\mathbf{q}_k, \dot{\mathbf{q}}_k]$。
#
# 约束：相邻节点满足动力学积分（如 Euler: $\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t \cdot \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$）。
#
# 优化变量：所有节点的状态和控制。
#
# 代价：$\sum_k (\text{控制代价} + \text{状态偏差})$。
#
# 结果是一个大规模稀疏 NLP，用 IPOPT 或 SNOPT 求解。

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from src.robotics_learning.planning import rrt_plan, edge_collision_free
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 Shortcut 平滑

# %%
def shortcut_smooth(path, collision_fn, max_iter=200):
    """反复尝试连接随机两点，若直线无碰撞则替代中间节点"""
    path = list(path)
    for _ in range(max_iter):
        if len(path) <= 2: break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i <= 1: continue
        # 检查直线段 (path[i] → path[j]) 是否无碰撞
        if edge_collision_free(path[i], path[j], collision_fn):
            path = path[:i+1] + path[j:]  # 删除中间节点
    return np.array(path)

# 2D 平面碰撞函数
obstacles = [np.array([3,5]), np.array([6,3]), np.array([2,2])]; obs_r = [0.8, 0.6, 0.5]
def col2d(q):
    for c,r in zip(obstacles, obs_r):
        if np.linalg.norm(q[:2]-c) < r: return False
    return 0<=q[0]<=10 and 0<=q[1]<=10

path_raw, _ = rrt_plan(col2d, np.array([[0,10]]*2), np.array([1.,1.]), np.array([9.,9.]),
                        max_iter=500, step_size=0.3, rng=rng)
path_smooth = shortcut_smooth(path_raw, col2d, max_iter=100)

fig, ax = plt.subplots(figsize=(8, 8))
for c,r in zip(obstacles, obs_r):
    ax.add_patch(plt.Circle(c, r, color='red', alpha=0.3))
ax.plot(np.array(path_raw)[:,0], np.array(path_raw)[:,1], 'b-', alpha=0.4, linewidth=1, label=f'RRT ({len(path_raw)} pts)')
ax.plot(path_smooth[:,0], path_smooth[:,1], 'g-', linewidth=3, label=f'Smoothed ({len(path_smooth)} pts)')
ax.scatter(1,1,c='green',s=100);ax.scatter(9,9,c='red',s=100)
ax.set_xlim([0,10]);ax.set_ylim([0,10]);ax.set_aspect('equal');ax.legend();ax.grid(True,alpha=0.3)
ax.set_title('Path Shortcut Smoothing')
plt.tight_layout()
plt.savefig('../outputs/16b_shortcut_smoothing.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.2 Direct Shooting 简化演示

# %%
# 1D 质点从 x0=0 到 xf=1，控制力 u(t)，最小化 ∫ u² dt
# Direct shooting: 离散为 N 段，每段常力 u_k
N_shoot = 10; dt_shoot = 0.1; x0_shoot = 0.0; xf_shoot = 1.0
# 决策变量: u[0:N]
# 约束: x_N = xf（终态约束）
# 代价: Σ u_k²

def shoot(u):
    x = x0_shoot; v = 0.0
    for uk in u: v += uk*dt_shoot; x += v*dt_shoot
    return x

# 简化：用无约束最小化（penalty）
from scipy.optimize import minimize
u_init = np.zeros(N_shoot)

def cost_and_constraint(u):
    xf = shoot(u)
    return np.sum(u**2) + 1000*(xf - xf_shoot)**2

res = minimize(cost_and_constraint, u_init, method='BFGS')
u_opt = res.x

# 前向积分显示
x_hist = [0.0]; v_hist = [0.0]; x = 0.0; v = 0.0
for uk in u_opt:
    v += uk*dt_shoot; x += v*dt_shoot
    x_hist.append(x); v_hist.append(v)

fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
t_shoot = np.linspace(0, N_shoot*dt_shoot, N_shoot+1)
axes[0].plot(t_shoot, x_hist, 'b-o', linewidth=2); axes[0].axhline(y=1, color='k', linestyle='--')
axes[0].set_ylabel('x'); axes[0].set_title('Direct Shooting — 1D Point Mass')
axes[1].plot(t_shoot, v_hist, 'g-o', linewidth=2); axes[1].set_ylabel('v')
axes[2].step(t_shoot[:-1], u_opt, 'r-', linewidth=2, where='post'); axes[2].set_ylabel('u (force)'); axes[2].set_xlabel('t (s)')
for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/16b_direct_shooting.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 练习题
#
# ### 概念题
# 1. Direct shooting 和 direct collocation 的区别？
# 2. Shortcut 平滑可能增加碰撞风险吗？为什么需要复检？
#
# ### 编程题
# 1. 实现 B-spline 轨迹平滑并在平滑后做碰撞检查。
# 2. 对 2R 臂做 direct collocation 简化版（3 个节点、动力学约束）。
