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
# # Notebook 16c：运动学约束规划与 Kinodynamic RRT

# %% [markdown]
# ## 1. 定位
#
# 标准 RRT 在 C-space 中连接任意两点，不考虑"机器人能不能沿这条边运动"。Kinodynamic RRT 用**微分约束**替代直线连接——新节点不再是直线扩展，而是前向仿真一段满足运动学/动力学的轨迹。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解微分约束（differential constraints）的含义
# - ⭐ 区分几何 RRT / 运动学 RRT / Kinodynamic RRT
# - ⭐ 实现简单的独轮车 Kinodynamic RRT
# - 📖 State lattice 与 belief-space planning 基础

# %% [markdown]
# ## 3. 微分约束

# %% [markdown]
# 独轮车模型：$\dot{x} = v\cos\theta, \dot{y} = v\sin\theta, \dot{\theta} = \omega$
#
# 这是一个**非完整约束**——虽然状态空间 3 维 $(x,y,\theta)$，但不能沿任意方向移动。从 $(0,0,0)$ 到 $(0,1,0)$ 必须绕弧线，不能侧移。
#
# 标准 RRT 假设可以在 C-space 中沿任意方向直线移动（完整约束），这在独轮车、汽车等系统上不成立。

# %% [markdown]
# ## 4. Kinodynamic RRT ⭐

# %% [markdown]
# ### 4.1 与标准 RRT 的区别
#
# | | 标准 RRT | Kinodynamic RRT |
# |---|---|---|
# | 扩展方式 | 直线插值 q_near → q_new | 前向仿真 x_near 施加控制 u 一段时间 |
# | 距离度量 | Euclidean in C-space | 需要最优控制代价（如到达时间） |
# | 连接性检查 | 边碰撞检测 | 整条仿真轨迹的碰撞检测 |

# %% [markdown]
# ### 4.2 算法
#
# ```
# 1. 随机采样状态 x_rand
# 2. 找最近节点 x_near（使用到达代价度量）
# 3. 选择控制 u* 使 x_near 在 Δt 后最接近 x_rand
# 4. 前向仿真: x_new = propagate(x_near, u*, Δt)
# 5. 若轨迹无碰撞: 加入树
# ```

# %% [markdown]
# ## 5. Python 实现 — 独轮车 Kinodynamic RRT

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %%
def unicycle_propagate(x, u, dt):
    """独轮车状态传播: x = [x, y, theta], u = [v, omega]"""
    return np.array([x[0] + u[0]*np.cos(x[2])*dt,
                     x[1] + u[0]*np.sin(x[2])*dt,
                     x[2] + u[1]*dt])

# 障碍物
obs_c = np.array([[3,4],[6,2],[5,7]]); obs_r = np.array([0.8,0.6,0.9])

def state_free(x):
    for c,r in zip(obs_c, obs_r):
        if np.linalg.norm(x[:2]-c) < r + 0.2: return False
    return 0<=x[0]<=10 and 0<=x[1]<=10

def kinodynamic_rrt(start, goal, max_iter=2000, dt=0.1, n_controls=20):
    """独轮车 Kinodynamic RRT"""
    # 离散控制集
    v_options = np.array([0.5, 1.0, 1.5])
    omega_options = np.linspace(-1.5, 1.5, n_controls//len(v_options))
    controls = [(v, w) for v in v_options for w in omega_options[:n_controls//len(v_options)]]

    nodes = [start.copy()]; parents = [-1]; controls_used = [None]
    x_goal = goal.copy()

    for _ in range(max_iter):
        if rng.random() < 0.1: x_rand = x_goal
        else: x_rand = np.array([rng.uniform(0,10), rng.uniform(0,10), rng.uniform(-np.pi,np.pi)])

        # 找最近（Cartesian 近似）
        dists = [np.linalg.norm(n[:2]-x_rand[:2]) + 0.5*abs(n[2]-x_rand[2]) for n in nodes]
        nidx = np.argmin(dists); x_near = nodes[nidx]

        # 试所有控制，选使 x_new 最接近 x_rand 的
        best_u = None; best_x = None; best_dist = np.inf
        for v, omega in controls:
            x_new = unicycle_propagate(x_near, np.array([v, omega]), dt)
            if not state_free(x_new): continue
            # 轨迹碰撞检测
            n_check = 5
            collides = False
            for alpha in np.linspace(0, 1, n_check+1)[1:]:
                x_mid = unicycle_propagate(x_near, np.array([v, omega]), alpha*dt)
                if not state_free(x_mid): collides = True; break
            if collides: continue
            d = np.linalg.norm(x_new[:2]-x_rand[:2]) + 0.5*abs(x_new[2]-x_rand[2])
            if d < best_dist: best_dist = d; best_u = (v, omega); best_x = x_new

        if best_x is None: continue
        nodes.append(best_x); parents.append(nidx); controls_used.append(best_u)

        # 目标检查
        if np.linalg.norm(best_x[:2]-x_goal[:2]) < 0.3:
            # 回溯路径
            path = [best_x]; idx = len(nodes)-1
            while parents[idx] >= 0:
                idx = parents[idx]; path.append(nodes[idx])
            return path[::-1], nodes, parents

    return None, nodes, parents

start_uni = np.array([1., 1., 0.])
goal_uni = np.array([9., 8., np.pi/4])
path_uni, nodes_uni, parents_uni = kinodynamic_rrt(start_uni, goal_uni, max_iter=1500)

fig, ax = plt.subplots(figsize=(10, 9))
for c,r in zip(obs_c, obs_r):
    ax.add_patch(plt.Circle(c, r, color='red', alpha=0.3))

# 画树
for i in range(1, len(nodes_uni)):
    p = parents_uni[i]
    if p >= 0:
        ax.plot([nodes_uni[i][0], nodes_uni[p][0]], [nodes_uni[i][1], nodes_uni[p][1]],
                'b-', alpha=0.15, linewidth=0.5)

if path_uni:
    p = np.array(path_uni)
    ax.plot(p[:,0], p[:,1], 'g-', linewidth=3, label=f'Path ({len(p)} steps)')

ax.quiver(nodes_uni[0][0], nodes_uni[0][1], np.cos(nodes_uni[0][2]), np.sin(nodes_uni[0][2]),
          color='green', scale=15, width=0.01, label='Start')
ax.quiver(goal_uni[0], goal_uni[1], np.cos(goal_uni[2]), np.sin(goal_uni[2]),
          color='red', scale=15, width=0.01, label='Goal')
ax.set_xlim([0,10]);ax.set_ylim([0,10]);ax.set_aspect('equal');ax.legend();ax.grid(True,alpha=0.3)
ax.set_title('Kinodynamic RRT — Unicycle Model')
plt.tight_layout()
plt.savefig('../outputs/16c_kinodynamic_rrt.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"Kinodynamic RRT: {'FOUND' if path_uni else 'FAILED'}, {len(nodes_uni)} nodes")

# %% [markdown]
# ## 6. 练习题
#
# ### 概念题
# 1. 标准 RRT 和 Kinodynamic RRT 的核心区别是什么？
# 2. 为什么独轮车是"非完整约束"系统？
#
# ### 编程题
# 1. 对 2R 臂实现 Kinodynamic RRT（约束关节速度和加速度）。
# 2. 在 Kinodynamic RRT 中使用到达时间作为距离度量。
