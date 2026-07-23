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
# # Notebook 16：采样运动规划 — PRM、RRT 与 RRT*
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB15 图搜索 ──→ NB16 采样规划 ──→ NB25 综合项目
# (网格,A*)       (PRM,RRT,RRT*)    (闭环系统)
# ```
#
# 对于高维 C-空间（6-DOF 以上），网格搜索的节点数随维度指数爆炸。采样规划通过**随机采样**和**增量扩展**来解决高维规划问题。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解 PRM（概率路图）的多查询框架
# - ⭐ 掌握 RRT（快速随机探索树）的算法流程
# - ⭐ 理解 RRT* 的 rewire 步骤和渐进最优性
# - ⭐ 理解 Voronoi 偏置——为什么 RRT 自然向未探索区域扩展
# - 📖 PRM vs RRT 的适用场景（多查询 vs 单查询）

# %% [markdown]
# ## 3. PRM（概率路图）⭐

# %% [markdown]
# ### 3.1 两阶段框架
#
# 1. **学习阶段（Learning Phase / Preprocessing）**：
#    - 在 $\mathcal{C}_{free}$ 中随机采样 $N$ 个节点
#    - 每个节点连接到 $k$ 个最近邻居（需做边碰撞检测）
#    - 构建图（路图 / Roadmap）
#
# 2. **查询阶段（Query Phase）**：
#    - 将起点和终点连接到路图中
#    - 在路图上运行 A*（或 Dijkstra）
#    - 路径 = 图上最短路径

# %% [markdown]
# ## 4. RRT（快速随机探索树）⭐

# %% [markdown]
# ### 4.1 算法流程
#
# ```
# RRT(start, goal, max_iter):
#   tree = {start}
#   for i = 1 to max_iter:
#       q_rand = random_sample()         # 在空间中随机采样
#       q_near = nearest(tree, q_rand)   # 找最近节点
#       q_new = extend(q_near, q_rand, Δ) # 一步扩展
#       if not collision(q_near → q_new):
#           tree.add_node(q_new, parent=q_near)
#       if distance(q_new, goal) < tol:
#           return backtrace_path(q_new)
# ```
#
# ### 4.2 Voronoi 偏置
#
# RRT 倾向于向**有更大 Voronoi 区域**的方向扩展。大 Voronoi 区域恰好是搜索树覆盖稀疏的区域——这使得 RRT 自然优先探索未探索的区域，不需要人工偏置。

# %% [markdown]
# ## 5. RRT*（渐进最优）⭐

# %% [markdown]
# ### 5.1 与 RRT 的两点区别
#
# 1. **选择最优父节点**：新节点不只是连接到最近节点，而是在搜索半径内选择使总代价最小的父节点
# 2. **Rewire（重连线）**：新节点可能改善已有节点的代价——如果通过新节点到已有节点更短，就重定向
#
# ### 5.2 渐进最优性
#
# RRT* 保证：当迭代次数 $N \to \infty$ 时，找到的路径收敛到全局最优解（在概率意义上）。

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.planning import (
    rrt_plan, rrt_star_plan, prm_plan
)
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 2D 平面 RRT

# %%
# 障碍物检测函数
obstacle_centers = np.array([[3.0, 5.0], [6.0, 3.0], [2.0, 2.0], [8.0, 7.0], [5.0, 8.0]])
obstacle_radii = np.array([0.8, 0.6, 0.5, 0.7, 0.9])
bounds = np.array([[0, 10], [0, 10]])

def is_free(q):
    for c, r in zip(obstacle_centers, obstacle_radii):
        if np.linalg.norm(q[:2] - c) < r:
            return False
    return 0 <= q[0] <= 10 and 0 <= q[1] <= 10

start, goal = np.array([1.0, 1.0]), np.array([9.0, 9.0])

path_rrt, nodes_rrt = rrt_plan(is_free, bounds, start, goal, max_iter=800, step_size=0.3, rng=rng)
path_rrts, nodes_rrts = rrt_star_plan(is_free, bounds, start, goal, max_iter=800, step_size=0.3, search_radius=0.8, rng=rng)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, nodes, path, title in [
    (axes[0], nodes_rrt, path_rrt, 'RRT'),
    (axes[1], nodes_rrts, path_rrts, 'RRT*')
]:
    # 画障碍物
    for c, r in zip(obstacle_centers, obstacle_radii):
        circle = plt.Circle(c, r, color='red', alpha=0.3)
        ax.add_patch(circle)
    # 画树
    for node in nodes:
        if node.parent is not None:
            ax.plot([node.q[0], node.parent.q[0]], [node.q[1], node.parent.q[1]],
                    'b-', alpha=0.15, linewidth=0.5)
    # 画路径
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 'g-', linewidth=3, label=f'Path ({len(path)} nodes)')

    ax.scatter(*start, c='green', s=150, marker='o', label='Start', zorder=5)
    ax.scatter(*goal, c='red', s=150, marker='*', label='Goal', zorder=5)
    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
    ax.set_title(f'{title} ({len(nodes)} nodes)')
    ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/16_rrt_vs_rrtstar.png', dpi=100, bbox_inches='tight')
plt.show()

# RRT* 路径代价对比
if path_rrt and path_rrts:
    cost_rrt = sum(np.linalg.norm(np.array(path_rrt)[i+1]-np.array(path_rrt)[i])
                   for i in range(len(path_rrt)-1))
    cost_rrts = sum(np.linalg.norm(np.array(path_rrts)[i+1]-np.array(path_rrts)[i])
                    for i in range(len(path_rrts)-1))
    print(f"RRT  path cost: {cost_rrt:.2f}")
    print(f"RRT* path cost: {cost_rrts:.2f}")
    print(f"RRT* is {cost_rrt/cost_rrts:.1%} of RRT's cost")

# %% [markdown]
# ### 6.2 2R 臂 C-空间规划模拟

# %%
# 模拟 2R 臂在笛卡尔空间有障碍物时的 C-空间规划
l1, l2 = 1.0, 0.8

def arm_collision_free(q):
    """检测 2R 臂是否碰撞笛卡尔空间中的障碍物"""
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])

    # 检查肘部和末端是否碰到障碍物
    for c, r in zip(obstacle_centers, obstacle_radii):
        # 检测肘部位置
        if np.linalg.norm(np.array([x1, y1]) - c) < r + 0.05:
            return False
        # 检测末端位置
        if np.linalg.norm(np.array([x2, y2]) - c) < r + 0.05:
            return False
        # 检测连杆与障碍物的距离（简化：取中点）
        mid1 = np.array([x1/2, y1/2])
        mid2 = np.array([(x1+x2)/2, (y1+y2)/2])
        if np.linalg.norm(mid1 - c) < r + 0.1 or np.linalg.norm(mid2 - c) < r + 0.1:
            return False
    return True

# C-space 范围
c_bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
q_start_c = np.array([np.pi/6, -np.pi/3])  # 一个无碰撞起点
q_goal_c = np.array([-np.pi/3, np.pi/4])

# RRT 在 C-空间中规划
path_c, nodes_c = rrt_plan(arm_collision_free, c_bounds, q_start_c, q_goal_c,
                            max_iter=1500, step_size=0.15, rng=rng)

# C-space 可视化
n_grid_c = 80
q1_g, q2_g = np.meshgrid(np.linspace(-np.pi, np.pi, n_grid_c),
                          np.linspace(-np.pi, np.pi, n_grid_c))
collision_map_c = np.zeros((n_grid_c, n_grid_c))
for i in range(n_grid_c):
    for j in range(n_grid_c):
        collision_map_c[i, j] = 0 if arm_collision_free(np.array([q1_g[i,j], q2_g[i,j]])) else 1

fig, ax = plt.subplots(figsize=(9, 8))
ax.contourf(np.degrees(q1_g), np.degrees(q2_g), collision_map_c,
            levels=[0, 0.5, 1], colors=['white', 'lightcoral'], alpha=0.5)
for node in nodes_c:
    if node.parent is not None:
        ax.plot([np.degrees(node.q[0]), np.degrees(node.parent.q[0])],
                [np.degrees(node.q[1]), np.degrees(node.parent.q[1])],
                'b-', alpha=0.2, linewidth=0.5)
if path_c:
    path_c_arr = np.array(path_c)
    ax.plot(np.degrees(path_c_arr[:, 0]), np.degrees(path_c_arr[:, 1]),
            'g-', linewidth=3, label=f'C-space Path')
ax.scatter(np.degrees(q_start_c[0]), np.degrees(q_start_c[1]),
           c='green', s=150, marker='o', label='Start')
ax.scatter(np.degrees(q_goal_c[0]), np.degrees(q_goal_c[1]),
           c='red', s=150, marker='*', label='Goal')
ax.set_xlabel('q₁ (°)'); ax.set_ylabel('q₂ (°)')
ax.set_title('2R Arm C-space Planning with RRT')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/16_cspace_rrt.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 练习题
#
# ### 概念题
# 1. PRM 的多查询 vs RRT 的单查询分别用于什么场景？
# 2. RRT* 的 rewire 步骤为什么能产生渐进最优？
#
# ### 编程题
# 1. 实现双向 RRT（RRT-Connect），比较其收敛速度。
# 2. 对 2R 臂做笛卡尔空间的 RRT 规划（在工作空间中采样，用 IK 求解）。
#
# > 答案见 `solutions/16_solutions.ipynb`

# %% [markdown]
# ## 8. 本节总结
#
# | 算法 | 类型 | 完备性 | 最优性 | 查询 |
# |------|------|:------:|:------:|:----:|
# | PRM | 采样+图搜索 | 概率完备 | 取决于图构建 | 多查询 |
# | RRT | 增量树 | 概率完备 | ✗ | 单查询 |
# | RRT* | 增量树+rewire | 概率完备 | 渐进最优 | 单查询 |
