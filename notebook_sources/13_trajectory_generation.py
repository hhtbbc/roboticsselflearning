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
# # Notebook 13：轨迹生成（Trajectory Generation）
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB06 IK ──→ NB13 轨迹生成 ──→ NB14 时间参数化 ──→ NB17-18 控制
#                  │
#                  ├── 路径（Path）= 空间中的曲线
#                  └── 轨迹（Trajectory）= 路径 + 时间律
# ```
#
# 轨迹生成将"要去哪里"转化为"每个时刻每关节应该在什么位置、什么速度、什么加速度"。这是控制器（NB17-18）的**参考输入**。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 区分 Path（几何）与 Trajectory（几何+时间）
# - ⭐ 掌握梯形速度曲线的构造和条件判断
# - ⭐ 掌握三次/五次多项式的系数求解
# - ⭐ 理解为什么需要 jerk 限制（S 曲线）
# - 📖 三次样条通过多路径点

# %% [markdown]
# ## 3. 路径 vs 轨迹 ⭐
#
# | | 路径 (Path) | 轨迹 (Trajectory) |
# |---|---|---|
# | 定义 | 空间中的曲线 $\mathbf{q}(s), s \in [0,1]$ | 时变函数 $\mathbf{q}(t), t \in [0,T]$ |
# | 信息量 | 只有几何形状 | 几何 + 时间 + 速度 + 加速度 |
# | 来源 | 运动规划器（NB15-16） | 轨迹生成器（本 Notebook） |
# | 数学 | 几何映射 | 时变函数及其导数 |

# %% [markdown]
# ## 4. 梯形速度曲线 ⭐

# %% [markdown]
# ### 4.1 三段式结构
#
# 给定最大速度 $v_{max}$ 和加速度 $a_{max}$，从 $q_0$ 到 $q_f$：
#
# ```
#  v ↑        ┌─────────┐
#    │       /│  巡航段  │\
#    │      / │  (v_max) │ \
#    │     /  │          │  \
#    │    /加速│          │减速\
#    └───┴────┴──────────┴─────┴→ t
#         t₁               t₂   T
# ```
#
# **关键判断**：是否能达到 $v_{max}$？
# $$\Delta q_{triangle} = a_{max} t_{acc}^2 = \frac{v_{max}^2}{a_{max}}$$
# - 若 $|q_f - q_0| > \Delta q_{triangle}$：有巡航段（梯形剖面）
# - 若 $|q_f - q_0| \leq \Delta q_{triangle}$：无巡航段（三角剖面）

# %% [markdown]
# ## 5. 多项式轨迹 ⭐

# %% [markdown]
# ### 5.1 五次多项式（最常用）
#
# 6 个边界条件：$q(0), q(T), \dot{q}(0), \dot{q}(T), \ddot{q}(0), \ddot{q}(T)$
# $$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$
#
# **为什么是五次？**
# - 三次：只能指定位置和速度（4 条件）→ 加速度在端点不连续
# - 五次：可以指定位置、速度和加速度（6 条件）→ 加速度连续
# - 七次：可以额外指定 jerk（8 条件）→ jerk 连续

# %% [markdown]
# ### 5.2 系数求解
#
# 线性系统 $\mathbf{A}\mathbf{c} = \mathbf{b}$：
# $$\begin{bmatrix} 1&0&0&0&0&0 \\ 1&T&T^2&T^3&T^4&T^5 \\ 0&1&0&0&0&0 \\ 0&1&2T&3T^2&4T^3&5T^4 \\ 0&0&2&0&0&0 \\ 0&0&2&6T&12T^2&20T^3 \end{bmatrix} \begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ a_3 \\ a_4 \\ a_5 \end{bmatrix} = \begin{bmatrix} q_0 \\ q_f \\ \dot{q}_0 \\ \dot{q}_f \\ \ddot{q}_0 \\ \ddot{q}_f \end{bmatrix}$$

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.trajectory import (
    trapezoidal_trajectory, quintic_trajectory, cubic_trajectory, via_point_trajectory
)
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 梯形速度曲线

# %%
q0, qf = 0.0, 2.0  # 从 0 到 2 rad
v_max, a_max = 1.5, 2.0

# 梯形剖面（可达到 v_max）
t_trap, q_trap, qd_trap, qdd_trap = trapezoidal_trajectory(q0, qf, v_max, a_max, dt=0.01)

# 三角剖面（不可达到 v_max）
v_low = 0.3
t_tri, q_tri, qd_tri, qdd_tri = trapezoidal_trajectory(q0, qf, v_low, a_max, dt=0.01)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for idx, (t, q, qd, qdd, label, T) in enumerate([
    (t_trap, q_trap, qd_trap, qdd_trap, f'Trapezoidal (T={t_trap[-1]:.2f}s)', t_trap[-1]),
    (t_tri, q_tri, qd_tri, qdd_tri, f'Triangular (T={t_tri[-1]:.2f}s)', t_tri[-1])
]):
    axes[0, idx].plot(t, q, 'b-', linewidth=2)
    axes[0, idx].set_ylabel('q (rad)'); axes[0, idx].set_title(label)
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].plot(t, qd, 'g-', linewidth=2, label='velocity')
    axes[1, idx].plot(t, qdd, 'r--', linewidth=2, label='acceleration')
    axes[1, idx].set_xlabel('t (s)'); axes[1, idx].set_ylabel('q̇ / q̈')
    axes[1, idx].legend(); axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/13_trapezoidal.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.2 五次多项式 vs 三次多项式

# %%
T = 3.0
t_quin, q_quin, qd_quin, qdd_quin, jerk_quin = quintic_trajectory(
    q0=0.0, qf=2.0, v0=0.0, vf=0.0, a0=0.0, af=0.0, T=T, dt=0.01
)
t_cub, q_cub, qd_cub, qdd_cub = cubic_trajectory(
    q0=0.0, qf=2.0, v0=0.0, vf=0.0, T=T, dt=0.01
)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t_quin, q_quin, 'b-', linewidth=2, label='Quintic')
axes[0].plot(t_cub, q_cub, 'r--', linewidth=2, label='Cubic')
axes[0].set_ylabel('q (rad)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_quin, qd_quin, 'b-', linewidth=2)
axes[1].plot(t_cub, qd_cub, 'r--', linewidth=2)
axes[1].set_ylabel('q̇ (rad/s)'); axes[1].grid(True, alpha=0.3)

axes[2].plot(t_quin, qdd_quin, 'b-', linewidth=2)
axes[2].plot(t_cub, qdd_cub, 'r--', linewidth=2)
axes[2].set_ylabel('q̈ (rad/s²)'); axes[2].grid(True, alpha=0.3)
axes[2].text(T/2, 1.5, 'Cubic acceleration\nDISCONTINUOUS at endpoints!',
             color='red', fontsize=10, fontweight='bold')

axes[3].plot(t_quin, jerk_quin, 'b-', linewidth=2)
axes[3].set_ylabel('jerk'); axes[3].set_xlabel('t (s)'); axes[3].grid(True, alpha=0.3)

plt.suptitle('Quintic (blue) vs Cubic (red) — 6 vs 4 boundary conditions')
plt.tight_layout()
plt.savefig('../outputs/13_quintic_vs_cubic.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.3 多关节轨迹 + 样条

# %%
# 用三次样条通过多个路径点
via_points = np.array([
    [0.0, 0.0],
    [0.5, -0.3],
    [1.2, 0.2],
    [0.8, 0.6],
    [1.5, 1.0],
])
via_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

t_spline, q_spline, qd_spline, qdd_spline = via_point_trajectory(via_points, via_times, dt=0.01)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for j, color in enumerate(['blue', 'red']):
    axes[0].plot(t_spline, q_spline[:, j], color=color, linewidth=2, label=f'Joint {j+1}')
    axes[0].scatter(via_times, via_points[:, j], color=color, s=50, zorder=5)
    axes[1].plot(t_spline, qd_spline[:, j], color=color, linewidth=1.5)
    axes[2].plot(t_spline, qdd_spline[:, j], color=color, linewidth=1.5)

axes[0].set_ylabel('q (rad)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_title('Cubic Spline Via-Point Trajectory (C² continuous)')
axes[1].set_ylabel('q̇ (rad/s)'); axes[1].grid(True, alpha=0.3)
axes[2].set_ylabel('q̈ (rad/s²)'); axes[2].set_xlabel('t (s)'); axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/13_spline_trajectory.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 常见错误
#
# 1. **路径 ≠ 轨迹**：面试中常被问到。路径是纯几何，轨迹包含时间信息。
# 2. **梯形剖面的判断**：忘记检查是否能达到 $v_{max}$ 会导致三角剖面的错误判断。
# 3. **三次 vs 五次**：三次在端点加速度不连续 → 产生无限 jerk → 可能激发机械共振。

# %% [markdown]
# ## 8. 练习题
#
# ### 手算题
# 1. 从静止到静止，$q_0=0, q_f=\pi/2, T=1$s，求五次多项式系数。
# 2. 判断 $v_{max}=2, a_{max}=3, dq=1.0$ 时是梯形还是三角剖面。
#
# ### 编程题
# 1. 实现 S 曲线（jerk 限制的梯形速度剖面）。
# 2. 比较梯形、五次多项式和 S 曲线在相同约束下的 jerk 大小。
#
# > 答案见 `solutions/13_solutions.ipynb`

# %% [markdown]
# ## 9. 本节总结
#
# | 方法 | 可指定 | 连续性 | 适用场景 |
# |------|--------|:------:|----------|
# | 梯形 | $v_{max}, a_{max}$ | q̈ 不连续 | 简单 PTP 运动 |
# | 三次 | $q_0,q_f,\dot{q}_0,\dot{q}_f$ | q̈ 不连续 | 端点速度已知 |
# | 五次 | 额外 $\ddot{q}_0,\ddot{q}_f$ | q̈ 连续 | 高精度跟踪 |
# | S 曲线 | 额外 jerk 限制 | jerk 连续 | 高速/高精度 |
# | 样条 | 多路径点 | C² | 复杂路径 |
