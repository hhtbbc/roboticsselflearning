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
# # Notebook 12c：接触与碰撞动力学

# ## 1. 定位
# NB12 介绍了约束动力学的拉格朗日乘子形式。本节深入接触的物理建模——单边约束、摩擦锥、碰撞冲量和抓取基本概念。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ 区分双边约束与单边约束
# - ⭐ 理解互补条件 $\phi \geq 0, \lambda_n \geq 0, \phi\lambda_n = 0$
# - ⭐ 库仑摩擦锥及其线性化
# - ⭐ 碰撞冲量与恢复系数
# - 📖 抓取矩阵与 force closure

# %% [markdown]
# ## 3. 单边接触 ⭐

# %% [markdown]
# ### 3.1 法向约束
# 接触条件（Signorini 条件）：
# $$\phi(\mathbf{q}) \geq 0, \quad \lambda_n \geq 0, \quad \phi(\mathbf{q})\lambda_n = 0$$
# - $\phi(\mathbf{q})$：穿透深度（>0 = 分离）
# - $\lambda_n$：法向接触力（只能是推力，不能是拉力）
# - $\phi\lambda_n = 0$：要么分离（$\phi>0, \lambda_n=0$），要么接触（$\phi=0, \lambda_n>0$）

# %% [markdown]
# ### 3.2 库仑摩擦锥
# $$\|\mathbf{f}_t\| \leq \mu f_n$$
# 切向摩擦力受法向力限制。在优化中线性化为摩擦棱锥：
# $$\mathbf{f}_t = \sum_i \beta_i \mathbf{d}_i, \quad \beta_i \geq 0$$
# 其中 $\mathbf{d}_i$ 是单位圆上的离散方向。

# %% [markdown]
# ## 4. 碰撞动力学

# %% [markdown]
# ### 4.1 碰撞冲量
# 碰撞前后速度跳跃：$\mathbf{M}(\dot{\mathbf{q}}^+ - \dot{\mathbf{q}}^-) = \mathbf{J}_c^T \boldsymbol{\Lambda}$
#
# Poisson 恢复系数 $e \in [0,1]$：$\dot{\phi}^+ = -e \dot{\phi}^-$

# %% [markdown]
# ## 5. Python — 2D 接触示例

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 摩擦锥可视化

# %%
theta = np.linspace(0, 2*np.pi, 200)
mu = 0.5
cone_x = mu * np.cos(theta); cone_y = mu * np.sin(theta)

# 线性化：8 个方向
n_dirs = 8
dirs = np.array([[np.cos(2*np.pi*i/n_dirs), np.sin(2*np.pi*i/n_dirs)] for i in range(n_dirs)])

fig, ax = plt.subplots(figsize=(7, 7))
ax.fill(cone_x, cone_y, color='lightblue', alpha=0.5, label=f'Friction Cone (μ={mu})')
for i in range(n_dirs):
    d = mu * dirs[i]
    ax.plot([0, d[0]], [0, d[1]], 'r-', linewidth=2)
    if i == 0: ax.plot([0, d[0]], [0, d[1]], 'r-', linewidth=2, label=f'{n_dirs}-gon approximation')
ax.quiver(0, 0, 1, 0, scale=3, color='blue', width=0.01, label='f_n (normal)')
ax.set_xlim([-1, 1.5]); ax.set_ylim([-1, 1])
ax.set_xlabel('f_x'); ax.set_ylabel('f_y')
ax.set_title('Coulomb Friction Cone & Linearization'); ax.set_aspect('equal')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/12c_friction_cone.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.2 互补条件演示 — 球落下碰地面

# %%
# 1D 球从高度 h 落下，碰地面弹跳
m, g, e = 1.0, 9.81, 0.7
y0, vy0 = 2.0, 0.0  # 初始高度 2m
dt_c = 0.001; T_c = 3.0

y, vy = y0, vy0
y_hist, vy_hist, lambda_hist = [], [], []
t_hist = []

for step in range(int(T_c/dt_c)):
    t = step * dt_c
    # 重力
    vy += -g * dt_c; y += vy * dt_c

    lambda_n = 0.0
    if y < 0:  # 穿透地面
        # 冲量修正
        y = 0.0
        if vy < 0:
            lambda_n = m * (1+e) * abs(vy) / dt_c  # 等效法向力
            vy = -e * vy  # 反弹

    y_hist.append(y); vy_hist.append(vy); lambda_hist.append(lambda_n); t_hist.append(t)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t_hist, y_hist, 'b-', linewidth=1.5); axes[0].axhline(y=0, c='k', ls='--', alpha=0.3)
axes[0].set_ylabel('y (m)'); axes[0].set_title('1D Ball Bouncing — Unilateral Contact')
axes[1].plot(t_hist, vy_hist, 'g-', linewidth=1.5); axes[1].set_ylabel('v_y (m/s)')
axes[2].plot(t_hist, lambda_hist, 'r-', linewidth=1); axes[2].set_ylabel('λ_n (N)'); axes[2].set_xlabel('t (s)')
for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/12c_bouncing_ball.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5.3 2D 抓取力分析

# %%
# 两指抓取物体，分析内力与外力
# 接触点 c1, c2，法向 n1, n2
c1, n1 = np.array([-0.3, 0.0]), np.array([1.0, 0.0])    # 左指
c2, n2 = np.array([0.3, 0.0]), np.array([-1.0, 0.0])   # 右指
mu_grasp = 0.6

# 抓取矩阵 G: τ = G^T f (接触力 → wrench)
G = np.zeros((3, 4))  # [f_x, f_y, τ_z]  # 2 contacts × 2 forces (normal + tangential)
G[:2, 0] = n1; G[:2, 1] = np.array([0.0, 1.0])  # n1, t1
G[:2, 2] = n2; G[:2, 3] = np.array([0.0, 1.0])  # n2, t2
# 2D cross product (scalar z-component): a_x*b_y - a_y*b_x
def cross2d(a, b): return a[0]*b[1] - a[1]*b[0]
G[2, 0] = cross2d(c1, n1); G[2, 1] = cross2d(c1, np.array([0.0, 1.0]))
G[2, 2] = cross2d(c2, n2); G[2, 3] = cross2d(c2, np.array([0.0, 1.0]))

# 摩擦锥约束: f_n >= 0, |f_t| <= μ f_n
# 外力 wrench: w_ext = [f_x, 0, 0, 0, 0, τ_z]（重力+外力）
# Force closure: 存在 λ ≥ 0 满足 Gλ = w_ext 且满足摩擦约束

# 简化：检查能否抵抗 ±f_x（水平力）
# 需要 f1_n * μ + f2_n * μ >= |f_x|
max_fx_per_finger = mu_grasp * 10  # 假设每指最大 10N 法向力
print(f"每指最大摩擦力: {max_fx_per_finger:.1f}N")
print(f"两指最大抵抗水平力: {2*max_fx_per_finger:.1f}N")
print(f"Friction cone angle = arctan({mu_grasp}) = {np.degrees(np.arctan(mu_grasp)):.1f}°")

# 可视化抓取
fig, ax = plt.subplots(figsize=(6, 6))
rect = plt.Rectangle((-0.3, -0.15), 0.6, 0.3, fill=False, linewidth=3, color='blue')
ax.add_patch(rect)
for ci, ni, color in [(c1, n1, 'red'), (c2, n2, 'green')]:
    for angle in np.linspace(-mu_grasp, mu_grasp, 20):
        d = np.array([np.cos(np.pi/2 + angle), np.sin(np.pi/2 + angle)]) * 0.15
        ax.arrow(ci[0], ci[1], d[0]*ni[0]*5, d[1]*5, head_width=0.02, color=color, alpha=0.3)
    ax.scatter(*ci, c=color, s=100)
ax.set_xlim([-0.8, 0.8]); ax.set_ylim([-0.5, 0.5])
ax.set_aspect('equal'); ax.set_title('2-Finger Grasp with Friction Cones')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/12c_grasp_friction_cones.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. 练习题
# 1. 互补条件 $\phi\lambda_n=0$ 的物理含义？三种可能状态？
# 2. 为什么摩擦锥用多面体近似？
# 3. Force closure 和 form closure 的区别？
