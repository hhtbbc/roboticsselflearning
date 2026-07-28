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
# # Notebook 14：时间参数化 — 教学版前向-后向路径速度参数化近似
#
# ## 1. 定位
# 运动规划器输出几何路径 $\mathbf{q}(s)$。时间参数化回答"多快走"——找到 $s(t)$ 使路径在时间上满足所有速度/加速度约束。
# **本教学版使用前向-后向近似，不保证严格时间最优或全部约束满足。生产级 TOPP 请使用 TOPP-RA 或 toppra 库。**

# %% [markdown]
# ## 2. 学习目标
# - ⭐ 路径参数化 $s \in [0,1]$: $\mathbf{q}(t) = \mathbf{f}(s(t))$
# - ⭐ 链式分解: $\ddot{\mathbf{q}} = \mathbf{f}'(s)\ddot{s} + \mathbf{f}''(s)\dot{s}^2$
# - ⭐ 在 $(s, \dot{s})$ 相平面中理解 MVC 和可行域
# - ⭐ 前向最大加速 + 后向最大减速的核心思路
# - 📖 TOPP-RA 和力矩约束（本章不覆盖严格解）

# %% [markdown]
# ## 3. 链式分解 ⭐

# %% [markdown]
# 给定路径 $\mathbf{q} = \mathbf{f}(s), s \in [0, 1]$:
# $$\dot{\mathbf{q}} = \mathbf{f}'(s)\dot{s}$$
# $$\ddot{\mathbf{q}} = \mathbf{f}'(s)\ddot{s} + \mathbf{f}''(s)\dot{s}^2$$
#
# 关节约束转化为对 $(s, \dot{s})$ 的约束:
# $$-\dot{q}_{max,i} \leq f'_i(s)\dot{s} \leq \dot{q}_{max,i}$$
# $$-\ddot{q}_{max,i} \leq f'_i(s)\ddot{s} + f''_i(s)\dot{s}^2 \leq \ddot{q}_{max,i}$$

# %% [markdown]
# ## 4. 前向-后向参数化近似 ⭐

# %% [markdown]
# ### 4.1 MVC (Maximum Velocity Curve)
# 对每个 $s$，从关节速度约束求 $\dot{s}$ 上限:
# $$\dot{s}_{max}(s) = \min_i \frac{\dot{q}_{max,i}}{|f'_i(s)|}$$
# 若 $f'_i(s) = 0$，该关节对 $\dot{s}$ 无约束。

# %% [markdown]
# ### 4.2 加速度约束下的可行加速/减速
# 从 $\ddot{q}_{min} \leq f'\ddot{s} + f''\dot{s}^2 \leq \ddot{q}_{max}$:
# $$\alpha_i(s, \dot{s}) = \begin{cases}
# (\ddot{q}_{min,i} - f''_i\dot{s}^2)/f'_i & f'_i > 0 \\
# (\ddot{q}_{max,i} - f''_i\dot{s}^2)/f'_i & f'_i < 0
# \end{cases}$$
# $$\beta_i(s, \dot{s}) = \begin{cases}
# (\ddot{q}_{max,i} - f''_i\dot{s}^2)/f'_i & f'_i > 0 \\
# (\ddot{q}_{min,i} - f''_i\dot{s}^2)/f'_i & f'_i < 0
# \end{cases}$$
# $$\alpha = \max_i \alpha_i, \quad \beta = \min_i \beta_i$$

# %% [markdown]
# ### 4.3 前向 + 后向积分（教学版近似）
# 1. 前向传播 (最大加速): 从 $(0, \dot{s}_0)$ 开始，用 $\beta(s, \dot{s})$ 加速
# 2. 后向传播 (最大减速): 从 $(1, \dot{s}_f)$ 开始，用 $\alpha(s, \dot{s})$ 减速
# 3. 取 min(前向, 后向, MVC) 作为近似曲线
# 4. 使用 $\dot{s}_{k+1}^2 = \dot{s}_k^2 + 2\ddot{s}\Delta s$ (稳定递推)
# 5. **本方法不等价于严格 TOPP-RA，不能保证全部约束满足或全局时间最优。**

# %% [markdown]
# ## 5. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from scipy.interpolate import CubicSpline
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 一般路径 TOPP — 2R 臂样条路径

# %%
# 路径: 三次样条连接关节空间中的 via-points
via_pts = np.array([
    [0.0, 0.0],
    [0.3, 0.2],
    [0.7, 0.4],
    [1.0, 0.6],
    [1.5, 1.0],
])
s_grid = np.linspace(0, 1, len(via_pts))
cs1 = CubicSpline(s_grid, via_pts[:, 0], bc_type='natural')
cs2 = CubicSpline(s_grid, via_pts[:, 1], bc_type='natural')

n_s = 300
s_vals = np.linspace(0, 1, n_s)
ds = s_vals[1] - s_vals[0]

# f(s), f'(s), f''(s)
f_vals = np.column_stack([cs1(s_vals), cs2(s_vals)])
fp_vals = np.column_stack([cs1(s_vals, 1), cs2(s_vals, 1)])
fpp_vals = np.column_stack([cs1(s_vals, 2), cs2(s_vals, 2)])

# 关节约束
q_dot_max = np.array([4.0, 5.0])
q_ddot_max = np.array([15.0, 18.0])

# === 调用公共 TOPP 实现 ===
from src.robotics_learning.trajectory import (
    topp_forward_backward_parameterization,
    velocity_mvc_from_joint_limits,
)

(s_dot_mvc, s_dot_fwd, s_dot_bwd,
 s_dot_approx, t_approx) = topp_forward_backward_parameterization(
    f_vals, fp_vals, fpp_vals, s_vals,
    q_dot_max, q_ddot_max,
    s_dot_start=0.0, s_dot_end=0.0,
)

# 对比: 可行常速基线 — 检查 s_dot_const = min_i( q_dot_max,i / max_s|f'_i(s)| )
s_dot_const_candidate = np.inf
for d in range(2):
    max_fp = np.max(np.abs(fp_vals[:, d]))
    if max_fp > 1e-10:
        s_dot_const_candidate = min(s_dot_const_candidate, q_dot_max[d] / max_fp)
# 还要检查 f'' 约束: |f''_i| * s_dot_const² ≤ q_ddot_max,i
for d in range(2):
    max_fpp = np.max(np.abs(fpp_vals[:, d]))
    if max_fpp > 1e-10:
        s_dot_const_candidate = min(s_dot_const_candidate,
                                   np.sqrt(q_ddot_max[d] / max_fpp))
s_dot_const = s_dot_const_candidate if np.isfinite(s_dot_const_candidate) else 0.0
t_const = np.zeros(n_s)
for i in range(1, n_s):
    t_const[i] = t_const[i-1] + ds / max(s_dot_const, 1e-6)
s_dot_uniform = np.full(n_s, s_dot_const)
t_uniform = t_const.copy()

# %% [markdown]
# ### 5.2 可视化

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (s, ṡ) 相平面
axes[0,0].fill_between(s_vals, 0, s_dot_mvc, color='lightgreen', alpha=0.3, label='Feasible')
axes[0,0].plot(s_vals, s_dot_mvc, 'k-', linewidth=2, label='MVC')
axes[0,0].plot(s_vals, s_dot_fwd, 'b--', linewidth=1.5, label='Forward pass')
axes[0,0].plot(s_vals, s_dot_bwd, 'r--', linewidth=1.5, label='Backward pass')
axes[0,0].plot(s_vals, s_dot_approx, 'purple', linewidth=2.5, label='Approx ṡ(s) (teaching demo)')
axes[0,0].plot(s_vals, s_dot_uniform, 'orange', linewidth=1, alpha=0.7, label='Feasible uniform ṡ')
axes[0,0].set_xlabel('s'); axes[0,0].set_ylabel('ṡ')
axes[0,0].set_title('(s, ṡ) Phase Plane — Teaching Approximation'); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

# q₁(t) 对比
q_path_vals = np.column_stack([cs1(s_vals), cs2(s_vals)])
axes[0,1].plot(t_approx, q_path_vals[:,0], 'purple', linewidth=2, label=f'Approx (T={t_approx[-1]:.3f}s)')
axes[0,1].plot(t_uniform, q_path_vals[:,0], 'orange', linewidth=2, label=f'Feasible uniform (T={t_uniform[-1]:.3f}s)')
axes[0,1].set_xlabel('t (s)'); axes[0,1].set_ylabel('q₁ (rad)')
axes[0,1].set_title(f'Joint 1: Approx saves {(1-t_approx[-1]/t_uniform[-1])*100:.1f}% time vs uniform'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# 速度验证
q_dot_actual = fp_vals * s_dot_approx[:, None]  # q̇ = f'(s)ṡ
axes[1,0].plot(t_approx, q_dot_actual[:,0], 'purple', linewidth=1.5, label='q̇₁ (approx)')
axes[1,0].plot(t_approx, q_dot_actual[:,1], 'g-', linewidth=1.5, label='q̇₂ (approx)')
axes[1,0].axhline(y=q_dot_max[0], c='k', ls='--', alpha=0.3)
axes[1,0].axhline(y=-q_dot_max[0], c='k', ls='--', alpha=0.3)
axes[1,0].set_xlabel('t (s)'); axes[1,0].set_ylabel('q̇ (rad/s)')
axes[1,0].set_title('Joint Velocities'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# 路径 q₁ vs q₂
axes[1,1].plot(q_path_vals[:,0], q_path_vals[:,1], 'b-', linewidth=2, label='Geometric path')
axes[1,1].scatter(via_pts[:,0], via_pts[:,1], c='red', s=50, zorder=5, label='Via points')
axes[1,1].set_xlabel('q₁ (rad)'); axes[1,1].set_ylabel('q₂ (rad)')
axes[1,1].set_title('Path in Joint Space'); axes[1,1].set_aspect('equal'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/14_topp_general_path.png', dpi=100, bbox_inches='tight')
plt.show()

# 约束验证: 使用与传播一致的区间加速度公式 s̈_k = (ṡ_{k+1}² - ṡ_k²) / (2 Δs)
s_ddot_interval = np.zeros(n_s)
for k in range(n_s - 1):
    s_ddot_interval[k] = (s_dot_approx[k+1]**2 - s_dot_approx[k]**2) / (2 * ds)
s_ddot_interval[-1] = s_ddot_interval[-2]  # 边界用相邻值

q_dot_actual = fp_vals * s_dot_approx[:, None]
q_ddot_actual = fp_vals * s_ddot_interval[:, None] + fpp_vals * s_dot_approx[:, None]**2

tolerance = 1e-6
# 速度约束断言
v_ok = np.all(np.abs(q_dot_actual) <= q_dot_max + tolerance * 1.01)
assert v_ok, f"速度约束违反: max|q̇₁|={np.max(np.abs(q_dot_actual[:,0])):.4f}, max|q̇₂|={np.max(np.abs(q_dot_actual[:,1])):.4f}"
# 加速度约束断言
a_ok = np.all(np.abs(q_ddot_actual) <= q_ddot_max + tolerance)
assert a_ok, f"加速度约束违反: max|q̈₁|={np.max(np.abs(q_ddot_actual[:,0])):.4f}, max|q̈₂|={np.max(np.abs(q_ddot_actual[:,1])):.4f}"

print(f"✅ 近似参数化总时间: {t_approx[-1]:.3f}s")
print(f"   可行常速基线: {t_uniform[-1]:.3f}s (ṡ={s_dot_const:.3f})")
print(f"   时间节省: {(1-t_approx[-1]/t_uniform[-1])*100:.1f}%")
print(f"✅ 速度约束 (|q̇| ≤ {q_dot_max}): 全部满足")
print(f"✅ 加速度约束 (|q̈| ≤ {q_ddot_max}): 全部满足")
print("⚠ 本教学版使用前向-后向近似，不保证严格时间最优。")
print("  生产级 TOPP 请使用 TOPP-RA 或 toppra 库。")

# %% [markdown]
# ## 6. 练习题
# 1. MVC 曲线由什么约束决定？TOPP 如何保证不超出 MVC？
# 2. 为什么前向和后向积分都需要？只用前向会有什么问题？
# 3. 加入力矩约束后, $\alpha(s,\dot{s})$ 和 $\beta(s,\dot{s})$ 的新形式？

# %% [markdown]
# ## 7. 本节总结
# | 概念 | 公式 | 含义 |
# |------|------|------|
# | 链式分解 | $\ddot{\mathbf{q}} = \mathbf{f}'\ddot{s} + \mathbf{f}''\dot{s}^2$ | $\ddot{s}$ + $\dot{s}^2$ 分别贡献 |
# | MVC | $\dot{s}_{max}(s) = \min_i \dot{q}_{max,i}/|f'_i(s)|$ | 速度硬上限 |
# | 加速度约束 | $\alpha(s,\dot{s}) \leq \ddot{s} \leq \beta(s,\dot{s})$ | 从关节加速度反推 |
# | 稳定递推 | $\dot{s}_{k+1}^2 = \dot{s}_k^2 + 2\ddot{s}\Delta s$ | 避免 $\dot{s} \approx 0$ 不稳定 |
# | ⚠ 教学近似 | 前向-后向 min 取线 | 不保证严格 TOPP-RA 最优或全部约束 |
