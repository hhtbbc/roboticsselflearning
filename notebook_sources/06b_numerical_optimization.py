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
# # Notebook 06b：机器人学中的数值优化基础
#
# ## 1. 定位
#
# 优化贯穿整个机器人学：数值 IK 是最小二乘、参数辨识是线性回归、轨迹优化是非线性规划、MPC 是约束 QP。本节建立统一的优化工具箱。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 梯度下降 vs 牛顿法 vs Gauss-Newton vs Levenberg-Marquardt
# - ⭐ 理解 KKT 条件与约束优化
# - ⭐ 掌握二次规划 (QP) 的基本形式
# - ⭐ 将数值 IK、参数辨识、Bundle Adjustment 统一为最小二乘问题
# - 📖 自动微分 vs 有限差分

# %% [markdown]
# ## 3. 无约束优化 ⭐

# %% [markdown]
# ### 3.1 梯度下降
# $$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)$$
# 一阶方法，简单但收敛慢。$\alpha$ 是步长（学习率）。

# %% [markdown]
# ### 3.2 牛顿法
# $$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}^{-1} \nabla f(\mathbf{x}_k)$$
# 二阶方法，收敛快但需 Hessian $\mathbf{H} = \nabla^2 f$。

# %% [markdown]
# ### 3.3 非线性最小二乘 — Gauss-Newton
#
# 对于 $\min_\mathbf{x} \frac{1}{2}\|\mathbf{r}(\mathbf{x})\|^2$（$\mathbf{r}$ 是残差向量）：
# $$\mathbf{J}^T\mathbf{J} \Delta\mathbf{x} = -\mathbf{J}^T \mathbf{r}$$
# 其中 $\mathbf{J} = \partial\mathbf{r}/\partial\mathbf{x}$。Gauss-Newton 用 $\mathbf{J}^T\mathbf{J}$ 近似 Hessian。

# %% [markdown]
# ### 3.4 Levenberg-Marquardt (LM)
#
# 在 $\mathbf{J}^T\mathbf{J}$ 病态时加阻尼：
# $$(\mathbf{J}^T\mathbf{J} + \lambda\mathbf{I}) \Delta\mathbf{x} = -\mathbf{J}^T \mathbf{r}$$
# $\lambda \to 0$ → Gauss-Newton；$\lambda \to \infty$ → 梯度下降。
# 这就是数值 IK 的 DLS 方法的数学基础！

# %% [markdown]
# ## 4. 约束优化与 KKT ⭐

# %% [markdown]
# 对于 $\min_\mathbf{x} f(\mathbf{x})$ s.t. $\mathbf{h}(\mathbf{x}) = \mathbf{0}, \mathbf{g}(\mathbf{x}) \leq \mathbf{0}$：
#
# KKT 条件：存在拉格朗日乘子 $\boldsymbol{\lambda}, \boldsymbol{\mu} \geq \mathbf{0}$ 使
# $$\nabla f + \nabla\mathbf{h}^T\boldsymbol{\lambda} + \nabla\mathbf{g}^T\boldsymbol{\mu} = \mathbf{0}$$
# $$\mathbf{h} = \mathbf{0}, \quad \boldsymbol{\mu} \circ \mathbf{g} = \mathbf{0} \quad (\mu_i g_i = 0)$$
#
# 互补条件 $\mu_i g_i = 0$ 是接触力学中 $\lambda_n \phi(q) = 0$ 的数学源头！

# %% [markdown]
# ## 5. 二次规划 (QP)

# %% [markdown]
# QP 是机器人学中最重要的约束优化形式：
# $$\min_\mathbf{x} \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x}$$
# $$\text{s.t. } \mathbf{A}\mathbf{x} = \mathbf{b}, \quad \mathbf{G}\mathbf{x} \leq \mathbf{h}$$
#
# 应用场景：
# - 全身控制（WBC）→ QP 形式的冗余分解
# - MPC → 每一步解一个 QP
# - 接触力优化 → 摩擦锥约束下最小化力

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 梯度下降 vs 牛顿法 vs LM

# %%
# Rosenbrock 函数: f(x,y) = (1-x)² + 100(y-x²)²
def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosenbrock_grad(x):
    return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def rosenbrock_hess(x):
    return np.array([[2 - 400*x[1] + 1200*x[0]**2, -400*x[0]], [-400*x[0], 200]])

x0 = np.array([-1.0, 1.0])
methods = {'GD': lambda x,g,H: x - 0.001*g,
           'Newton': lambda x,g,H: x - np.linalg.solve(H, g),
           'LM': lambda x,g,H: x - np.linalg.solve(H + 10*np.eye(2), g)}

histories = {}
for name, update in methods.items():
    x = x0.copy(); hist = [x.copy()]
    for _ in range(100):
        g = rosenbrock_grad(x); H = rosenbrock_hess(x)
        x = update(x, g, H); hist.append(x.copy())
        if np.linalg.norm(g) < 1e-6: break
    histories[name] = np.array(hist)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, hist) in zip(axes, histories.items()):
    ax.plot(hist[:,0], hist[:,1], 'b-o', markersize=3, linewidth=1, alpha=0.7)
    ax.scatter(1, 1, c='red', s=100, marker='*', zorder=5)
    ax.scatter(*x0, c='green', s=80, zorder=5)
    ax.set_title(f'{name} ({len(hist)} iters)'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
plt.suptitle('Optimization on Rosenbrock Function', fontsize=14)
plt.tight_layout()
plt.savefig('../outputs/06b_optimization_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.2 非线性最小二乘 — 数值 IK 的本质

# %%
# 数值 IK = 求解 min_q ||FK(q) - T_des|| 的非线性最小二乘
from src.robotics_learning.kinematics import forward_kinematics, compute_geometric_jacobian
from src.robotics_learning.transforms import so3_log

dh = np.array([[1.0, 0, 0], [0.8, 0, 0]])

# 目标位姿
T_des, _ = forward_kinematics(np.column_stack([dh, [np.pi/3, np.pi/6]]))
q = np.array([0.2, 0.2])  # 远离目标的初值

err_history = []
for it in range(50):
    T_curr, _ = forward_kinematics(np.column_stack([dh, q]))
    p_err = T_des[:3,3] - T_curr[:3,3]
    omega_err = so3_log(T_des[:3,:3] @ T_curr[:3,:3].T)
    r = np.concatenate([p_err, omega_err])

    J = compute_geometric_jacobian(dh, q)
    # Gauss-Newton: (J^T J) Δq = -J^T r
    # Levenberg-Marquardt: (J^T J + λI) Δq = -J^T r
    lam = 0.1 * (1.0 / (1 + 0.1*it))  # 递减阻尼
    JTJ = J.T @ J
    delta_q = np.linalg.solve(JTJ + lam * np.eye(2), J.T @ r)
    q = q + 0.5 * delta_q  # 步长 0.5 防超调
    err_history.append(np.linalg.norm(r))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(err_history, 'b-o', linewidth=2, markersize=4)
ax.set_xlabel('Iteration'); ax.set_ylabel('||Residual||')
ax.set_title('Numerical IK as Nonlinear Least Squares (LM with decaying λ)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/06b_ik_as_optimization.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"最终误差: {err_history[-1]:.2e}, 关节角: {np.round(np.degrees(q),1)}°")

# %% [markdown]
# ### 6.3 QP 求解（简化全身控制）

# %%
# 2R 臂：末端跟踪 + 最小化力矩（二次代价）
# min ½||ẋ_d - J q̇||² + ε||q̇||²  →  (J^T J + εI) q̇ = J^T ẋ_d
J_test = compute_geometric_jacobian(np.array([[1.,0,0],[0.8,0,0]]), np.array([0.5, 0.3]))
x_dot_des = np.array([0.1, 0.05, 0, 0, 0, 0])

# QP 形式: min ½ q̇^T Q q̇ + c^T q̇
Q = J_test.T @ J_test + 0.01 * np.eye(2)
c = -J_test.T @ x_dot_des
q_dot_opt = np.linalg.solve(Q, -c)
print(f"QP 解 q̇ = {np.round(q_dot_opt, 4)}")
print(f"产生末端速度: {np.round(J_test @ q_dot_opt, 4)}")
print(f"与期望:        {np.round(x_dot_des, 4)}")

# %% [markdown]
# ## 7. 练习题
#
# ### 概念题
# 1. Gauss-Newton 用的 Hessian 近似 $\mathbf{J}^T\mathbf{J}$ 在什么时候是好的近似？
# 2. KKT 互补条件 $\mu_i g_i = 0$ 的物理含义是什么？
#
# ### 编程题
# 1. 实现带不等式约束的 QP 求解器（Active Set 或内点法简化版）。
# 2. 用 LM 求解器替代数值 IK 的简单 DLS。
