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
# # Notebook 09-10：动力学建模与拉格朗日法
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB05 FK ──→ ┌─────────────────────────────────────┐
# NB07 J  ──→│  NB09-10 动力学                      │
#             │  M(q)q̈ + C(q,q̇)q̇ + g(q) = τ      │
#             └─────────────────────────────────────┘
#                              │
#     ┌────────────────────────┼────────────────────────┐
#     ↓                        ↓                        ↓
# NB11 牛顿-欧拉        NB17-18 控制           NB14 时间参数化
# (高效递推)        (需要 M/C/g 做前馈)     (力矩约束)
# ```
#
# 动力学将运动学扩展到**力与运动的关系**。运动学回答"它在哪？"，动力学回答"需要多少力才能这样动？"

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解动力学方程的通用结构 $\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
# - ⭐ 区分正动力学（FD: $\boldsymbol{\tau} \to \ddot{\mathbf{q}}$）和逆动力学（ID: $\mathbf{q},\dot{\mathbf{q}},\ddot{\mathbf{q}} \to \boldsymbol{\tau}$）
# - ⭐ 从拉格朗日量推导 2R 臂的 $\mathbf{M}, \mathbf{C}, \mathbf{g}$ 显式表达式
# - ⭐ 用 SymPy 符号推导并转为数值函数验证
# - ⭐ 验证 $\dot{\mathbf{M}} - 2\mathbf{C}$ 的反对称性（控制稳定性分析的核心）
# - 📖 拉格朗日法 vs 牛顿-欧拉法的适用场景
# - 📚 带摩擦和弹性的拉格朗日模型

# %% [markdown]
# ## 3. 前置知识
#
# - NB05：正运动学、DH 参数
# - NB07：雅可比（用于计算连杆速度）
# - NB01：矩阵正定性、对称性

# %% [markdown]
# ## 4. 动力学方程的标准形式 ⭐

# %% [markdown]
# ### 4.1 方程结构
#
# $$\boxed{\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}}$$
#
# | 项 | 符号 | 维度 | 物理含义 | 性质 |
# |----|------|:----:|----------|------|
# | 惯性项 | $\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}$ | $n\times 1$ | 加速度产生惯性力 | $\mathbf{M} \succ 0$ 对称正定 |
# | 科氏力/离心力 | $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}$ | $n\times 1$ | 速度乘积项 | $\dot{\mathbf{M}} - 2\mathbf{C}$ 反对称 |
# | 重力项 | $\mathbf{g}(\mathbf{q})$ | $n\times 1$ | 重力产生的力矩 | $\mathbf{g}(\mathbf{q}) = \partial\mathcal{P}/\partial\mathbf{q}$ |
# | 外力项 | $\mathbf{J}^T(\mathbf{q})\mathbf{F}_{ext}$ | $n\times 1$ | 末端力映射 | 来自虚功原理 (NB07) |

# %% [markdown]
# ### 4.2 正动力学 vs 逆动力学
#
# - **逆动力学（Inverse Dynamics, ID）**：已知 $\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}$，求 $\boldsymbol{\tau}$。
#   - 用途：**控制前馈**——给定期望轨迹，计算需要的力矩
# - **正动力学（Forward Dynamics, FD）**：已知 $\mathbf{q}, \dot{\mathbf{q}}, \boldsymbol{\tau}$，求 $\ddot{\mathbf{q}}$（然后积分得到 $\dot{\mathbf{q}}, \mathbf{q}$）。
#   - 用途：**仿真**——施加力矩后，模拟机器人运动
#   - 实现：$\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\mathbf{q})(\boldsymbol{\tau} - \mathbf{C}\dot{\mathbf{q}} - \mathbf{g})$

# %% [markdown]
# ### 4.3 $\dot{\mathbf{M}} - 2\mathbf{C}$ 的反对称性 ⭐
#
# **定理**：矩阵 $\mathbf{N}(\mathbf{q},\dot{\mathbf{q}}) = \dot{\mathbf{M}}(\mathbf{q}) - 2\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})$ 是**反对称矩阵**，即 $\mathbf{N}^T = -\mathbf{N}$。
#
# 这等价于：
# $$\dot{\mathbf{q}}^T (\dot{\mathbf{M}} - 2\mathbf{C}) \dot{\mathbf{q}} = 0, \quad \forall \dot{\mathbf{q}}$$
#
# 物理含义：科氏力和离心力**不做功**——它们改变速度的方向但不改变速率。
#
# 工程意义：这是**无源性**和**李雅普诺夫稳定性分析**的基础（NB17-18）。

# %% [markdown]
# ## 5. 拉格朗日法推导步骤 ⭐

# %% [markdown]
# ### 5.1 拉格朗日量
#
# $$\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = \mathcal{K}(\mathbf{q}, \dot{\mathbf{q}}) - \mathcal{P}(\mathbf{q})$$
#
# 其中 $\mathcal{K}$ 是系统总动能，$\mathcal{P}$ 是总势能。

# %% [markdown]
# ### 5.2 拉格朗日方程
#
# $$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i} = \tau_i, \quad i = 1,\dots,n$$
#
# 展开得到三大项的构造方法：
# 1. $\mathbf{M}(\mathbf{q})$：从动能 $\mathcal{K} = \frac{1}{2}\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}}$ 提取
# 2. $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})$：通过对 $\mathbf{M}$ 求时间导数 + Christoffel 符号
# 3. $\mathbf{g}(\mathbf{q})$：势能对位形的梯度

# %% [markdown]
# ### 5.3 Christoffel 符号构造 C 矩阵
#
# $$c_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)$$
#
# $$C_{ij} = \sum_{k=1}^n c_{ijk} \dot{q}_k$$
#
# 科氏力/离心力项为 $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} = \sum_k c_{ijk} \dot{q}_j \dot{q}_k$。

# %% [markdown]
# ## 6. 2R 平面臂完整人工推导 ⭐

# %% [markdown]
# ### 6.1 连杆参数
#
# - $m_1, m_2$：质量
# - $l_1, l_2$：连杆长度
# - $l_{c1}, l_{c2}$：质心距关节的距离
# - $I_1, I_2$：绕质心的转动惯量
# - $g$：重力加速度（沿 $-y$ 方向作用，即图面向下）

# %% [markdown]
# ### 6.2 动能
#
# 连杆 1 质心位置和速度：
# $$x_{c1} = l_{c1}\cos q_1, \quad y_{c1} = l_{c1}\sin q_1$$
# $$v_{c1}^2 = l_{c1}^2 \dot{q}_1^2$$
#
# 连杆 2 质心位置和速度：
# $$x_{c2} = l_1\cos q_1 + l_{c2}\cos(q_1+q_2)$$
# $$y_{c2} = l_1\sin q_1 + l_{c2}\sin(q_1+q_2)$$
# $$v_{c2}^2 = l_1^2\dot{q}_1^2 + l_{c2}^2(\dot{q}_1+\dot{q}_2)^2 + 2l_1 l_{c2}\dot{q}_1(\dot{q}_1+\dot{q}_2)\cos q_2$$
#
# 总动能：
# $$\mathcal{K} = \frac{1}{2}m_1 v_{c1}^2 + \frac{1}{2}I_1 \dot{q}_1^2 + \frac{1}{2}m_2 v_{c2}^2 + \frac{1}{2}I_2 (\dot{q}_1+\dot{q}_2)^2$$
#
# 提出 $\frac{1}{2}\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}}$ 形式得到质量矩阵。

# %% [markdown]
# ### 6.3 质量矩阵 M(q)
#
# 引入复合参数简化书写：
# $$\alpha = I_1 + I_2 + m_1 l_{c1}^2 + m_2(l_1^2 + l_{c2}^2)$$
# $$\beta = m_2 l_1 l_{c2}$$
# $$\delta = I_2 + m_2 l_{c2}^2$$
#
# $$\mathbf{M}(\mathbf{q}) = \begin{bmatrix}
# \alpha + 2\beta\cos q_2 & \delta + \beta\cos q_2 \\
# \delta + \beta\cos q_2 & \delta
# \end{bmatrix}$$
#
# 验证：
# - 对称：$M_{12} = M_{21}$ ✓
# - 正定：$\det(\mathbf{M}) = \alpha\delta - (\delta + \beta c_2)^2 + 2\beta\delta c_2 > 0$ ✓

# %% [markdown]
# ### 6.4 科氏力/离心力项 C(q,q̇)q̇
#
# $$\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} = \begin{bmatrix}
# -\beta s_2 \dot{q}_2(2\dot{q}_1 + \dot{q}_2) \\
# \beta s_2 \dot{q}_1^2
# \end{bmatrix}$$
#
# 其中 $s_2 = \sin q_2$。
#
# 注意：$C_{11}$ 含 $\dot{q}_2^2$（离心力），$C_{12}$ 含 $\dot{q}_1\dot{q}_2$（科氏力）。

# %% [markdown]
# ### 6.5 重力项 g(q)
#
# 取重力沿 $-y$ 方向（图面竖直向下）：
# $$\mathbf{g}(\mathbf{q}) = \begin{bmatrix}
# (m_1 l_{c1} + m_2 l_1)g\cos q_1 + m_2 l_{c2} g\cos(q_1+q_2) \\
# m_2 l_{c2} g\cos(q_1+q_2)
# \end{bmatrix}$$

# %% [markdown]
# ## 7. Python 实现与验证 ⭐

# %%
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics, simulate_dynamics
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 7.1 SymPy 符号推导 2R 臂动力学

# %%
# 符号变量
m1, m2, l1, l2, lc1, lc2, I1, I2, g_sym = sym.symbols('m1 m2 l1 l2 lc1 lc2 I1 I2 g', positive=True)
q1, q2, dq1, dq2, ddq1, ddq2 = sym.symbols('q1 q2 dq1 dq2 ddq1 ddq2')
t = sym.symbols('t')

# 质心位置
xc1 = lc1 * sym.cos(q1)
yc1 = lc1 * sym.sin(q1)
xc2 = l1*sym.cos(q1) + lc2*sym.cos(q1+q2)
yc2 = l1*sym.sin(q1) + lc2*sym.sin(q1+q2)

# 质心速度（链式求导）
vc1_sq = sym.diff(xc1, t).subs({sym.Derivative(q1,t): dq1})**2 + sym.diff(yc1, t).subs({sym.Derivative(q1,t): dq1})**2
dxc2_dt = sym.diff(xc2, t).subs({sym.Derivative(q1,t): dq1, sym.Derivative(q2,t): dq2})
dyc2_dt = sym.diff(yc2, t).subs({sym.Derivative(q1,t): dq1, sym.Derivative(q2,t): dq2})
vc2_sq = sym.simplify(dxc2_dt**2 + dyc2_dt**2)

# 动能
K = sym.simplify(0.5*m1*vc1_sq + 0.5*I1*dq1**2 + 0.5*m2*vc2_sq + 0.5*I2*(dq1+dq2)**2)
# 势能（g 沿 -y）
P = sym.simplify(m1*g_sym*yc1 + m2*g_sym*yc2)
# 拉格朗日量
L = K - P

print("动能 K =")
sym.pprint(K)
print("\n势能 P =")
sym.pprint(P)

# %% [markdown]
# ### 7.2 提取 M, C, g

# %%
# 拉格朗日方程: d/dt(∂L/∂q̇) - ∂L/∂q = τ
dL_ddq1 = sym.diff(L, dq1)
dL_ddq2 = sym.diff(L, dq2)

# 时间导数
# 创建 q(t), q̇(t) 的符号函数用于求导
q1_t = sym.Function('q1')(t)
q2_t = sym.Function('q2')(t)

# 用 Christoffel 符号方式提取 M
# M_{ij} = ∂²K/(∂q̇_i ∂q̇_j)
M11 = sym.simplify(sym.diff(K, dq1, dq1))
M12 = sym.simplify(sym.diff(K, dq1, dq2))
M22 = sym.simplify(sym.diff(K, dq2, dq2))

print("质量矩阵元素（化简后）:")
print(f"M11 = {M11}")
print(f"M12 = {M22}")
print(f"M12 = {M12}")

# 重力项: g_i = ∂P/∂q_i
g1 = sym.simplify(sym.diff(P, q1))
g2 = sym.simplify(sym.diff(P, q2))
print(f"\ng1 = {g1}")
print(f"g2 = {g2}")

# %% [markdown]
# ### 7.3 SymPy → NumPy 数值函数 + 与手写显式公式验证

# %%
# 数值参数
p = {m1:1.0, m2:1.0, l1:1.0, l2:0.8, lc1:0.5, lc2:0.4, I1:0.083, I2:0.067, g_sym:9.81}

# 转为 NumPy 函数
M11_fn = sym.lambdify([q1,q2], M11.subs(p), 'numpy')
M12_fn = sym.lambdify([q1,q2], M12.subs(p), 'numpy')
M22_fn = sym.lambdify([q1,q2], M22.subs(p), 'numpy')
g1_fn = sym.lambdify([q1,q2], g1.subs(p), 'numpy')
g2_fn = sym.lambdify([q1,q2], g2.subs(p), 'numpy')

# SymPy 结果
q_test = np.array([np.pi/4, np.pi/3])
M_sym = np.array([[M11_fn(*q_test), M12_fn(*q_test)],
                   [M12_fn(*q_test), M22_fn(*q_test)]])
g_sym = np.array([g1_fn(*q_test), g2_fn(*q_test)])

print("SymPy M(q):")
print(np.round(M_sym, 4))
print(f"SymPy g(q) = {np.round(g_sym, 4)}")

# 与手写显式公式对比
dyn = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=1.0, l2=0.8, lc1=0.5, lc2=0.4, I1=0.083, I2=0.067, g=9.81)
M_manual = dyn.mass_matrix(q_test)
g_manual = dyn.gravity_vector(q_test)

print(f"\n手写 M(q):")
print(np.round(M_manual, 4))
print(f"手写 g(q) = {np.round(g_manual, 4)}")
print(f"\nSymPy vs 手写 M 一致? {np.allclose(M_sym, M_manual)}")
print(f"SymPy vs 手写 g 一致? {np.allclose(g_sym, g_manual)}")

# %% [markdown]
# ### 7.4 验证 Ṁ − 2C 的反对称性

# %%
q_test = np.array([np.pi/3, np.pi/6])
q_dot_test = np.array([1.5, -0.8])

M = dyn.mass_matrix(q_test)
C = dyn.coriolis_matrix(q_test, q_dot_test)

# 数值差分 Ṁ
eps = 1e-6
M_plus = dyn.mass_matrix(q_test + eps * q_dot_test)
M_dot_num = (M_plus - M) / eps

N = M_dot_num - 2 * C
print(f"N = Ṁ - 2C =")
print(np.round(N, 4))
print(f"\nN + N^T = 0? {np.allclose(N + N.T, np.zeros((2,2)), atol=1e-3)}")
print(f"q̇^T N q̇ = {q_dot_test @ N @ q_dot_test:.2e} (应该 ≈ 0)")

# %% [markdown]
# ### 7.5 2R 臂自由摆动仿真

# %%
# 无输入力矩，仅重力作用下的自由摆动
dyn_free = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=1.0, l2=0.8)
q0 = np.array([np.pi/3, -np.pi/4])  # 抬起到一定角度
q_dot0 = np.array([0.0, 0.0])       # 静止释放

def zero_torque(t, q, q_dot):
    return np.zeros(2)

t_hist, q_hist, q_dot_hist, tau_hist = simulate_dynamics(
    dyn_free, q0, q_dot0, zero_torque, t_span=5.0, dt=0.01, method='rk4'
)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# 位置
axes[0].plot(t_hist, np.degrees(q_hist[:, 0]), label='q₁', linewidth=1.5)
axes[0].plot(t_hist, np.degrees(q_hist[:, 1]), label='q₂', linewidth=1.5)
axes[0].set_ylabel('Angle (°)')
axes[0].set_title('2R Arm Free Swing (no torque, gravity only)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 速度
axes[1].plot(t_hist, q_dot_hist[:, 0], label='q̇₁', linewidth=1.5)
axes[1].plot(t_hist, q_dot_hist[:, 1], label='q̇₂', linewidth=1.5)
axes[1].set_ylabel('Velocity (rad/s)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 能量
K_hist = 0.5 * np.array([q_dot_hist[i] @ dyn_free.mass_matrix(q_hist[i]) @ q_dot_hist[i] for i in range(len(t_hist))])
# 简化势能（仅 y 方向重力）
g_energy = 9.81
P_hist = np.zeros(len(t_hist))
for i in range(len(t_hist)):
    y1 = dyn_free.lc1 * np.sin(q_hist[i, 0])
    y2 = dyn_free.l1 * np.sin(q_hist[i, 0]) + dyn_free.lc2 * np.sin(q_hist[i, 0] + q_hist[i, 1])
    P_hist[i] = -dyn_free.m1 * g_energy * y1 - dyn_free.m2 * g_energy * y2
E_total = K_hist + P_hist

axes[2].plot(t_hist, K_hist, label='Kinetic K', linewidth=1.5)
axes[2].plot(t_hist, P_hist, label='Potential P', linewidth=1.5)
axes[2].plot(t_hist, E_total, 'k--', label='Total E = K+P', linewidth=1.5, alpha=0.7)
axes[2].set_ylabel('Energy (J)')
axes[2].set_xlabel('Time (s)')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/09_free_swing.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"总能量变化: {np.std(E_total)/np.mean(E_total)*100:.3f}% (应接近 0)")

# %% [markdown]
# ### 7.6 质量矩阵元素随构型变化

# %%
q2_range = np.linspace(-np.pi, np.pi, 200)
M11_vals = np.zeros_like(q2_range)
M12_vals = np.zeros_like(q2_range)
M22_vals = np.zeros_like(q2_range)
det_M_vals = np.zeros_like(q2_range)

for i, q2v in enumerate(q2_range):
    M = dyn_free.mass_matrix(np.array([0.0, q2v]))
    M11_vals[i] = M[0, 0]
    M12_vals[i] = M[0, 1]
    M22_vals[i] = M[1, 1]
    det_M_vals[i] = np.linalg.det(M)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(np.degrees(q2_range), M11_vals, label='M₁₁', linewidth=2)
ax1.plot(np.degrees(q2_range), M12_vals, label='M₁₂=M₂₁', linewidth=2)
ax1.plot(np.degrees(q2_range), M22_vals, label='M₂₂', linewidth=2)
ax1.set_xlabel('q₂ (°)'); ax1.set_ylabel('Inertia')
ax1.set_title('Mass Matrix Elements vs q₂')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(np.degrees(q2_range), det_M_vals, 'r-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('q₂ (°)'); ax2.set_ylabel('det(M)')
ax2.set_title('det(M(q)) — always > 0 (positive definite)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/09_mass_matrix.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"min det(M) = {np.min(det_M_vals):.4f} > 0 → M 始终正定 ✓")

# %% [markdown]
# ## 8. 常见错误与易混淆概念
#
# 1. **C 矩阵不唯一**：科氏力矩阵 $\mathbf{C}$ 不是唯一确定的——只有 $\mathbf{C}\dot{\mathbf{q}}$ 是唯一确定的。Christoffel 符号构造出的 $\mathbf{C}$ 满足 $\dot{\mathbf{M}}-2\mathbf{C}$ 反对称，这是额外条件。
# 2. **$\mathbf{C}$ 不是对称的**：在一般约定下 $\mathbf{C}$ 不是对称矩阵，这与 $\mathbf{M}$ 不同。
# 3. **符号约定**：重力势能的正负取决于坐标方向。本文中 $y$ 向上 → 重力沿 $-y$ → 势能 $\mathcal{P} = mgy$。
# 4. **拉格朗日 vs 牛顿-欧拉**：拉格朗日适合推导简单机械臂的显式公式；牛顿-欧拉适合计算效率（O(n) 递推，NB11）。

# %% [markdown]
# ## 9. 练习题
#
# ### 概念题
# 1. 正动力学和逆动力学分别解决什么问题？
# 2. 为什么 $\dot{\mathbf{M}}-2\mathbf{C}$ 的反对称性对稳定性分析至关重要？
#
# ### 手算题
# 1. 对 1R 单摆（$m, l, I$）手推拉格朗日动力学方程（应得到 $I\ddot{\theta} + mgl\cos\theta = \tau$）
# 2. 验证 2R 臂 $\dot{\mathbf{M}}-2\mathbf{C}$ 的反对称性（代入手写显式公式）
#
# ### 编程题
# 1. 用 SymPy 推导 3R 平面臂的动力学。
# 2. 实现 M/C/g 函数的自动微分（autodiff）验证与 SymPy 一致。
#
# > 答案见 `solutions/09_10_solutions.ipynb`

# %% [markdown]
# ## 10. 本节总结
#
# | 概念 | 公式 | 说明 |
# |------|------|------|
# | 动力学方程 | $\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g} = \boldsymbol{\tau}$ | 刚体机器人标准形式 |
# | 质量矩阵 | $M_{ij} = \partial^2\mathcal{K}/(\partial\dot{q}_i\partial\dot{q}_j)$ | 对称正定 |
# | Christoffel | $c_{ijk} = \frac{1}{2}(\partial M_{ij}/\partial q_k + \dots)$ | 从 M 构造 C |
# | 反对称性 | $\dot{\mathbf{M}} - 2\mathbf{C} = -(\dot{\mathbf{M}} - 2\mathbf{C})^T$ | 无源性基础 |
# | 正动力学 | $\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{C}\dot{\mathbf{q}} - \mathbf{g})$ | 仿真用 |
# | 逆动力学 | $\boldsymbol{\tau} = \mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g}$ | 控制前馈用 |

# %% [markdown]
# ## 11. 与下一节的联系
#
# NB11 将介绍**牛顿-欧拉递推动力学（RNEA）**——一种 O(n) 复杂度的逆动力学算法，是工程中实际使用的高效方法。我们将实现它并与拉格朗日结果进行一致性验证。
