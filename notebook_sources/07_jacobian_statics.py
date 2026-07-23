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
# # Notebook 07：雅可比矩阵、速度运动学与静力学
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB05 FK  ──→ NB07 雅可比 ──→ NB08 奇异性/可操作度
#                │
#                ├──→ NB06 数值 IK (伪逆用 J)
#                ├──→ NB19 操作空间控制 (J^T 力映射)
#                └──→ NB20 力控制 (虚功原理 τ = J^T F)
# ```
#
# 雅可比矩阵 $\mathbf{J}(\mathbf{q})$ 是关节空间与操作空间之间的**瞬时线性映射**。它回答了：\textbf{"关节在转，末端怎么动？"（速度映射 $\dot{\mathbf{x}} = \mathbf{J}\dot{\mathbf{q}}$）和"末端有力的作用，关节需要多少力矩？"（力映射 $\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$）。}

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解几何雅可比与解析雅可比的本质区别
# - ⭐ 掌握几何雅可比的类几何构造法（列 = 关节轴 $\times$ 位置矢量）
# - ⭐ 从虚功原理推导 $\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$
# - ⭐ 理解力域与速度域的对偶关系
# - 📖 雅可比的数值差分验证
# - 📚 闭链和移动基座雅可比

# %% [markdown]
# ## 3. 前置知识
#
# - NB05：正运动学、DH 参数、FK 变换链
# - NB02：齐次变换、叉乘矩阵 $[\boldsymbol{\omega}]_\times$
# - NB01：矩阵伪逆

# %% [markdown]
# ## 4. 几何雅可比 ⭐

# %% [markdown]
# ### 4.1 定义
#
# 几何雅可比 $\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{6 \times n}$ 将关节速度映射到末端**空间速度**（spatial velocity / twist）：
#
# $$\begin{bmatrix} \mathbf{v}_e \\ \boldsymbol{\omega}_e \end{bmatrix} = \mathbf{J}(\mathbf{q})\dot{\mathbf{q}}$$
#
# - $\mathbf{v}_e \in \mathbb{R}^3$：末端相对于基座参考系的**线速度**
# - $\boldsymbol{\omega}_e \in \mathbb{R}^3$：末端相对于基座参考系的**角速度**
# - $\dot{\mathbf{q}} \in \mathbb{R}^n$：关节速度
#
# 注意：这里的线速度和角速度都是相对于**基座参考系 {0}** 表达的。

# %% [markdown]
# ### 4.2 逐列构造 — 几何直觉
#
# 对于旋转关节 $i$（最常见情况），第 $i$ 列 $\mathbf{J}_i$ 表示**仅第 $i$ 个关节运动 1 rad/s 时，末端产生的速度**：
#
# $$\mathbf{J}_i = \begin{bmatrix} \mathbf{z}_{i-1} \times (\mathbf{p}_e - \mathbf{p}_{i-1}) \\ \mathbf{z}_{i-1} \end{bmatrix}$$
#
# 其中：
# - $\mathbf{z}_{i-1}$：关节 $i$ 的旋转轴（在基座系 {0} 中），从 FK 变换 ${}^{0}_{i-1}\mathbf{T}$ 的第三列提取
# - $\mathbf{p}_e$：末端原点位置（在 {0} 中）
# - $\mathbf{p}_{i-1}$：连杆 $i-1$ 的原点位置（在 {0} 中）
#
# **上半部分** $[\mathbf{z} \times (\mathbf{p}_e - \mathbf{p})]$：关节旋转 $\dot{\theta}_i$ 在关节轴 $\mathbf{z}$ 上产生角速度 $\dot{\theta}_i\mathbf{z}$，该角速度又通过叉乘 $\boldsymbol{\omega} \times \mathbf{r}$ 在末端产生线速度。
#
# **下半部分** $[\mathbf{z}]$：关节旋转直接贡献末端角速度。

# %% [markdown]
# ### 4.3 移动关节的列
#
# 对于移动关节（较少见）：
# $$\mathbf{J}_i = \begin{bmatrix} \mathbf{z}_{i-1} \\ \mathbf{0} \end{bmatrix}$$
# 移动关节只产生线速度，不产生角速度。

# %% [markdown]
# ## 5. 解析雅可比 ⭐

# %% [markdown]
# ### 5.1 定义与区别
#
# 解析雅可比 $\mathbf{J}_A$ 将关节速度映射到**某种位姿表示参数**的导数（如欧拉角速率）：
#
# $$\dot{\boldsymbol{\phi}} = \mathbf{J}_A(\mathbf{q})\dot{\mathbf{q}}$$
#
# 其中 $\boldsymbol{\phi} = [x, y, z, \alpha, \beta, \gamma]^T$（如 ZYX 欧拉角）。
#
# 几何雅可比与解析雅可比的关系：
# $$\mathbf{J}(\mathbf{q}) = \mathbf{B}(\boldsymbol{\phi}) \mathbf{J}_A(\mathbf{q})$$
#
# $$ \mathbf{B}(\boldsymbol{\phi}) = \begin{bmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{T}(\boldsymbol{\phi}) \end{bmatrix}$$
#
# 其中 $\mathbf{T}(\boldsymbol{\phi})$ 将姿态参数导数转换为角速度。
#
# 对于 ZYX 欧拉角：$\boldsymbol{\omega} = \begin{bmatrix} 0 & -s_\alpha & c_\alpha c_\beta \\ 0 & c_\alpha & s_\alpha c_\beta \\ 1 & 0 & -s_\beta \end{bmatrix} \begin{bmatrix} \dot{\alpha} \\ \dot{\beta} \\ \dot{\gamma} \end{bmatrix}$

# %% [markdown]
# ### 5.2 何时需要区分？
#
# - **控制**：如果控制器在操作空间中使用姿态最小表示（如欧拉角误差），需要解析雅可比
# - **力域**：$\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$ 中的 $\mathbf{J}$ 是**几何雅可比**（因为力和力矩直接作用在物理空间）
# - **大多数应用**：几何雅可比已足够

# %% [markdown]
# ## 6. 静力学与雅可比转置 ⭐

# %% [markdown]
# ### 6.1 虚功原理推导
#
# 假设机械臂在静态平衡中。末端受外力/力矩 $\mathbf{F} = [\mathbf{f}^T, \mathbf{n}^T]^T$。根据**虚功原理**（Principle of Virtual Work）：
#
# 末端虚位移 $\delta\mathbf{x}$ 做的功 = 关节虚位移 $\delta\mathbf{q}$ 做的功：
# $$\delta W = \mathbf{F}^T \delta\mathbf{x} = \boldsymbol{\tau}^T \delta\mathbf{q}$$
#
# 由速度运动学 $\delta\mathbf{x} = \mathbf{J}\delta\mathbf{q}$：
# $$\mathbf{F}^T \mathbf{J} \delta\mathbf{q} = \boldsymbol{\tau}^T \delta\mathbf{q}$$
#
# 这对任意 $\delta\mathbf{q}$ 成立，因此：
# $$\boxed{\boldsymbol{\tau} = \mathbf{J}^T(\mathbf{q}) \mathbf{F}}$$

# %% [markdown]
# ### 6.2 物理直觉
#
# - $\mathbf{J}$ 的每一列表示"关节转 1 rad/s 时末端的运动"
# - $\mathbf{J}^T$ 的每一行表示"末端受 1 N 力时，该关节需要的力矩"
# - $\mathbf{J}^T$ 本质上是在计算**力臂**（moment arm）

# %% [markdown]
# ### 6.3 非线性效应 vs 线性映射
#
# **重要澄清**：
# - $\dot{\mathbf{x}} = \mathbf{J}(\mathbf{q})\dot{\mathbf{q}}$ 是**瞬时线性关系**（在给定 $\mathbf{q}$ 下成立）
# - $\boldsymbol{\tau} = \mathbf{J}^T(\mathbf{q})\mathbf{F}$ 也是**瞬时线性关系**（在给定 $\mathbf{q}$ 下成立）
# - 在更大的运动范围内（$\mathbf{q}$ 变化时），$\mathbf{J}(\mathbf{q})$ 是 $\mathbf{q}$ 的函数——这是非线性的!

# %% [markdown]
# ## 7. Python 实现与可视化

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '..')
from src.robotics_learning.kinematics import (
    forward_kinematics, compute_geometric_jacobian, compute_analytical_jacobian
)
from src.robotics_learning.transforms import skew
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 7.1 逐列构造雅可比 + 数值差分验证

# %%
# 3R 空间臂
dh = np.array([[0.0, np.pi/2, 1.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]])
q = np.array([np.pi/4, np.pi/6, -np.pi/3])

# 几何雅可比
J = compute_geometric_jacobian(dh, q)
print("几何雅可比 J (6×3):")
print(np.round(J, 4))

# 用数值差分验证
eps = 1e-6
J_numerical = np.zeros((6, 3))
dh_ref = np.column_stack([dh, q])
T_ref, _ = forward_kinematics(dh_ref)
p_ref = T_ref[:3, 3]
R_ref = T_ref[:3, :3]

for i in range(3):
    q_plus = q.copy()
    q_plus[i] += eps
    dh_plus = np.column_stack([dh, q_plus])
    T_plus, _ = forward_kinematics(dh_plus)
    # 位置差分
    J_numerical[:3, i] = (T_plus[:3, 3] - p_ref) / eps
    # 姿态差分（用角速度近似）
    dR = T_plus[:3, :3] @ R_ref.T
    omega_approx = 0.5 * np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])
    J_numerical[3:, i] = omega_approx / eps

print(f"\n数值差分 vs 几何构造: {np.allclose(J, J_numerical, atol=1e-4)}")
print(f"最大差异: {np.max(np.abs(J - J_numerical)):.2e}")

# %% [markdown]
# ### 7.2 速度映射验证

# %%
q_dot = np.array([0.5, -0.3, 0.8])  # 关节速度 rad/s

# 几何雅可比映射
twist = J @ q_dot
v_pred = twist[:3]
omega_pred = twist[3:]
print(f"预测末端速度: v = {np.round(v_pred, 4)}, ω = {np.round(omega_pred, 4)}")

# 数值验证：小时间步后 FK 变化
dt = 0.001
q_new = q + q_dot * dt
dh_new = np.column_stack([dh, q_new])
T_new, _ = forward_kinematics(dh_new)
v_numerical = (T_new[:3, 3] - T_ref[:3, 3]) / dt
dR = T_new[:3, :3] @ T_ref[:3, :3].T
omega_numerical = 0.5 * np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / dt

print(f"数值验证:     v = {np.round(v_numerical, 4)}, ω = {np.round(omega_numerical, 4)}")
print(f"线速度一致? {np.allclose(v_pred, v_numerical, atol=1e-3)}")
print(f"角速度一致? {np.allclose(omega_pred, omega_numerical, atol=1e-3)}")

# %% [markdown]
# ### 7.3 静力学：虚功原理验证

# %%
# 在末端施加外力
F_ext = np.array([10.0, -5.0, 0.0, 0.0, 0.0, 2.0])  # [f_x, f_y, f_z, n_x, n_y, n_z]
tau_JT = J.T @ F_ext
print(f"τ = J^T F = {np.round(tau_JT, 4)}")

# 验证虚功原理：τ^T δq = F^T δx
delta_q = np.array([0.01, -0.005, 0.008])
delta_x = J @ delta_q

work_joint = tau_JT @ delta_q
work_ee = F_ext @ delta_x
print(f"\n关节功 τ^T δq = {work_joint:.6f}")
print(f"末端功 F^T δx = {work_ee:.6f}")
print(f"虚功原理成立? {np.allclose(work_joint, work_ee, atol=1e-10)}")

# %% [markdown]
# ### 7.4 力椭球与速度椭球的对偶性

# %%
# 对 2R 臂在某个构型下分析速度和力传递
q_2r = np.array([np.pi/3, -np.pi/4])
dh_2r = np.array([[1.0, 0, 0], [0.8, 0, 0]])
J_2r = compute_geometric_jacobian(dh_2r, q_2r)[:2, :2]  # 只取 XY 线速度

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 速度椭球：||q̇|| ≤ 1 → ẋ^T (JJ^T)⁻¹ ẋ ≤ 1
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])  # ||q̇|| = 1
x_dot_ellipse = J_2r @ circle

# 力椭球：||F|| ≤ 1 → τ^T (JJ^T) τ ≤ 1
F_circle = np.array([np.cos(theta), np.sin(theta)])
tau_ellipse = J_2r.T @ F_circle

axes[0].fill(x_dot_ellipse[0], x_dot_ellipse[1], color='blue', alpha=0.3)
axes[0].plot(x_dot_ellipse[0], x_dot_ellipse[1], 'b-', linewidth=2)
axes[0].set_title('Velocity Ellipsoid: $\\dot{x} = J \\dot{q}$, $\\|\\dot{q}\\| \\leq 1$')
axes[0].set_xlabel('$\\dot{x}$'); axes[0].set_ylabel('$\\dot{y}$')
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

axes[1].fill(tau_ellipse[0], tau_ellipse[1], color='red', alpha=0.3)
axes[1].plot(tau_ellipse[0], tau_ellipse[1], 'r-', linewidth=2)
axes[1].set_title('Force Ellipsoid: $\\tau = J^T F$, $\\|F\\| \\leq 1$')
axes[1].set_xlabel('$\\tau_1$'); axes[1].set_ylabel('$\\tau_2$')
axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

# 标注主轴
U_v, s_v, _ = np.linalg.svd(J_2r @ J_2r.T)
U_f, s_f, _ = np.linalg.svd(J_2r.T @ J_2r)

for i in range(2):
    v_dir = np.sqrt(s_v[i]) * U_v[:, i]
    axes[0].arrow(0, 0, v_dir[0], v_dir[1], head_width=0.08, fc='darkblue', ec='darkblue')
    f_dir = np.sqrt(s_f[i]) * U_f[:, i]
    axes[1].arrow(0, 0, f_dir[0], f_dir[1], head_width=0.08, fc='darkred', ec='darkred')

plt.tight_layout()
plt.savefig('../outputs/07_velocity_force_ellipsoid.png', dpi=100, bbox_inches='tight')
plt.show()

print("速度椭球的主轴 = JJ^T 的特征方向（末端最能动的方向）")
print("力椭球的主轴 = J^T J 的特征方向（关节力矩最能产生末端力的方向）")
print(f"速度椭球主轴长度: √λ₁={np.sqrt(s_v[0]):.3f}, √λ₂={np.sqrt(s_v[1]):.3f}")
print(f"力椭球主轴长度:   √λ₁={np.sqrt(s_f[0]):.3f}, √λ₂={np.sqrt(s_f[1]):.3f}")

# %% [markdown]
# ## 8. 常见错误与易混淆概念
#
# 1. **几何 vs 解析**：$\mathbf{J}$ 映射到物理角速度（rad/s），$\mathbf{J}_A$ 映射到姿态参数导数（如 °/s）。这两者的数值不同！在力域中必须用几何雅可比。
# 2. **雅可比依赖构型**：$\mathbf{J}(\mathbf{q})$ 是 $\mathbf{q}$ 的函数——每次 $\mathbf{q}$ 改变都需要重新计算。
# 3. **转置不是逆**：$\mathbf{J}^T$ 和 $\mathbf{J}^{-1}$ 不是同一个东西。$\mathbf{J}^T$ 映射力，$\mathbf{J}^{-1}$（或 $\mathbf{J}^+$) 映射速度逆。

# %% [markdown]
# ## 9. 工程应用
#
# - **力控**：$\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$ 用于将操作空间力指令转为关节力矩
# - **速度控制**：$\dot{\mathbf{q}} = \mathbf{J}^{-1}\dot{\mathbf{x}}$ 用于分解运动速率控制
# - **力估计**：从关节力矩估计末端力 $\mathbf{F} = (\mathbf{J}^T)^+\boldsymbol{\tau}$

# %% [markdown]
# ## 10. 练习题
#
# ### 概念题
# 1. 几何雅可比和解析雅可比的本质区别是什么？
# 2. 为什么力用 $\mathbf{J}^T$ 而不是 $\mathbf{J}^{-1}$？
#
# ### 手算题
# 1. 为 2R 臂 ($l_1=1, l_2=0.8$) 在 $q_1=30°, q_2=45°$ 处手算雅可比矩阵。
#
# ### 编程题
# 1. 实现雅可比的数值差分验证函数——对比解析构造和差分近似的所有元素。
# 2. 计算并可视化 2R 臂整个工作空间内的力椭球分布。
#
# > 答案见 `solutions/07_solutions.ipynb`

# %% [markdown]
# ## 11. 本节总结
#
# | 概念 | 公式 | 解释 |
# |------|------|------|
# | 几何雅可比 | $\dot{\mathbf{x}} = \mathbf{J}\dot{\mathbf{q}}$ | 关节速度 → 末端空间速度 |
# | 雅可比列 | $[\mathbf{z}\times(\mathbf{p}_e-\mathbf{p}); \mathbf{z}]$ | 第 i 个关节的贡献 |
# | 静力学 | $\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$ | 虚功原理 |
# | 解析雅可比 | $\dot{\boldsymbol{\phi}} = \mathbf{J}_A\dot{\mathbf{q}}$ | 关节速度 → 姿态参数速率 |

# %% [markdown]
# ## 12. 与下一节的联系
#
# NB08 将探讨 $\mathbf{J}(\mathbf{q})$ 在什么时候"失效"——即**奇异性**。当 $\det(\mathbf{J}) = 0$ 时，某些方向的末端速度无法由关节速度产生，同时 $\mathbf{J}^T$ 在力域也有对称的问题。我们还将介绍**可操作度椭球**作为量化机器人灵活性（dexterity）的工具。
