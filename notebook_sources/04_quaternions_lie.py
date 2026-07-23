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
# # Notebook 04：四元数与李群李代数基础
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB03 旋转表示 ──→ NB04 四元数与李群
# (欧拉角/轴角)     (无奇异性表示 + SO(3)/SE(3)流形)
#                        │
#                        ├──→ NB05 FK（PoE 表示）
#                        ├──→ NB23 EKF（四元数误差状态）
#                        └──→ NB22 KF / 位姿优化
# ```
#
# 四元数是处理三维旋转的"终极工具"——无奇异性、组合方便（乘法）、插值光滑（slerp）。李群视角将旋转理解为流形上的连续运动，是现代状态估计（VIO、SLAM）和机器人优化的数学基础。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解单位四元数的代数结构和几何含义
# - ⭐ 掌握四元数与轴角/旋转矩阵的相互转换
# - ⭐ 实现 slerp（球面线性插值）
# - ⭐ 理解 SO(3) 的指数映射 $\exp: \mathfrak{so}(3) \to SO(3)$ 及其与罗德里格斯公式的等价性
# - ⭐ 理解 BCH 公式在扰动处理中的作用
# - 📖 理解四元数与 $\mathfrak{so}(3)$ 元素的联系
# - 📚 了解对偶四元数和 SE(3) 的李代数

# %% [markdown]
# ## 3. 前置知识
#
# - NB02：旋转矩阵 SO(3)
# - NB03：欧拉角、轴角表示
# - NB01：向量/矩阵运算

# %% [markdown]
# ## 4. 单位四元数 ⭐

# %% [markdown]
# ### 4.1 定义
#
# 四元数（Quaternion）由 Hamilton 于 1843 年引入，是复数在四维空间的推广：
#
# $$\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k} = (w, \mathbf{v})$$
#
# 其中 $\mathbf{v} = [x, y, z]^T$ 是**向量部分**，$w$ 是**标量部分**。
#
# 单位四元数（$\|\mathbf{q}\| = 1$）可以用来表示三维旋转：
#
# $$\mathbf{q} = \left(\cos\frac{\theta}{2}, \sin\frac{\theta}{2} \mathbf{k}\right)$$
#
# 其中 $(\mathbf{k}, \theta)$ 是轴角表示。这个公式揭示了四元数与轴角的直接联系。

# %% [markdown]
# ### 4.2 四元数的性质
#
# | 性质 | 说明 |
# |------|------|
# | 参数数 | 4（但有 $\|\mathbf{q}\|=1$ 约束，有效 DOF = 3） |
# | 双覆盖（Double Cover） | $\mathbf{q}$ 和 $-\mathbf{q}$ 表示同一旋转 |
# | 单位四元数的集合 | 三维球面 $\mathbb{S}^3$ |
# | 奇异性 | **无**！覆盖整个 SO(3) 无奇点 |
# | 旋转组合 | 四元数乘法 $\mathbf{q}_{12} = \mathbf{q}_1 \otimes \mathbf{q}_2$ |
# | 旋转逆 | 共轭 $\mathbf{q}^{-1} = \mathbf{q}^* = (w, -\mathbf{v})$ |

# %% [markdown]
# ### 4.3 四元数乘法（Hamilton 乘积）
#
# $$\mathbf{q}_1 \otimes \mathbf{q}_2 = (w_1 w_2 - \mathbf{v}_1\cdot\mathbf{v}_2, w_1\mathbf{v}_2 + w_2\mathbf{v}_1 + \mathbf{v}_1\times\mathbf{v}_2)$$
#
# 注意：**四元数乘法不满足交换律** $\mathbf{q}_1 \otimes \mathbf{q}_2 \neq \mathbf{q}_2 \otimes \mathbf{q}_1$！这与旋转组合不可交换对应。
#
# 矩阵形式（左乘和右乘）：
#
# $$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix}
# w_1 & -x_1 & -y_1 & -z_1 \\
# x_1 & w_1 & -z_1 & y_1 \\
# y_1 & z_1 & w_1 & -x_1 \\
# z_1 & -y_1 & x_1 & w_1
# \end{bmatrix} \begin{bmatrix} w_2 \\ x_2 \\ y_2 \\ z_2 \end{bmatrix}$$

# %% [markdown]
# ### 4.4 用四元数旋转向量
#
# $$\mathbf{v}' = \mathbf{q} \otimes (0, \mathbf{v}) \otimes \mathbf{q}^*$$
#
# 等价于用旋转矩阵：
# $$\mathbf{v}' = \mathbf{R}(\mathbf{q}) \mathbf{v}$$
#
# 其中旋转矩阵：
#
# $$\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
# 1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\
# 2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\
# 2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2
# \end{bmatrix}$$

# %% [markdown]
# ### 4.5 球面线性插值（slerp）
#
# 在两个四元数 $\mathbf{q}_1$ 和 $\mathbf{q}_2$ 之间进行常速旋转插值：
#
# $$\text{slerp}(\mathbf{q}_1, \mathbf{q}_2, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\mathbf{q}_1 + \frac{\sin(t\Omega)}{\sin\Omega}\mathbf{q}_2$$
#
# 其中 $\Omega = \arccos(\mathbf{q}_1\cdot\mathbf{q}_2)$ 是四元数在 $\mathbb{S}^3$ 上的夹角。
#
# slerp 是**最短路径**（geodesic）上的匀速运动。与直接对欧拉角线性插值对比，slerp 产生的是真正的常角速度旋转。

# %% [markdown]
# ## 5. SO(3) 的李群基础 ⭐

# %% [markdown]
# ### 5.1 什么是李群？
#
# SO(3) = {$\mathbf{R} \in \mathbb{R}^{3\times3} \mid \mathbf{R}^T\mathbf{R} = \mathbf{I}, \det(\mathbf{R})=1$} 既是一个**群**（对矩阵乘法封闭），又是一个**光滑流形**（3 维）。这种"群+流形"的结构称**李群（Lie Group）**。
#
# 流形的含义：SO(3) 在每一点局部看来像 $\mathbb{R}^3$，但全局形状像一个 3 维"球面"（实际上是 $\mathbb{RP}^3$，即 $\mathbb{S}^3$ 的对径点等同）。

# %% [markdown]
# ### 5.2 李代数 $\mathfrak{so}(3)$
#
# SO(3) 在恒等元 $\mathbf{I}$ 处的**切空间**称为李代数 $\mathfrak{so}(3)$：
#
# $$\mathfrak{so}(3) = \{[\boldsymbol{\omega}]_\times \in \mathbb{R}^{3\times3} \mid \boldsymbol{\omega} \in \mathbb{R}^3\}$$
#
# $\mathfrak{so}(3)$ 中的元素是 $3\times3$ 的反对称矩阵，与 $\mathbb{R}^3$ 有一段一一对应。

# %% [markdown]
# ### 5.3 指数映射（Exponential Map）
#
# **指数映射 $\exp: \mathfrak{so}(3) \to SO(3)$** 将切空间中的元素"推"回流形上：
#
# $$\mathbf{R} = \exp([\boldsymbol{\omega}]_\times) = \sum_{n=0}^{\infty} \frac{[\boldsymbol{\omega}]_\times^n}{n!}$$
#
# 这个无穷级数有**闭式解**——正是罗德里格斯公式！
#
# $$\exp([\boldsymbol{\omega}]_\times) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2$$
#
# 其中 $\theta = \|\boldsymbol{\omega}\|$。
#
# 物理含义：$\boldsymbol{\omega} = \theta\mathbf{k}$ 是旋转向量；$\exp$ 将其"放大"为有限旋转 $\mathbf{R} = \exp([\theta\mathbf{k}]_\times)$。

# %% [markdown]
# ### 5.4 对数映射（Logarithm Map）
#
# **对数映射 $\log: SO(3) \to \mathfrak{so}(3)$** 是从流形回到切空间：
#
# $$\theta = \arccos\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right)$$
# $$\log(\mathbf{R}) = \frac{\theta}{2\sin\theta}(\mathbf{R} - \mathbf{R}^T)$$
#
# $\exp$ 和 $\log$ 互逆：$\exp(\log(\mathbf{R})) = \mathbf{R}$，$\log(\exp([\boldsymbol{\omega}]_\times)) = [\boldsymbol{\omega}]_\times$（除 $\theta=\pi$ 外）。

# %% [markdown]
# ### 5.5 BCH 公式与扰动
#
# Baker-Campbell-Hausdorff (BCH) 公式给出了两个李代数元素指数映射乘积的李代数：
#
# $$\log(\exp([\mathbf{a}]_\times)\exp([\mathbf{b}]_\times)) \approx [\mathbf{a} + \mathbf{b}]_\times + \frac{1}{2}[\mathbf{a}, \mathbf{b}] + \cdots$$
#
# **工程用法（小角度近似）**：
# 当 $\mathbf{b}$ 很小（如一个微小的姿态扰动）时：
# $$\exp([\mathbf{a}]_\times)\exp([\mathbf{b}]_\times) \approx \exp([\mathbf{a} + \mathbf{J}_l(\mathbf{a})^{-1}\mathbf{b}]_\times)$$
#
# 其中 $\mathbf{J}_l$ 是 SO(3) 的**左雅可比**：
# $$\mathbf{J}_l(\boldsymbol{\omega}) = \mathbf{I} + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\omega}]_\times^2$$
#
# 在 EKF（NB23）和位姿图优化中，BCH 的近似用于处理姿态误差状态。

# %% [markdown]
# ### 5.6 SE(3) 简介
#
# SE(3) = SO(3) ⋉ $\mathbb{R}^3$（旋转 + 平移的半直积）具有对应的李代数 $\mathfrak{se}(3)$：
#
# $$\boldsymbol{\xi} = \begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^6 \longleftrightarrow [\boldsymbol{\xi}]_\wedge = \begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{bmatrix} \in \mathfrak{se}(3)$$
#
# 指数映射 $\exp: \mathfrak{se}(3) \to SE(3)$ 产生齐次变换矩阵。

# %% [markdown]
# ## 6. Python 实现与可视化

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '..')
from src.robotics_learning.transforms import (
    Quaternion, slerp, so3_exp, so3_log, se3_exp,
    axis_angle_to_rot, rot_to_axis_angle, skew
)
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 轴角 ↔ 四元数 ↔ 旋转矩阵

# %%
# 轴角 → 四元数 → 旋转矩阵 → 四元数（往返验证）
axis = np.array([1.0, 0.5, 0.2])
axis = axis / np.linalg.norm(axis)
angle = np.pi / 3

q = Quaternion.from_axis_angle(axis, angle)
print(f"四元数: {q}")
print(f"范数: {np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2):.6f} (应为 1)")

# 四元数 → 旋转矩阵
R_from_q = q.to_rot()
print(f"\nR (from quaternion):\n{np.round(R_from_q, 4)}")

# 验证与轴角直接构造的旋转矩阵一致
R_from_aa = axis_angle_to_rot(axis, angle)
print(f"\n与轴角直接构造一致? {np.allclose(R_from_q, R_from_aa, atol=1e-10)}")

# 旋转矩阵 → 四元数 → 旋转矩阵（往返）
q_back = Quaternion.from_rot(R_from_q)
R_back = q_back.to_rot()
print(f"往返一致? {np.allclose(R_from_q, R_back, atol=1e-10)}")

# %% [markdown]
# ### 6.2 用四元数旋转向量

# %%
v = np.array([1.0, 2.0, 3.0])
v_rotated_q = q.rotate_vector(v)
v_rotated_R = R_from_q @ v
print(f"q ⊗ v ⊗ q*:  {np.round(v_rotated_q, 4)}")
print(f"R @ v:        {np.round(v_rotated_R, 4)}")
print(f"一致? {np.allclose(v_rotated_q, v_rotated_R)}")

# %% [markdown]
# ### 6.3 slerp vs 欧拉角线性插值

# %%
# 两个四元数
q1 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0)
q2 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), 3*np.pi/4)  # 135°

t_vals = np.linspace(0, 1, 50)
v_test = np.array([1.0, 0.0, 0.0])

# slerp
v_slerp = np.zeros((50, 3))
for i, t in enumerate(t_vals):
    qi = slerp(q1, q2, t)
    v_slerp[i] = qi.rotate_vector(v_test)

# 对比：直接对欧拉角线性插值
yaw1, yaw2 = 0, 3*np.pi/4
v_euler_lerp = np.zeros((50, 3))
for i, t in enumerate(t_vals):
    yaw = (1-t)*yaw1 + t*yaw2
    R = axis_angle_to_rot(np.array([0.0, 0.0, 1.0]), yaw)
    v_euler_lerp[i] = R @ v_test

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# XY 平面轨迹
axes[0].plot(v_slerp[:, 0], v_slerp[:, 1], 'b-', linewidth=2, label='slerp')
axes[0].plot(v_euler_lerp[:, 0], v_euler_lerp[:, 1], 'r--', linewidth=2, label='Euler lerp')
axes[0].scatter(*v_test[:2], c='green', s=100, marker='o', label='Start')
axes[0].scatter(*v_slerp[-1, :2], c='red', s=100, marker='x', label='End')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
axes[0].set_title('Vector Trajectory (XY Projection)')
axes[0].set_aspect('equal'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 旋转角随时间变化
theta_slerp = np.arctan2(v_slerp[:, 1], v_slerp[:, 0])
theta_euler = np.arctan2(v_euler_lerp[:, 1], v_euler_lerp[:, 0])
axes[1].plot(t_vals, np.degrees(theta_slerp), 'b-', linewidth=2, label='slerp')
axes[1].plot(t_vals, np.degrees(theta_euler), 'r--', linewidth=2, label='Euler lerp')
axes[1].set_xlabel('t'); axes[1].set_ylabel('Angle (°)')
axes[1].set_title('Rotation Angle vs Time')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 角速度（数值差分）
omega_slerp = np.diff(theta_slerp) / (t_vals[1] - t_vals[0])
omega_euler = np.diff(theta_euler) / (t_vals[1] - t_vals[0])
axes[2].plot(t_vals[1:], omega_slerp, 'b-', linewidth=2, label='slerp')
axes[2].plot(t_vals[1:], omega_euler, 'r--', linewidth=2, label='Euler lerp')
axes[2].set_xlabel('t'); axes[2].set_ylabel('Angular velocity')
axes[2].set_title('Angular Velocity (constant for slerp!)')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/04_slerp_vs_euler.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ slerp 产生常角速度旋转，欧拉角线性插值则不然！")
print("   这就是为什么动画和姿态估计中首选四元数 slerp。")

# %% [markdown]
# ### 6.4 SO(3) 指数映射与对数映射

# %%
omega = np.array([0.5, -0.3, 0.8])
print(f"so(3) 元素 ω = {omega}")

# exp: so(3) → SO(3)
R_exp = so3_exp(omega)
print(f"\nexp(ω):\n{np.round(R_exp, 4)}")

# 验证：与罗德里格斯公式一致
theta = np.linalg.norm(omega)
axis = omega / theta
R_rodrigues = axis_angle_to_rot(axis, theta)
print(f"\nexp = Rodrigues? {np.allclose(R_exp, R_rodrigues, atol=1e-10)}")

# log: SO(3) → so(3)
omega_back = so3_log(R_exp)
print(f"\nlog(R) = {np.round(omega_back, 4)}")
print(f"log(exp(ω)) = ω? {np.allclose(omega, omega_back, atol=1e-10)}")

# %% [markdown]
# ### 6.5 四元数球面可视化

# %%
# 用 Hopf 坐标可视化四元数旋转在 S^3 上的路径
# slerp 产生大圆（geodesic）上的匀速运动

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 四元数分量随 t 的变化
t_fine = np.linspace(0, 1, 100)
q_components = np.zeros((100, 4))
for i, t in enumerate(t_fine):
    qi = slerp(q1, q2, t)
    q_components[i] = [qi.w, qi.x, qi.y, qi.z]

axes[0].plot(t_fine, q_components[:, 0], label='w', linewidth=2)
axes[0].plot(t_fine, q_components[:, 1], '--', label='x', linewidth=2)
axes[0].plot(t_fine, q_components[:, 2], '--', label='y', linewidth=2)
axes[0].plot(t_fine, q_components[:, 3], '--', label='z', linewidth=2)
axes[0].set_xlabel('t'); axes[0].set_ylabel('Component value')
axes[0].set_title('Quaternion Components during slerp')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 四元数轨迹在 w-x 投影（简化的 S^3 → 圆）
axes[1].plot(q_components[:, 0], q_components[:, 1], 'b-', linewidth=2)
axes[1].scatter(*q_components[0, :2], c='green', s=100, zorder=5, label='Start')
axes[1].scatter(*q_components[-1, :2], c='red', s=100, zorder=5, label='End')
# 单位圆
theta_circle = np.linspace(0, 2*np.pi, 100)
axes[1].plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.2)
axes[1].set_xlabel('w'); axes[1].set_ylabel('x')
axes[1].set_title('Slerp Path (w-x projection of S³)')
axes[1].set_aspect('equal'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([-1.2, 1.2]); axes[1].set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('../outputs/04_slerp_components.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 常见错误与易混淆概念
#
# 1. **$\mathbf{q}$ 和 $-\mathbf{q}$ 表示同一旋转**：单位四元数双覆盖 SO(3)。在 slerp 中如果不检查符号，插值路径可能绕远路。
# 2. **四元数参数的顺序**：本文用 $(w, x, y, z)$（标量在前）。有些库用 $(x, y, z, w)$（向量在前，如 ROS/ Eigen）。使用前必须确认。
# 3. **$\mathfrak{so}(3)$ 的元素**：$\mathfrak{so}(3)$ 是 3×3 反对称矩阵。但在计算中，通常只存储其生成向量 $\boldsymbol{\omega} \in \mathbb{R}^3$（3 个元素），因为 $[\boldsymbol{\omega}]_\times \leftrightarrow \boldsymbol{\omega}$ 是一一对应。
# 4. **指数映射中的 $\theta$** ：$\exp([\boldsymbol{\omega}]_\times)$ 中 $\theta = \|\boldsymbol{\omega}\|$。小角度近似 $\exp([\boldsymbol{\omega}]_\times) \approx \mathbf{I} + [\boldsymbol{\omega}]_\times$ 只在 $\theta \ll 1$ 时成立。

# %% [markdown]
# ## 8. 工程应用
#
# - **姿态估计（Attitude Estimation）**：EKF 用四元数误差状态（NB23）
# - **SLAM / VIO**：位姿图优化在 SE(3) 流形上进行（gtsam, g2o）
# - **动画与游戏**：slerp 用于相机和角色旋转
# - **无人机**：PX4/ArduPilot 用四元数表示姿态

# %% [markdown]
# ## 9. 面试常见问题
#
# 1. **四元数和欧拉角各有什么优缺点？** → INTERVIEW_CHECKLIST #1.2
# 2. **SO(3) 和 so(3) 的关系？** → #1.7
# 3. **exp: so(3) → SO(3) 的公式和工程用途？** → #1.8

# %% [markdown]
# ## 10. 练习题
#
# ### 概念题
# 1. 为什么单位四元数没有奇异性而欧拉角有？
# 2. $\exp$ 和 $\log$ 映射在机器人扰动分析中的作用是什么？
#
# ### 手算题
# 1. 给定轴角 $(\mathbf{k}=[0,0,1]^T, \theta=60°)$，求对应的单位四元数。
# 2. 计算两个四元数 $\mathbf{q}_1 = (1,0,0,0)$ 和 $\mathbf{q}_2 = (0.707, 0.707, 0, 0)$ 的 slerp(t=0.5)。
#
# ### 编程题
# 1. 实现 slerp 并验证插值路径是 S³ 上的大圆。
# 2. 比较四元数 slerp、欧拉角线性插值和轴角线性插值的角速度。
#
# > 答案见 `solutions/04_solutions.ipynb`

# %% [markdown]
# ## 11. 本节总结
#
# | 概念 | 公式/符号 | 核心性质 |
# |------|-----------|----------|
# | 单位四元数 | $\mathbf{q} = (\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\mathbf{k})$ | 无奇异性, $\mathbb{S}^3$ 双覆盖 SO(3) |
# | slerp | $\frac{\sin((1-t)\Omega)}{\sin\Omega}\mathbf{q}_1 + \frac{\sin(t\Omega)}{\sin\Omega}\mathbf{q}_2$ | 最短路径匀速插值 |
# | $\exp([\boldsymbol{\omega}]_\times)$ | $\mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2$ | = 罗德里格斯公式 |
# | $\log(\mathbf{R})$ | $\frac{\theta}{2\sin\theta}(\mathbf{R} - \mathbf{R}^T)$ | 恢复旋转向量 |
# | 左雅可比 $\mathbf{J}_l$ | $\approx \mathbf{I}$ 小角度 | 姿态误差传播 |
# | $\mathfrak{se}(3)$ | $[\mathbf{v}^T, \boldsymbol{\omega}^T]^T$ | SE(3) 的切空间 |

# %% [markdown]
# ## 12. 与下一节的联系
#
# 下一节（NB05）将使用齐次变换矩阵和 DH 参数构建机械臂的**正运动学**模型。我们将看到这些数学工具如何串联起来——每个关节的齐次变换连乘即得到末端位姿。
