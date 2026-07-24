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
# # Notebook 05：正运动学（Forward Kinematics）
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB02 刚体变换/SE(3) ──→ NB05 正运动学 ──→ NB06 逆运动学
#                               │
#                               └──→ NB07 雅可比 (∂FK/∂q)
#                               └──→ NB10 动力学 (动能需要速度)
# ```
#
# 正运动学回答机器人学最基础的问题：**"给定每个关节的角度，末端在哪里？"**

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解串联机械臂的结构描述（连杆、关节、关节变量）
# - ⭐ 掌握 **Denavit-Hartenberg (DH) 参数**的物理含义和坐标系附着规则
# - ⭐ 能区分标准 DH 和改进 DH 并正确使用
# - ⭐ 能手动推导 2R、3R 平面臂和简单空间臂的 FK
# - ⭐ 用 Python 实现通用 FK 函数
# - 📖 理解 SDH 和 MDH 两种约定的差异
# - 📚 了解 Product of Exponentials (PoE) 表示法

# %% [markdown]
# ## 3. 前置知识
#
# - NB02：齐次变换矩阵、旋转矩阵
# - NB04：旋转表示（尤其是轴角）

# %% [markdown]
# ## 4. 串联机械臂的结构描述 ⭐
#
# 串联机械臂（Serial Manipulator）由 **$n$ 个连杆（Links）** 通过 **$n$ 个关节（Joints）** 串联而成。从基座到末端依次编号 $1, 2, \dots, n$。
#
# 每个关节只有一个自由度——旋转关节（Revolute）或移动关节（Prismatic）。用一个**关节变量**来描述：
# - 旋转关节：$q_i = \theta_i$（角度）
# - 移动关节：$q_i = d_i$（位移）

# %% [markdown]
# ## 5. Denavit-Hartenberg (DH) 参数 ⭐

# %% [markdown]
# ### 5.1 核心思想
#
# 相邻连杆 $i-1$ 和 $i$ 之间的相对位姿可以用 **4 个参数**完全描述。这就是 DH 参数的核心洞察——任意两个空间系之间的变换（6 DOF）只需要 4 个参数，因为两个额外的自由度被"坐标系附着规则"消耗。
#
# 这 4 个参数是：
#
# | 参数 | 符号 | 含义 | 单位 |
# |------|:----:|------|:----:|
# | 连杆长度 (Link Length) | $a_i$ | 沿 $X_i$ 轴，$Z_{i-1}$ 到 $Z_i$ 的最近距离 | m |
# | 连杆扭转角 (Link Twist) | $\alpha_i$ | 绕 $X_i$ 轴，$Z_{i-1}$ 到 $Z_i$ 的转角 | rad |
# | 连杆偏置 (Link Offset) | $d_i$ | 沿 $Z_{i-1}$ 轴，$X_{i-1}$ 到 $X_i$ 的最近距离 | m |
# | 关节角 (Joint Angle) | $\theta_i$ | 绕 $Z_{i-1}$ 轴，$X_{i-1}$ 到 $X_i$ 的转角 | rad |

# %% [markdown]
# ### 5.2 标准 DH（Standard DH / SDH）
#
# **坐标系附着规则**（标准 DH）：
# 1. $Z_{i-1}$ 轴 = 关节 $i$ 的转轴（$\theta_i$ 的旋转轴）
# 2. $X_i$ 轴 = 沿 $Z_{i-1}$ 与 $Z_i$ 的公垂线方向（从 Z_{i-1} 指向 Z_i）
# 3. $Y_i$ 轴 = 右手定则补全
# 4. 原点 = 公垂线与 $Z_i$ 的交点（对关节 i+1 来说）
#
# **单连杆变换**（关键公式！）：
#
# $${}^{i-1}_{i}\mathbf{T} = \mathbf{R}_z(\theta_i) \cdot \mathbf{T}_z(d_i) \cdot \mathbf{T}_x(a_i) \cdot \mathbf{R}_x(\alpha_i)$$
#
# 展开：
# $${}^{i-1}_{i}\mathbf{T} = \begin{bmatrix}
# \cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
# \sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
# 0 & \sin\alpha_i & \cos\alpha_i & d_i \\
# 0 & 0 & 0 & 1
# \end{bmatrix}$$

# %% [markdown]
# ### 5.3 改进 DH（Modified DH / Khalil-Kleinfinger）
#
# 改进 DH 的坐标系附着在连杆上（而非关节上）：
#
# $${}^{i-1}_{i}\mathbf{T} = \mathbf{R}_x(\alpha_{i-1}) \cdot \mathbf{T}_x(a_{i-1}) \cdot \mathbf{R}_z(\theta_i) \cdot \mathbf{T}_z(d_i)$$
#
# **区别记忆法**：
# - 标准 DH：先绕 Z 转，再沿 Z 平移，再沿 X 平移，再绕 X 转
# - 改进 DH：先绕 X 转，再沿 X 平移，再绕 Z 转，再沿 Z 平移
# - 标准 DH 的角标：$a_i, \alpha_i, d_i, \theta_i$
# - 改进 DH 的角标：$a_{i-1}, \alpha_{i-1}, d_i, \theta_i$

# %% [markdown]
# ## 6. 正运动学方程 ⭐

# %% [markdown]
# ### 6.1 变换链
#
# 末端相对于基座的位姿 = 所有连杆变换的乘积：
#
# $${}^{0}_{n}\mathbf{T}(\mathbf{q}) = {}^{0}_{1}\mathbf{T}(q_1) \cdot {}^{1}_{2}\mathbf{T}(q_2) \cdot \dots \cdot {}^{n-1}_{n}\mathbf{T}(q_n)$$
#
# ### 6.2 FK 的计算步骤
# 1. 建立 DH 参数表
# 2. 对每个关节，将 $\theta_i$（或 $d_i$ 对于移动关节）代入 DH 变换公式
# 3. 按顺序连乘所有变换矩阵
# 4. 提取 $\mathbf{T}_{end}$ 中的旋转矩阵 $\mathbf{R}$ 和位置 $\mathbf{p}$

# %% [markdown]
# ## 7. 手算示例 ⭐

# %% [markdown]
# ### 7.1 2R 平面机械臂
#
# ```
# 连杆 1：长 l₁，关节角 q₁
# 连杆 2：长 l₂，关节角 q₂
# 全部在 XY 平面内运动，Z 轴垂直于纸面
# ```
#
# DH 参数表（标准 DH）：
#
# | 关节 i | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
# |:------:|:-----:|:----------:|:-----:|:----------:|
# | 1 | $l_1$ | 0 | 0 | $q_1$ |
# | 2 | $l_2$ | 0 | 0 | $q_2$ |
#
# 末端位置（FK 结果）：
# $$x = l_1\cos q_1 + l_2\cos(q_1+q_2)$$
# $$y = l_1\sin q_1 + l_2\sin(q_1+q_2)$$
# $$\phi = q_1 + q_2 \quad \text{(末端方向)}$$

# %% [markdown]
# ### 7.2 带 Z 轴偏移的 3R 空间臂（简化 PUMA 型）
#
# ```
# 关节 1: 绕 Z₀ 转 (base rotation)
# 关节 2: 绕 Z₁ 转 (shoulder) — 有连杆长度 a₂
# 关节 3: 绕 Z₂ 转 (elbow) — 有连杆长度 a₃
# ```
#
# DH 参数表（标准 DH）：
#
# | 关节 | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
# |:----:|:-----:|:----------:|:-----:|:----------:|
# | 1 | 0 | π/2 | $d_1$ | $q_1$ |
# | 2 | $a_2$ | 0 | 0 | $q_2$ |
# | 3 | $a_3$ | 0 | 0 | $q_3$ |
#
# 注意关节 1 的 $\alpha_1 = \pi/2$ 表示 Z₀ 到 Z₁ 绕 X₁ 转了 90°。这使肩关节的旋转平面从水平变为垂直。

# %% [markdown]
# ## 8. Python 实现 ⭐

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, FloatSlider
import sys
sys.path.insert(0, '..')
from src.robotics_learning.kinematics import (
    dh_transform, forward_kinematics
)
from src.robotics_learning.transforms import rot_z, rot_x
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 8.1 单连杆 DH 变换

# %%
a, alpha, d, theta = 1.0, np.pi/2, 0.5, np.pi/4
T = dh_transform(a, alpha, d, theta, convention='sdh')
print(f"DH 变换 (a={a}, α={np.degrees(alpha):.0f}°, d={d}, θ={np.degrees(theta):.0f}°):")
print(np.round(T, 4))

# 验证：检查旋转部分的正交性
R = T[:3, :3]
print(f"\nR^T R = I? {np.allclose(R.T @ R, np.eye(3))}")
print(f"det(R) = {np.linalg.det(R):.4f}")

# %% [markdown]
# ### 8.2 2R 平面臂 FK

# %%
# 2R 平面臂 DH 表（标准 DH）
dh_2r = np.array([
    [1.0, 0.0, 0.0, np.pi/4],   # [a, α, d, θ=q₁]
    [0.8, 0.0, 0.0, np.pi/3],   # [a, α, d, θ=q₂]
])

T_end, transforms = forward_kinematics(dh_2r, convention='sdh')
print("末端位姿 T_end:")
print(np.round(T_end, 4))
print(f"\n末端位置: {np.round(T_end[:3, 3], 4)}")

# 验证手算公式
q1, q2 = dh_2r[0, 3], dh_2r[1, 3]
l1, l2 = dh_2r[0, 0], dh_2r[1, 0]
x_manual = l1*np.cos(q1) + l2*np.cos(q1+q2)
y_manual = l1*np.sin(q1) + l2*np.sin(q1+q2)
print(f"手算 x = {x_manual:.4f}, FK x = {T_end[0,3]:.4f}")
print(f"手算 y = {y_manual:.4f}, FK y = {T_end[1,3]:.4f}")
print(f"一致? {np.allclose([x_manual, y_manual], T_end[:2, 3])}")

# %% [markdown]
# ### 8.3 交互式 2R 机械臂

# %%
def plot_2r_interactive(theta1_deg=45.0, theta2_deg=60.0):
    """用 ipywidgets 交互操控 2R 臂"""
    l1, l2 = 1.0, 0.8
    q1, q2 = np.radians(theta1_deg), np.radians(theta2_deg)

    dh = np.array([[l1, 0.0, 0.0, q1], [l2, 0.0, 0.0, q2]])
    T_end, transforms = forward_kinematics(dh)

    # 提取关节点
    points = np.array([T[:3, 3] for T in transforms])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(points[:, 0], points[:, 1], 'b-o', linewidth=3, markersize=8)
    ax.plot(0, 0, 'ks', markersize=10, label='Base')
    ax.plot(points[-1, 0], points[-1, 1], 'r*', markersize=15, label='End-Effector')

    # 工作空间参考圆
    r_max = l1 + l2
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_max*np.cos(theta_c), r_max*np.sin(theta_c), 'k--', alpha=0.2, label='Max reach')

    ax.set_xlim([-2.5, 2.5]); ax.set_ylim([-2.5, 2.5])
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'2R Arm: θ₁={theta1_deg:.0f}°, θ₂={theta2_deg:.0f}°')
    ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

_ = interact(plot_2r_interactive,
             theta1_deg=FloatSlider(min=-180, max=180, step=1, value=45, description='θ₁'),
             theta2_deg=FloatSlider(min=-180, max=180, step=1, value=60, description='θ₂'))

# %% [markdown]
# ### 8.4 3D 空间臂（3R 带 Z 轴偏移）

# %%
# 3R 空间臂 DH 表
l1, l2, l3 = 1.0, 0.8, 0.5
dh_3r = np.array([
    [0.0, np.pi/2, l1,  np.pi/4],    # 关节1: 肩部旋转
    [l2,  0.0,     0.0, np.pi/6],    # 关节2: 大臂
    [l3,  0.0,     0.0, -np.pi/3],   # 关节3: 小臂
])

T_end, transforms = forward_kinematics(dh_3r)
points = np.array([T[:3, 3] for T in transforms])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 画连杆
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-o', linewidth=3, markersize=8)
# 画基座
ax.scatter([0], [0], [0], c='black', s=150, marker='s', label='Base')
# 画末端
ax.scatter(*points[-1], c='red', s=150, marker='*', label='End-Effector')

# 画各坐标系的缩放版
for T in transforms:
    R = T[:3, :3]
    o = T[:3, 3]
    for i, c in enumerate(['r', 'g', 'b']):
        ax.quiver(o[0], o[1], o[2], R[0,i]*0.15, R[1,i]*0.15, R[2,i]*0.15, color=c, alpha=0.6)

ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([0, 3])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('3R Spatial Arm — Forward Kinematics')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/05_3r_spatial_arm.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"末端位置: {np.round(T_end[:3, 3], 4)}")
print(f"末端 ZYX 欧拉角: {np.round(np.degrees(np.array([np.arctan2(T_end[2,1], T_end[2,2]), np.arcsin(-T_end[2,0]), np.arctan2(T_end[1,0], T_end[0,0])])), 1)}°")

# %% [markdown]
# ### 8.5 工作空间采样

# %%
# 对 2R 臂随机采样关节角，绘制工作空间
n_samples = 5000
rng = np.random.RandomState(42)
q_samples = rng.uniform(-np.pi, np.pi, (n_samples, 2))
xy_samples = np.zeros((n_samples, 2))

for i in range(n_samples):
    dh = np.array([[1.0, 0.0, 0.0, q_samples[i, 0]],
                   [0.8, 0.0, 0.0, q_samples[i, 1]]])
    T_end, _ = forward_kinematics(dh)
    xy_samples[i] = T_end[:2, 3]

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(xy_samples[:, 0], xy_samples[:, 1],
                     c=np.linalg.norm(xy_samples, axis=1),
                     s=1, alpha=0.5, cmap='viridis')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('2R Arm Workspace (5000 random samples)')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Distance from base (m)')
plt.tight_layout()
plt.savefig('../outputs/05_workspace_sampling.png', dpi=100, bbox_inches='tight')
plt.show()

print("工作空间呈圆环形，内外径分别为 |l₁ - l₂| 和 l₁ + l₂。")

# %% [markdown]
# ### 8.6 标准 DH vs 改进 DH 对比

# %%
# 同一个 2R 臂用两种约定计算，结果应一致
a1, a2 = 1.0, 0.8

# 标准 DH: [a, α, d, θ]
dh_standard = np.array([
    [a1, 0.0, 0.0, np.pi/4],
    [a2, 0.0, 0.0, np.pi/3]
])

# 改进 DH: [a_{i-1}, α_{i-1}, d_i, θ_i]
dh_modified = np.array([
    [0.0, 0.0, 0.0, np.pi/4],   # 第一个连杆: a₀=0, α₀=0
    [a1,  0.0, 0.0, np.pi/3],   # 第二个连杆: a₁=l₁
])

T_std, _ = forward_kinematics(dh_standard, convention='sdh')
T_mod, _ = forward_kinematics(dh_modified, convention='mdh')

print("标准 DH 末端位置:", np.round(T_std[:3, 3], 4))
print("改进 DH 末端位置:", np.round(T_mod[:3, 3], 4))
print(f"两者一致? {np.allclose(T_std, T_mod, atol=1e-10)}")

# %% [markdown]
# ## 9. 常见错误与易混淆概念
#
# 1. **DH 约定的选择**：不同教材使用不同的 DH 约定。Craig 用的是本文的"标准 DH"（实际上他称之为改进 DH），而 Spong/Vidyasagar 用的是另一种。使用任何 DH 参数表前必须确认约定。
# 2. **第一个和最后一个坐标系**：基座系 {0} 和末端系 {n} 的放置有自由度。DH 参数的 $a_1, d_1$ 和末端工具的变换通常需要额外处理。
# 3. **关节变量是哪个参数**：对于旋转关节，$\theta_i = q_i$；对于移动关节，$d_i = q_i$。两者不可混淆。
# 4. **α 的符号**：$\alpha_i$ 的正负取决于右手定则绕 $X_i$ 轴的旋转方向。写错符号会导致 FK 结果崩溃。

# %% [markdown]
# ## 10. 工程应用
#
# - **工业机器人编程**：示教器中显示的末端位姿由 FK 实时计算
# - **仿真与可视化**：Gazebo、MuJoCo、Isaac Sim 中使用 FK 更新视觉
# - **轨迹规划**：规划器在关节空间计算，但末端约束检查需要用 FK 映射到工作空间
# - **标定（Calibration）**：通过测量实际末端位姿 vs FK 预测来修正 DH 参数

# %% [markdown]
# ## 11. 面试常见问题
#
# 1. **DH 参数是什么？标准 DH 和改进 DH 的区别？** → INTERVIEW_CHECKLIST #2.1
# 2. **正运动学解决什么问题？计算步骤？** → #2.2
# 3. **为什么 4 个参数足以描述任何一个关节变换？** → 两个额外的自由度被坐标系附着规则吸收

# %% [markdown]
# ## 12. 练习题
#
# ### 概念题
# 1. 为什么 4 个 DH 参数足够描述相邻连杆之间的 $4\times4$ 齐次变换？
# 2. 标准 DH 与改进 DH 的变换顺序分别为哪四步？
#
# ### 手算题
# 1. 给定 2R 臂 DH 参数 ($l_1=1, l_2=0.8$)，计算 $q_1=30°$, $q_2=45°$ 时的末端位置。
# 2. 为 3R 平面臂构造 DH 参数表。
#
# ### 编程题
# 1. 实现通用 FK 函数，接受 DH 表 + 约定选择参数，返回末端位姿和中间变换。
# 2. 用 6R 工业机器人（如 KUKA KR6）的 DH 参数表实现 FK 并画图。
#
# > 答案见 `solutions/05_solutions.ipynb`

# %% [markdown]
# ## 13. 本节总结
#
# | 概念 | 公式/含义 |
# |------|-----------|
# | DH 参数 | 4 个参数 $(a_i, \alpha_i, d_i, \theta_i)$ 描述相邻连杆变换 |
# | 标准 DH | ${}^{i-1}_{i}\mathbf{T} = R_z(\theta_i) T_z(d_i) T_x(a_i) R_x(\alpha_i)$ |
# | FK 方程 | ${}^{0}_{n}\mathbf{T} = \prod_{i=1}^n {}^{i-1}_{i}\mathbf{T}(q_i)$ |
# | 2R 臂末端 | $x = l_1c_1 + l_2c_{12}, y = l_1s_1 + l_2s_{12}$ |
# | 工作空间 | 所有 $\mathbf{p}_{end}(\mathbf{q})$ 的集合, $\mathbf{q} \in [q_{min}, q_{max}]$ |

# %% [markdown]
# ## 14. 与下一节的联系
#
# 下一节（NB06）将解决 IK 问题："给定目标末端位姿，求关节角。" 这是一个更困难的问题——**从 6 维（或 3 维）位姿反向求解 n 维关节角**。我们将看到几何法、代数法和数值法三种不同的 IK 策略。
