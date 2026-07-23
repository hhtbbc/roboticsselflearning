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
# # Notebook 03：旋转的多种表示 — 欧拉角、固定角与轴角
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB02 齐次变换/SO(3) ──→ NB03 旋转表示 ──→ NB04 四元数与李群
#         (旋转矩阵)        (欧拉角/轴角)       (无奇异性表示)
#                                │
#                                └──→ NB06 IK 中处理姿态
#                                └──→ NB07 解析雅可比（需要最小表示）
# ```
#
# 旋转矩阵 $\mathbf{R}$ 有 9 个元素但只有 3 个自由度。使用旋转矩阵来表示旋转在计算上冗余，在优化中不方便。本节介绍两种**最小表示**（各 3 参数）以及一种**旋量表示**。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解欧拉角的多种约定（ZYX, ZYZ, XYZ 固定角）
# - ⭐ 理解万向锁（Gimbal Lock）的数学与几何本质
# - ⭐ 掌握轴角表示与罗德里格斯公式（Rodrigues' Formula）
# - ⭐ 实现所有表示之间的相互转换
# - 📖 理解固定角与欧拉角的区别
# - 📚 了解旋转矩阵微分与角速度的关系

# %% [markdown]
# ## 3. 旋转表示比较
#
# | 表示 | 参数数 | 奇异性 | 插值 | 组合 | 用途 |
# |------|:------:|:------:|:----:|:----:|------|
# | 旋转矩阵 SO(3) | 9 | 无 | 困难 | 矩阵乘 | 数学严谨，计算用 |
# | 欧拉角 | 3 | 万向锁 | 可 | 困难 | 人机界面、姿态显示 |
# | 轴角 | 4(3) | 180° | 可 | 中等 | 物理旋转、伺服控制 |
# | 四元数 | 4 | 无 | slerp | 乘法 | 姿态估计、动画、SLAM |

# %% [markdown]
# ## 4. 欧拉角（Euler Angles）⭐

# %% [markdown]
# ### 4.1 定义与约定
#
# 欧拉角将任意旋转分解为绕三个不同轴的**三次连续旋转**。根据旋转轴的选择和旋转方式，存在多种约定：
#
# **Proper Euler Angles**（第一轴和第三轴相同）：
# - **ZYZ**：$\mathbf{R} = \mathbf{R}_z(\phi) \mathbf{R}_y(\theta) \mathbf{R}_z(\psi)$
# - ZXZ, XYX 等
#
# **Tait-Bryan Angles**（三轴各不相同，也称 Cardan Angles）：
# - **ZYX**（最常见）：$\mathbf{R} = \mathbf{R}_z(\text{yaw}) \mathbf{R}_y(\text{pitch}) \mathbf{R}_x(\text{roll})$
# - XYZ（固定角）

# %% [markdown]
# ### 4.2 ZYX 欧拉角（Tait-Bryan / Yaw-Pitch-Roll）
#
# 定义：先绕 Z 转 yaw（偏航），再绕（新的）Y 转 pitch（俯仰），最后绕（新的）X 转 roll（滚转）。
#
# 这是机器人学中最常见的"RPY 角"（Roll-Pitch-Yaw，在固定轴约定下）。
#
# $$\mathbf{R}_{ZYX} = \mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$$
#
# 展开后：
#
# $$\mathbf{R}_{ZYX} = \begin{bmatrix}
# c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi \\
# s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi \\
# -s_\theta & c_\theta s_\phi & c_\theta c_\phi
# \end{bmatrix}$$
#
# 其中 $c_\psi = \cos\psi$, $s_\psi = \sin\psi$。

# %% [markdown]
# ### 4.3 ZYX 逆解（从旋转矩阵提取欧拉角）
#
# 给定 $\mathbf{R} = [r_{ij}]$，求解 $(\phi, \theta, \psi)$ = (roll, pitch, yaw)：
#
# $$\theta = \text{atan2}\left(-r_{31}, \sqrt{r_{11}^2 + r_{21}^2}\right)$$
#
# - 若 $\cos\theta \neq 0$：
#   $$\phi = \text{atan2}(r_{32}, r_{33})$$
#   $$\psi = \text{atan2}(r_{21}, r_{11})$$
#
# - 若 $\cos\theta \approx 0$（即 $\theta = \pm 90^\circ$）→ **万向锁**：
#   此时 $r_{11}=r_{21}=r_{32}=r_{33}=0$，有无穷多组 $(\phi, \psi)$ 产生同样的旋转。只能确定 $\phi \pm \psi$ 的值。

# %% [markdown]
# ### 4.4 万向锁（Gimbal Lock）⭐ 面试高频
#
# **定义**：当第二次旋转（pitch）为 ±90° 时，第一和第三旋转轴重合，系统失去一个旋转自由度。
#
# **数学根源**：欧拉角的参数化在 $\theta = \pm\pi/2$ 处有一个奇异性——映射 $\mathbb{R}^3 \to SO(3)$ 的雅可比矩阵秩降为 2。这不是物理上丢失了自由度，而是**参数化本身的缺陷**。
#
# **几何直观**：
# ```
# 初始: Z-up → Y 转 90° → X 与 Z 共线
#                          此时再绕 X 转和再绕 Z 转产生相同的效果
# ```
#
# **解决方案**：
# 1. 避开 ±90°（机械限位）
# 2. 使用四元数（NB04）——无奇异性
# 3. 使用旋转矩阵 + 重新参数化

# %% [markdown]
# ### 4.5 固定角（Fixed Angles）vs 欧拉角（Euler Angles）
#
# | | 固定角 (Fixed) | 欧拉角 (Euler) |
# |---|---|---|
# | 旋转轴 | 相对于**固定**参考系的轴 | 相对于**运动体自身**的轴 |
# | 乘法顺序 | 左乘（从右到左 = 从左到右施加） | 右乘（从左到右） |
# | XYZ 固定角 | $\mathbf{R}_z(\gamma)\mathbf{R}_y(\beta)\mathbf{R}_x(\alpha)$ | — |
# | ZYX 欧拉角 | — | $\mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$ |
# | 关系 | XYZ 固定角 $(\alpha,\beta,\gamma)$ = ZYX 欧拉角 $(\phi,\theta,\psi)$ | — |
#
# **关键结论**：XYZ 固定角与 ZYX 欧拉角产生**相同的旋转矩阵**！旋转的顺序刚好相反。

# %% [markdown]
# ## 5. 轴角表示（Axis-Angle）⭐

# %% [markdown]
# ### 5.1 定义
#
# 根据欧拉旋转定理（Euler's Rotation Theorem）：**任何刚体的旋转都可以用绕某个固定轴 $\mathbf{k}$ 旋转角度 $\theta$ 来描述。**
#
# 轴角表示：$(\mathbf{k}, \theta)$，其中 $\|\mathbf{k}\| = 1$，$\theta \in [0, \pi]$。
#
# 旋转向量（Rotation Vector）：$\boldsymbol{\omega} = \theta\mathbf{k} \in \mathbb{R}^3$，方向 = 旋转轴，大小 = 旋转角。
# 旋转向量只有 3 个参数，但 $\theta = \pi$ 时存在奇异性（$\mathbf{k}$ 和 $-\mathbf{k}$ 对应同一旋转）。

# %% [markdown]
# ### 5.2 罗德里格斯公式（Rodrigues' Formula）— 推导
#
# 绕单位轴 $\mathbf{k}$ 旋转 $\theta$ 的旋转矩阵：
#
# $$\mathbf{R}(\mathbf{k}, \theta) = \mathbf{I} + \sin\theta [\mathbf{k}]_\times + (1 - \cos\theta) [\mathbf{k}]_\times^2$$
#
# **推导思路**（关键步骤）：
# 1. 将任意向量 $\mathbf{v}$ 分解为平行于 $\mathbf{k}$ 的 $\mathbf{v}_\parallel$ 和垂直于 $\mathbf{k}$ 的 $\mathbf{v}_\perp$
# 2. 旋转后：$\mathbf{v}_\parallel$ 不变，$\mathbf{v}_\perp$ 在垂直于 $\mathbf{k}$ 的平面内旋转 $\theta$
# 3. $\mathbf{v}_{rot} = \mathbf{v}_\parallel + \cos\theta \cdot \mathbf{v}_\perp + \sin\theta \cdot (\mathbf{k} \times \mathbf{v}_\perp)$
# 4. 整理为矩阵形式即得上式

# %% [markdown]
# ### 5.3 逆映射（旋转矩阵 → 轴角）
#
# 从 $\mathbf{R} = [r_{ij}]$ 恢复 $(\mathbf{k}, \theta)$：
#
# $$\theta = \arccos\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right)$$
#
# 当 $\sin\theta \neq 0$：
# $$\mathbf{k} = \frac{1}{2\sin\theta}\begin{bmatrix} r_{32} - r_{23} \\ r_{13} - r_{31} \\ r_{21} - r_{12} \end{bmatrix}$$
#
# 当 $\theta = \pi$（180° 旋转）：需要使用特判，$\mathbf{k}$ 从 $\mathbf{R} + \mathbf{I}$ 的列中提取。

# %% [markdown]
# ### 5.4 旋转向量与角速度的区别
#
# - **旋转向量** $\boldsymbol{\omega} = \theta\mathbf{k}$：表示从参考位姿到当前位姿的**有限旋转**
# - **角速度** $\boldsymbol{\omega} = \dot{\theta}\mathbf{k}$：表示旋转的**瞬时速率**
#
# 二者不是同一个东西！$\boldsymbol{\omega}_{rot\_vec}$ 和 $\boldsymbol{\omega}_{ang\_vel}$ 的关系涉及 SO(3) 的指数映射（NB04）。

# %% [markdown]
# ## 6. Python 实现与可视化

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '..')
from src.robotics_learning.transforms import (
    rot_x, rot_y, rot_z,
    euler_zyx_to_rot, rot_to_euler_zyx, euler_zyz_to_rot,
    axis_angle_to_rot, rot_to_axis_angle, skew
)
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 6.1 欧拉角 → 旋转矩阵 → 欧拉角（往返验证）

# %%
# 原始欧拉角
roll, pitch, yaw = np.radians([30, -45, 60])
print(f"Original: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")

# 转为旋转矩阵
R = euler_zyx_to_rot(roll, pitch, yaw)
print(f"\nR_ZYX:\n{np.round(R, 4)}")

# 再转回欧拉角
r, p, y = rot_to_euler_zyx(R)
print(f"\nRecovered: roll={np.degrees(r):.1f}°, pitch={np.degrees(p):.1f}°, yaw={np.degrees(y):.1f}°")
print(f"往返一致? {np.allclose([roll, pitch, yaw], [r, p, y])}")

# %% [markdown]
# ### 6.2 万向锁演示

# %%
# 设置 pitch = 90°（万向锁条件）
pitch = np.pi / 2
roll_vals = np.linspace(0, np.pi, 5)

print("pitch = 90° 时，不同 (roll, yaw) 对产生相同旋转：")
for roll in roll_vals:
    for yaw in [0.0, np.pi/4]:
        R = euler_zyx_to_rot(roll, pitch, yaw)
        # 检查：R 的第三行是否相同（因为 pitch=90° 时 roll 和 yaw 耦合）
        print(f"  (r={np.degrees(roll):5.1f}°, y={np.degrees(yaw):5.1f}°) → R[2,:] = {np.round(R[2,:], 3)}")

print("\n注意：无论 roll 和 yaw 怎么变，R[2,:]（第 3 行）几乎不变！")
print("这就是万向锁——两个自由度产生相同的旋转效果。")

# %% [markdown]
# ### 6.3 轴角 ↔ 旋转矩阵

# %%
# 随机旋转轴和角度
axis = np.array([1.0, 0.5, 0.2])
axis = axis / np.linalg.norm(axis)
angle = np.pi / 3  # 60°

# 轴角 → 旋转矩阵（罗德里格斯公式）
R = axis_angle_to_rot(axis, angle)
print(f"Axis: {axis}, Angle: {np.degrees(angle):.1f}°")
print(f"R (Rodrigues):\n{np.round(R, 4)}")

# 旋转矩阵 → 轴角
k_recovered, theta_recovered = rot_to_axis_angle(R)
print(f"\nRecovered: axis = {np.round(k_recovered, 4)}, angle = {np.degrees(theta_recovered):.1f}°")
print(f"往返一致? {np.allclose(axis, k_recovered) and np.allclose(angle, theta_recovered)}")

# 验证罗德里格斯公式的等价性
K = skew(axis)
R_manual = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
print(f"\n手动公式 = 库函数? {np.allclose(R, R_manual, atol=1e-10)}")

# %% [markdown]
# ### 6.4 旋转表示全景对比图

# %%
# 绕 Z 轴旋转 0 到 2π，比较不同表示
angles = np.linspace(0, 2*np.pi, 100)
r11_vals = []
axis_z_vals = []
euler_yaw_vals = []

for theta in angles:
    R = rot_z(theta)
    r11_vals.append(R[0, 0])            # 旋转矩阵的一个元素
    _, _, yaw = rot_to_euler_zyx(R)     # 欧拉角 yaw
    euler_yaw_vals.append(yaw)
    _, theta_aa = rot_to_axis_angle(R)  # 轴角
    axis_z_vals.append(theta_aa)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(np.degrees(angles), r11_vals, 'b-', linewidth=2)
axes[0].set_title('旋转矩阵: $r_{11} = \cos\\theta$')
axes[0].set_xlabel('$\\theta$ (°)'); axes[0].set_ylabel('$r_{11}$')
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.degrees(angles), np.degrees(euler_yaw_vals), 'g-', linewidth=2)
axes[1].set_title('ZYX 欧拉角: yaw')
axes[1].set_xlabel('$\\theta$ (°)'); axes[1].set_ylabel('yaw (°)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(np.degrees(angles), np.degrees(axis_z_vals), 'r-', linewidth=2)
axes[2].set_title('轴角: 旋转角')
axes[2].set_xlabel('$\\theta$ (°)'); axes[2].set_ylabel('角度 (°)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/03_representation_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 6.5 ZYZ 欧拉角

# %%
phi, theta, psi = np.radians([30, 60, 45])
R_zyz = euler_zyz_to_rot(phi, theta, psi)
print(f"ZYZ Euler ({np.degrees(phi):.0f}°, {np.degrees(theta):.0f}°, {np.degrees(psi):.0f}°):")
print(np.round(R_zyz, 4))

# 对比 ZYX 的结果
R_zyx = euler_zyx_to_rot(np.radians(30), np.radians(60), np.radians(45))
print(f"\nZYX Euler 不同的结果!")
print(f"ZYZ ≠ ZYX? {not np.allclose(R_zyz, R_zyx)}")

# %% [markdown]
# ## 7. 常见错误与易混淆概念
#
# 1. **欧拉角约定不明确**："欧拉角"本身不是一个唯一确定的表示。必须明确：用什么轴？什么顺序？固定轴还是运动轴？面试中必须追问约定再回答。
# 2. **万向锁不是物理锁**：不是机械结构的物理限制——纯数学参数化的奇异性。任何 3 参数表示都有奇异性（这是拓扑学结论：SO(3) 不是 $\mathbb{R}^3$ 的覆盖空间）。
# 3. **atan vs atan2**：角度提取必须用 atan2（四象限反正切），用 atan 会丢失象限信息。
# 4. **角度单位**：三角函数的输入输出都是**弧度**（rad）。显示给人类时转换到度（deg）。

# %% [markdown]
# ## 8. 工程应用
#
# - **欧拉角**：人机界面（示教器）、飞控（roll/pitch/yaw）、ROS 中的 RPY
# - **轴角**：伺服电机控制（给出旋转轴和角增量）、物理仿真中的有限旋转
# - **旋转矩阵**：计算（组合方便，无奇异性）
# - **四元数**：姿态估计（NB23）、SLAM、动画（slerp）

# %% [markdown]
# ## 9. 面试常见问题
#
# 1. **万向锁是什么？四元数为什么没有？** → INTERVIEW_CHECKLIST #1.3
# 2. **欧拉角和四元数的优缺点？** → #1.2
# 3. **给一个轴和角度，写出旋转矩阵。** → #1.6

# %% [markdown]
# ## 10. 练习题
#
# ### 概念题
# 1. XYZ 固定角 $(\alpha,\beta,\gamma)$ 和 ZYX 欧拉角 $(\phi,\theta,\psi)$ 为什么产生相同的旋转矩阵？
# 2. 罗德里格斯公式中 $\sin\theta$ 和 $(1-\cos\theta)$ 的几何含义是什么？
#
# ### 手算题
# 1. 绕轴 $\mathbf{k}=[1,0,0]^T$ 旋转 90°，求旋转矩阵。
# 2. 给定欧拉角 ZYX (30°, -45°, 60°)，手算 R_31 的值。
#
# ### 编程题
# 1. 实现欧拉角所有 12 种约定的旋转矩阵构造。
# 2. 绘制万向锁时旋转向量空间的奇异性曲面。
#
# > 答案见 `solutions/03_solutions.ipynb`

# %% [markdown]
# ## 11. 本节总结
#
# | 概念 | 公式 | 参数数 | 奇异性 |
# |------|------|:------:|:------:|
# | 旋转矩阵 | $\mathbf{R} \in SO(3)$ | 9(3DOF) | 无 |
# | ZYX 欧拉角 | $\mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$ | 3 | pitch=±90° |
# | ZYZ 欧拉角 | $\mathbf{R}_z(\phi)\mathbf{R}_y(\theta)\mathbf{R}_z(\psi)$ | 3 | θ=0°,180° |
# | 轴角 | $(\mathbf{k}, \theta)$ | 4(3DOF) | θ=π |
# | 罗德里格斯 | $\mathbf{R} = \mathbf{I} + s_\theta[\mathbf{k}]_\times + (1-c_\theta)[\mathbf{k}]_\times^2$ | — | — |

# %% [markdown]
# ## 12. 与下一节的联系
#
# 下一节（NB04）将介绍**四元数**——一种无奇异性的 4 参数旋转表示，以及**SO(3) 和 SE(3) 的李群结构**。四元数将轴角 $(\mathbf{k}, \theta)$ 改写为 $(\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\mathbf{k})$，从而消除 θ=π 的奇异性。而 SO(3) 的指数映射（$\exp: \mathfrak{so}(3) \to SO(3)$）本质上就是罗德里格斯公式的几何化重述。
