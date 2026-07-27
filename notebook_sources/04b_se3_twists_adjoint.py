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
# # Notebook 04b：SE(3)、Twist、Wrench 与 Adjoint
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB04 四元数/SO(3) ──→ NB04b SE(3) ──→ NB05b PoE FK
#                            │
#                            ├── twist/wrench 的统一表达
#                            └── Adjoint 变换（力/速度在不同系之间的映射）
# ```
#
# SO(3) 只描述旋转。SE(3) 统一描述旋转+平移——这是机器人学中刚体运动的完整数学框架。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 掌握 hat (∧) 和 vee (∨) 操作符
# - ⭐ 理解 twist $\mathcal{V} = [\boldsymbol{\omega}; \mathbf{v}]$ 的物理含义
# - ⭐ 区分 space twist 和 body twist
# - ⭐ 掌握 SE(3) 指数映射和 twist 的物理意义
# - ⭐ 掌握 Adjoint 变换
# - ⭐ 理解 wrench 的对偶变换
# - 📖 区分 left/right perturbation

# %% [markdown]
# ## 3. 约定

# %% [markdown]
# 本课程使用 **Lynch & Park (Modern Robotics)** 约定：
# - twist 排列：$\mathcal{V} = [\boldsymbol{\omega}; \mathbf{v}]$（角速度在前，线速度在后，Modern Robotics 约定）
# - hat 操作符：$\boldsymbol{\xi}^\wedge = \begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{bmatrix} \in \mathfrak{se}(3)$
# - 空间 twist $\mathcal{V}_s$ 和物体 twist $\mathcal{V}_b$ 的关系：$\mathcal{V}_s = \text{Ad}_{T} \mathcal{V}_b$

# %% [markdown]
# ## 4. Hat 和 Vee ⭐

# %% [markdown]
# ### 4.1 $\mathfrak{so}(3)$ 上的 hat/vee
#
# $$\boldsymbol{\omega}^\wedge = [\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix} \in \mathfrak{so}(3)$$
# $$([\boldsymbol{\omega}]_\times)^\vee = \boldsymbol{\omega}$$

# %% [markdown]
# ### 4.2 $\mathfrak{se}(3)$ 上的 hat/vee
#
# $$\boldsymbol{\xi}^\wedge = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}^\wedge = \begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{bmatrix} \in \mathfrak{se}(3)$$
# $$\left(\begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{bmatrix}\right)^\vee = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}$$

# %% [markdown]
# ## 5. Twist（旋量）⭐

# %% [markdown]
# ### 5.1 空间 Twist（Spatial Twist）
#
# 刚体运动可视为绕空间系中某个螺旋轴 $\mathcal{S}$ 的旋转+平移。瞬时速度用 twist 描述：
# $$\mathcal{V}_s = \begin{bmatrix} \boldsymbol{\omega}_s \\ \mathbf{v}_s \end{bmatrix} \in \mathbb{R}^6$$
#
# $\mathbf{v}_s$ 不是物体上某点的线速度，而是**假设物体无限延伸、在空间系原点处**的线速度。
#
# 空间 twist 与 SE(3) 的关系：
# $$\dot{T} = \mathcal{V}_s^\wedge T$$

# %% [markdown]
# ### 5.2 物体 Twist（Body Twist）
#
# 物体 twist 是速度在**物体自身参考系**中的表达：
# $$\mathcal{V}_b = \begin{bmatrix} \boldsymbol{\omega}_b \\ \mathbf{v}_b \end{bmatrix} = \text{Ad}_{T^{-1}} \mathcal{V}_s$$
#
# 物体 twist 与 SE(3) 的关系：
# $$\dot{T} = T \mathcal{V}_b^\wedge$$

# %% [markdown]
# ### 5.3 从 Twist 恢复 SE(3) 的指数映射
#
# 给定空间 twist $\mathcal{V}_s = [\boldsymbol{\omega}_s; \mathbf{v}_s]$，在时间 $\Delta t$ 后的位姿变化：
# $$T(\Delta t) = \exp(\mathcal{V}_s^\wedge \Delta t) T(0)$$
#
# 其中 SE(3) 指数映射已经实现在 `se3_exp()` 中（见 NB04）。

# %% [markdown]
# ## 6. Adjoint 变换 ⭐

# %% [markdown]
# ### 6.1 Adjoint 矩阵
#
# 给定 $T = (R, \mathbf{p}) \in SE(3)$，Adjoint 矩阵 $\text{Ad}_T \in \mathbb{R}^{6\times 6}$ 将物体系中的 twist 变换到空间系：
# $$\mathcal{V}_s = \text{Ad}_T \mathcal{V}_b$$
#
# $$\text{Ad}_T = \begin{bmatrix} R & 0 \\ [\mathbf{p}]_\times R & R \end{bmatrix}$$

# %% [markdown]
# ### 6.2 Wrench 的对偶变换
#
# Wrench $\mathcal{F} = [\mathbf{n}; \mathbf{f}]$（力+力矩）按照 Adjoint 的**转置逆**变换：
# $$\mathcal{F}_s = \text{Ad}_T^{-T} \mathcal{F}_b = \begin{bmatrix} R & [\mathbf{p}]_\times R \\ 0 & R \end{bmatrix} \begin{bmatrix} \mathbf{n}_b \\ \mathbf{f}_b \end{bmatrix}$$
#
# $\boldsymbol{\tau} = \mathbf{J}^T \mathbf{F}$ 来自虚功原理（$\boldsymbol{\tau}^T\dot{\mathbf{q}} = \mathbf{F}^T\mathbf{V}$），
# 是速度映射 $\mathbf{V} = \mathbf{J}\dot{\mathbf{q}}$ 的对偶关系。Wrench 更换参考系时还需要 Adjoint 的逆转置。

# %% [markdown]
# ## 7. Spatial Jacobian vs Body Jacobian

# %% [markdown]
# 空间雅可比 $\mathbf{J}_s(\mathbf{q})$ 和物体雅可比 $\mathbf{J}_b(\mathbf{q})$ 的关系：
# $$\mathbf{J}_s(\mathbf{q}) = \text{Ad}_{T_{sb}(\mathbf{q})} \mathbf{J}_b(\mathbf{q})$$
#
# - $\mathbf{J}_s$ 的每一列是关节 $i$ 的 twist 在**空间系**中的表达
# - $\mathbf{J}_b$ 的每一列是关节 $i$ 的 twist 在**物体系**中的表达
# - NB07 的 `compute_geometric_jacobian` 计算的是**经典几何雅可比**（末端点速度+角速度）
# - Lie 群空间雅可比使用 `compute_space_jacobian_poe`，两者概念不同

# %% [markdown]
# ## 8. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
from src.robotics_learning.transforms import (
    skew, se3_exp, so3_exp, axis_angle_to_rot, rot_z,
    homogeneous_transform, adjoint, adjoint_inv_transpose, se3_log
)
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %% [markdown]
# ### 8.1 Hat/Vee 和 SE(3) 工具

# %%
def se3_hat(twist):
    """ξ = [ω; v] → ξ^∧ ∈ se(3) (Modern Robotics convention)"""
    omega, v = twist[:3], twist[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = skew(omega)
    Xi[:3, 3] = v
    return Xi

def se3_vee(Xi):
    """ξ^∧ ∈ se(3) → ξ = [ω; v]"""
    omega = np.array([Xi[2,1], Xi[0,2], Xi[1,0]])
    v = Xi[:3, 3]
    return np.concatenate([omega, v])

# 测试 Adjoint 往返
R_test = axis_angle_to_rot(rng.randn(3), 1.0)
T_test = homogeneous_transform(R_test, rng.uniform(-1, 1, 3))
Ad = adjoint(T_test)
Ad_inv = adjoint(homogeneous_transform(R_test.T, -R_test.T @ T_test[:3, 3]))
assert np.allclose(Ad @ Ad_inv, np.eye(6), atol=1e-10)
print("✅ Adjoint 往返测试通过")

# %% [markdown]
# ### 8.2 Twist 变换演示

# %%
# 2R 臂在 q1=30°, q2=45° — 经典几何雅可比 vs Lie 群空间雅可比
from src.robotics_learning.kinematics import (
    compute_geometric_jacobian, compute_space_jacobian_poe, forward_kinematics
)
dh = np.array([[1.0, 0, 0], [0.8, 0, 0]])
q = np.array([np.pi/6, np.pi/4])

# 经典几何雅可比 (末端点速度 + 角速度): J_geom = [J_v; J_ω]
J_geom = compute_geometric_jacobian(dh, q)

# Lie 群空间雅可比 (PoE, [ω; v] twist): 需要螺旋轴
# 2R 臂螺旋轴 [ω; v] (Modern Robotics)
l1 = 1.0
S1 = np.array([0, 0, 1, 0, 0, 0])          # 基座关节: ω=[0,0,1], v=[0,0,0]
S2 = np.array([0, 0, 1, 0, -l1, 0])         # 肘关节: ω=[0,0,1], v=-ω×p=[0,-l1,0]
J_s = compute_space_jacobian_poe([S1, S2], q)

# 末端位姿
T_end, _ = forward_kinematics(np.column_stack([dh, q]))
Ad_end = adjoint(T_end)

# 物体雅可比 J_b = Ad_{T^{-1}} J_s
J_b = np.linalg.solve(Ad_end, J_s)

# 验证: Ad_T · V_b = V_s
q_dot = np.array([1.0, -0.5])
V_s = J_s @ q_dot
V_b = J_b @ q_dot
V_s_from_adj = Ad_end @ V_b
print(f"Space Jacobian V_s:\n{np.round(V_s[:3], 4)}  ← [ω; v]")
print(f"Ad_T · V_b:\n{np.round(V_s_from_adj[:3], 4)}")
print(f"Adjoint一致? {np.allclose(V_s, V_s_from_adj, atol=1e-10)}")

# 对比几何雅可比: 上半部分是末端点速度 ṗ_E
V_geom = J_geom @ q_dot
print(f"\n几何雅可比 [ṗ_E; ω]:\n{np.round(V_geom, 4)}")
print(f"ṗ_E ≠ v_s (v_s = v at spatial origin, ṗ_E = v_s + ω×p_E)")

# %% [markdown]
# ### 8.3 Wrench 变换验证

# %%
# 末端受 wrench F_b = [nx, ny, nz, fx, fy, fz] (在 body 系，[n;f] 排列)
F_b = np.array([10.0, -5.0, 0.0, 0.0, 0.0, 2.0])
# 转换到空间系
Ad_invT = adjoint_inv_transpose(T_end)
F_s = Ad_invT @ F_b

# 关节力矩: τ = J_s^T F_s = J_b^T F_b
tau_from_space = J_s.T @ F_s
tau_from_body = J_b.T @ F_b
print(f"τ (from space J^T F_s): {np.round(tau_from_space, 4)}")
print(f"τ (from body J^T F_b):   {np.round(tau_from_body, 4)}")
print(f"一致? {np.allclose(tau_from_space, tau_from_body, atol=1e-10)}")

# %% [markdown]
# ### 8.4 左扰动 vs 右扰动

# %%
# 左扰动: T_new = exp(δξ^∧) · T   (在空间系施加扰动)
# 右扰动: T_new = T · exp(δξ^∧)   (在物体系施加扰动)
delta = np.array([0, 0, 0.05, 0.01, 0, 0])  # 小扰动 [ω; v]
Xi = se3_hat(delta)

T_original = T_end.copy()
T_left = np.real(T_original.copy())
# 简化: T_left ≈ (I + Xi) @ T (一级近似)
T_left = (np.eye(4) + Xi) @ T_left
T_right = T_original @ (np.eye(4) + Xi)

# 检查两种扰动的区别
print(f"左扰动后位置: {np.round(T_left[:3,3], 4)}")
print(f"右扰动后位置: {np.round(T_right[:3,3], 4)}")
print(f"原始位置:      {np.round(T_original[:3,3], 4)}")
print("左扰动：δ 在空间系表达 → 影响物体系的全局位置。")
print("右扰动：δ 在物体系表达 → 影响物体相对于自身的姿态。")

# %% [markdown]
# ## 9. 常见错误
#
# 1. **twist 排列不一致**：本课程统一使用 $[\boldsymbol{\omega}; \mathbf{v}]$（角速度在前，Lynch & Park 约定）。不同教材可能使用相反的排列，混用会导致 Adjoint 和 wrench 变换错误。
# 2. **空间 twist vs 物体 twist**：$\mathcal{V}_s$ 中 $\mathbf{v}_s$ 不是物体上某点的实际线速度。真实末端线速度 $\dot{\mathbf{p}}_E = \mathbf{v}_s + \boldsymbol{\omega}_s \times \mathbf{p}_E$。
# 3. **Adjoint vs 坐标变换**：$\text{Ad}_T$ 将 twist/wrench 在坐标系间映射，不是简单的 $\mathbb{R}^6$ 旋转。

# %% [markdown]
# ## 10. 练习题
#
# ### 概念题
# 1. $\mathcal{V}_s$ 和 $\mathcal{V}_b$ 中的线速度分量 $\mathbf{v}_s$ 和 $\mathbf{v}_b$ 分别表示什么意思？
# 2. 为什么 wrench 的变换是 $\text{Ad}_T^{-T}$ 而非 $\text{Ad}_T$？
#
# ### 手算题
# 1. 给定 $T = (R_z(30°), \mathbf{p}=[1,2,0]^T)$，计算 $\text{Ad}_T$。
# 2. 验证 $\text{Ad}_{T_1 T_2} = \text{Ad}_{T_1} \text{Ad}_{T_2}$。
#
# ### 编程题
# 1. 验证 $\mathcal{V}_b^T \mathcal{F}_b = \mathcal{V}_s^T \mathcal{F}_s$（功率不变性）。
# 2. 实现 SE(3) 的 exp 映射并用数值差分验证 $\dot{T} = \mathcal{V}_s^\wedge T$。
#
# > 答案见 `solutions/solutions_week2.ipynb`
