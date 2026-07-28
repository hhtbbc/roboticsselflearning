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
# # Notebook 05b：Product of Exponentials (PoE) 正运动学
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB04b SE(3)/twist ──→ NB05b PoE FK ──→ NB05 DH FK（对比）
# ```
#
# PoE（Product of Exponentials）是 Lynch & Park (Modern Robotics) 的核心方法。与 DH 参数相比，PoE 不需要逐连杆附着坐标系——只需定义各关节在**零位构型**下的螺旋轴。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解螺旋轴（Screw Axis）的定义
# - ⭐ 掌握 PoE 正运动学：空间系 $T = e^{[S_1]q_1}\dots e^{[S_n]q_n}M$，物体系 $T = M e^{[B_1]q_1}\dots e^{[B_n]q_n}$
# - ⭐ 理解"零位构型"(home configuration) 和末端变换 M
# - ⭐ 能够为简单机械臂建立 PoE 模型
# - 📖 PoE vs DH 的优缺点比较

# %% [markdown]
# ## 3. PoE 正运动学公式 ⭐

# %% [markdown]
# ### 3.1 空间系 PoE
#
# $$T_{sb}(\mathbf{q}) = e^{[\mathcal{S}_1]q_1} e^{[\mathcal{S}_2]q_2} \dots e^{[\mathcal{S}_n]q_n} M$$
#
# - $M \in SE(3)$：零位时末端在空间系中的位姿
# - $\mathcal{S}_i \in \mathbb{R}^6$：关节 $i$ 的螺旋轴（在空间系中表达）
# - $e^{[\mathcal{S}_i]q_i}$：沿螺旋轴的运动（SE(3) 指数映射）
# - 注意：指数项**从基座到末端**依次左乘

# %% [markdown]
# ### 3.2 身体系 PoE
#
# $$T_{sb}(\mathbf{q}) = M e^{[\mathcal{B}_1]q_1} e^{[\mathcal{B}_2]q_2} \dots e^{[\mathcal{B}_n]q_n}$$
#
# - $\mathcal{B}_i = \text{Ad}_{M^{-1}} \mathcal{S}_i$：关节 $i$ 的螺旋轴（在末端系中表达）

# %% [markdown]
# ### 3.3 螺旋轴的定义
#
# 对于旋转关节（旋转轴 $\boldsymbol{\omega}$，轴上一点 $\mathbf{q}$）：
# $$\mathcal{S} = \begin{bmatrix} \boldsymbol{\omega} \\ -\boldsymbol{\omega} \times \mathbf{q} \end{bmatrix} \quad \text{或} \quad \mathcal{S} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{q} \times \boldsymbol{\omega} \end{bmatrix}$$
#
# 本课程使用 Lynch & Park 约定：$\mathcal{S} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}$，其中 $\mathbf{v} = -\boldsymbol{\omega} \times \mathbf{q}$。

# %% [markdown]
# ## 4. PoE vs DH

# %% [markdown]
# | | PoE | DH |
# |---|---|---|
# | 参数数 | 螺旋轴各 6 分量(含几何约束) | 每关节 4 个 |
# | 坐标系 | 只需空间系和末端系 | 每个关节一块坐标系 |
# | 表达退化 | 避免部分坐标系退化问题 | 平行轴时参数选择不直观 |
# | 机械臂奇异性 | 仍然存在(rank(J)<m) | 仍然存在 |
# | 导数 | ∂T/∂q_i 有简洁闭式 | 需通过 DH 变换链求导 |
# | 工业使用 | 学术界和现代框架中增长 | 传统工业机器人广泛使用 |

# %% [markdown]
# ## 5. 2R 臂 PoE 示例

# %% [markdown]
# 2R 平面臂，零位时两连杆沿 X 轴伸展：
# $$M = \begin{bmatrix} 1 & 0 & 0 & l_1+l_2 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
#
# 螺旋轴（绕 Z 轴旋转，旋转轴上的点分别为原点和 $(l_1,0,0)$）：
# $$\mathcal{S}_1 = [0,0,1, 0,0,0]^T, \quad \mathcal{S}_2 = [0,0,1, 0,-l_1,0]^T$$

# %% [markdown]
# ## 6. Python 实现

# %%
import numpy as np
import sys; sys.path.insert(0, '..')
from src.robotics_learning.transforms import skew, se3_exp, homogeneous_transform, rot_z, adjoint
from src.robotics_learning.kinematics import forward_kinematics, compute_space_jacobian_poe
%matplotlib inline
print("✅ 导入完成")

# %%
def se3_hat(twist):
    """ξ = [ω; v] → ξ^∧ ∈ se(3)"""
    omega, v = twist[:3], twist[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = skew(omega); Xi[:3, 3] = v
    return Xi

def poe_fk_space(screw_axes, M, q):
    """空间系 PoE: T = exp([S1]q1)···exp([Sn]qn) M"""
    T = np.eye(4)
    for Si, qi in zip(screw_axes, q):
        T = T @ se3_exp(Si * qi)
    return T @ M

def poe_fk_body(screw_axes_body, M, q):
    """物体系 PoE: T = M exp([B1]q1)···exp([Bn]qn)"""
    T = M.copy()
    for Bi, qi in zip(screw_axes_body, q):
        T = T @ se3_exp(Bi * qi)
    return T

# %% [markdown]
# ### 6.1 2R 臂 PoE vs DH 对比

# %%
l1, l2 = 1.0, 0.8
M_2r = homogeneous_transform(np.eye(3), np.array([l1+l2, 0, 0]))

# 螺旋轴 [ω; v]（Lynch & Park 约定）
S1 = np.array([0, 0, 1, 0, 0, 0])          # [ω; v] = 绕Z旋转, 轴上点=(0,0,0)
S2 = np.array([0, 0, 1, 0, -l1, 0])         # [ω; v] = 绕Z旋转, 轴上点=(l1,0,0)

q_test = np.array([np.pi/4, np.pi/3])

T_poe = poe_fk_space([S1, S2], M_2r, q_test)

# DH 对比
dh_2r = np.array([[l1, 0, 0, q_test[0]], [l2, 0, 0, q_test[1]]])
T_dh, _ = forward_kinematics(dh_2r, 'sdh')

print(f"PoE FK:\n{np.round(T_poe, 4)}")
print(f"\nDH FK:\n{np.round(T_dh, 4)}")
print(f"\n一致? {np.allclose(T_poe, T_dh, atol=1e-10)}")

# PoE 空间雅可比
J_poe = compute_space_jacobian_poe([S1, S2], q_test)

# 经典几何雅可比（用于对比）
from src.robotics_learning.kinematics import compute_geometric_jacobian
J_dh = compute_geometric_jacobian(np.array([[l1,0,0],[l2,0,0]]), q_test)
print(f"\nPoE Space Jacobian [ω; v]:\n{np.round(J_poe, 4)}")
print(f"DH 几何雅可比 [ṗ_E; ω]:\n{np.round(J_dh, 4)}")

# 验证: 角速度部分应一致, FK 应一致
assert np.allclose(T_poe, T_dh, atol=1e-10), "PoE FK ≠ DH FK!"
assert np.allclose(J_poe[:3], J_dh[3:], atol=1e-10), "角速度部分不一致!"
print("✅ PoE FK 与 DH FK 一致, 角速度部分一致")
print("(线速度部分不同：J_poe 的 v 是空间系原点速度, J_dh 上半部是末端点速度)")

# %% [markdown]
# ## 7. 练习题
#
# ### 手算题
# 1. 为 2R 臂在 $q_1=30°, q_2=45°$ 时手算 PoE FK，验证与 DH 结果一致。
# 2. 用 PoE 公式推导 2R 臂的空间雅可比。
#
# ### 编程题
# 1. 实现通用 n-DOF PoE FK 函数。
# 2. 对 6R 工业机器人（如 UR5）建立 PoE 模型并与 DH 对比。
#
# > 答案见 `solutions/solutions_week2.ipynb`
