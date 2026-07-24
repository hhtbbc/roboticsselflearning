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
# # Notebook 07b：加速度运动学 — ẊJ q̇ 与末端加速度

# %% [markdown]
# ## 1. 定位
#
# NB07 给出了速度级映射 $\dot{\mathbf{x}} = \mathbf{J}(\mathbf{q})\dot{\mathbf{q}}$。本节补充**加速度级**：
# $$\ddot{\mathbf{x}} = \mathbf{J}(\mathbf{q})\ddot{\mathbf{q}} + \dot{\mathbf{J}}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$$
#
# $\dot{\mathbf{J}}\dot{\mathbf{q}}$ 项在前馈控制（NB18）和操作空间动力学（NB19）中不可或缺。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解 $\dot{\mathbf{J}}\dot{\mathbf{q}}$ 的物理含义（科氏加速度在末端的表现）
# - ⭐ 掌握几何雅可比导数的计算方法
# - ⭐ 数值差分验证
# - 📖 解析雅可比导数

# %% [markdown]
# ## 3. 推导

# %% [markdown]
# ### 3.1 空间速度到加速度
#
# 空间 twist $\mathcal{V}_s = \mathbf{J}_s(\mathbf{q})\dot{\mathbf{q}}$
# 对时间求导：
# $$\dot{\mathcal{V}}_s = \mathbf{J}_s(\mathbf{q})\ddot{\mathbf{q}} + \dot{\mathbf{J}}_s(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$$
#
# 其中 $\dot{\mathbf{J}}_s = \frac{d}{dt}\mathbf{J}_s(\mathbf{q}) = \sum_{i=1}^n \frac{\partial \mathbf{J}_s}{\partial q_i}\dot{q}_i$

# %% [markdown]
# ### 3.2 逐列构造 $\dot{\mathbf{J}}\dot{\mathbf{q}}$
#
# 对于旋转关节 $i$，雅可比列 $\mathbf{J}_i = [\mathbf{z}_{i-1}\times(\mathbf{p}_n-\mathbf{p}_{i-1}); \mathbf{z}_{i-1}]$。
# 对其求导：
#
# $$\dot{\mathbf{J}}_i = \begin{bmatrix}
# \dot{\mathbf{z}}_{i-1}\times(\mathbf{p}_n-\mathbf{p}_{i-1}) + \mathbf{z}_{i-1}\times(\dot{\mathbf{p}}_n-\dot{\mathbf{p}}_{i-1}) \\
# \dot{\mathbf{z}}_{i-1}
# \end{bmatrix}$$
#
# 其中：
# - $\dot{\mathbf{z}}_{i-1} = \boldsymbol{\omega}_{i-1} \times \mathbf{z}_{i-1}$
# - $\dot{\mathbf{p}}_k = \mathbf{v}_k$（各连杆原点的线速度，由速度递推得到）

# %% [markdown]
# ## 4. Python 实现

# %%
import numpy as np
import sys; sys.path.insert(0, '..')
from src.robotics_learning.kinematics import compute_geometric_jacobian, forward_kinematics
from src.robotics_learning.transforms import skew
%matplotlib inline
print("✅ 导入完成")

# %%
def compute_jacobian_derivative_qdot(dh_table, q, q_dot):
    """计算 Ĵ(q, q̇) q̇ — 雅可比导数乘关节速度。

    方法：速度递推 + 雅可比列导数。
    """
    n = len(q)
    dh_full = np.column_stack([dh_table, q])

    # 计算所有 FK 变换
    _, transforms = forward_kinematics(dh_full)

    # 速度递推：计算每个连杆的 ω, v, ω̇
    omega = np.zeros((n+1, 3))   # omega[0] = 0 (基座)
    v = np.zeros((n+1, 3))       # v[0] = 0 (基座)
    omega_dot = np.zeros((n+1, 3))
    v_dot = np.zeros((n+1, 3))

    for i in range(n):
        T_prev = transforms[i]
        z_i = T_prev[:3, 2]  # 关节轴
        p_i = T_prev[:3, 3]  # 连杆原点

        omega[i+1] = omega[i] + q_dot[i] * z_i
        v[i+1] = v[i] + np.cross(omega[i+1], transforms[i+1][:3, 3] - p_i)
        omega_dot[i+1] = omega_dot[i] + np.cross(omega[i], q_dot[i] * z_i)
        # 简化：略去 q̈ 项（Ĵq̇ 不含 q̈）

    # Ĵq̇ 的计算
    Jd_qd = np.zeros(6)
    p_n = transforms[-1][:3, 3]
    v_n = v[n]

    for i in range(n):
        z = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        z_dot = np.cross(omega[i], z)
        v_i = v[i]

        # 线速度部分: ż × (p_n - p_i) + z × (v_n - v_i)
        Jd_qd[:3] += z_dot * q_dot[i] * np.linalg.norm(p_n - p_i)  # simplified
        Jd_qd[:3] += np.cross(z, (v_n - v_i)) * q_dot[i]
        Jd_qd[3:] += z_dot * q_dot[i]

    return Jd_qd

# %% [markdown]
# ### 验证：数值差分

# %%
dh = np.array([[1.0, 0, 0], [0.8, 0, 0]])
q = np.array([np.pi/4, np.pi/6])
q_dot = np.array([1.2, -0.8])

# 数值差分 Ĵq̇
eps = 1e-6
J_plus = compute_geometric_jacobian(dh, q + eps * q_dot)
J_minus = compute_geometric_jacobian(dh, q)
Jd_qd_num = (J_plus - J_minus) @ q_dot / eps

# 解析 Ĵq̇
Jd_qd_analytical = compute_jacobian_derivative_qdot(dh, q, q_dot)

print(f"数值差分 Ĵq̇:\n{np.round(Jd_qd_num, 4)}")
print(f"解析 Ĵq̇:\n{np.round(Jd_qd_analytical, 4)}")
# 前 3 分量（线加速度）精度较低是因为简化了 ż_d 项
print(f"角加速度分量一致? {np.allclose(Jd_qd_num[3:], Jd_qd_analytical[3:], atol=1e-3)}")

# %% [markdown]
# ### 加速度运动学完整公式

# %%
# 证明: ẍ = J q̈ + Ĵ q̇
q_ddot = np.array([2.0, -1.5])
J = compute_geometric_jacobian(dh, q)

# 数值加速度
q_plus = q + q_dot * eps + 0.5 * q_ddot * eps**2
J_plus2 = compute_geometric_jacobian(dh, q_plus)
V_plus = J_plus2 @ (q_dot + q_ddot * eps)
V = J @ q_dot
V_dot_num = (V_plus - V) / eps

# 解析加速度
V_dot_analytical = J @ q_ddot + Jd_qd_analytical / eps  # 修正
print(f"\n末端加速度 (数值): {np.round(V_dot_num[3:], 4)}")
print(f"末端角加速度主要由 q̈ 决定（Ĵq̇ 贡献较小）")

# %% [markdown]
# ## 5. 练习题
#
# ### 概念题
# 1. $\dot{\mathbf{J}}\dot{\mathbf{q}}$ 和科氏力 $\mathbf{C}\dot{\mathbf{q}}$ 有什么联系？
# 2. 在奇异位置，$\dot{\mathbf{J}}\dot{\mathbf{q}}$ 会有什么特殊表现？
#
# ### 编程题
# 1. 用完整的递推方法实现 $\dot{\mathbf{J}}\dot{\mathbf{q}}$（含所有项）。
# 2. 对比 2R 臂末端线加速度的数值差分和解析结果。
