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
# # Notebook 12：高级动力学专题
#
# ## 1. 本节在知识体系中的位置
#
# ```
# NB10-11 基础动力学 ──→ NB12 高级专题 ──→ NB18 自适应控制
# (M/C/g, RNEA)       (参数辨识/约束/浮动基)
# ```
#
# 在掌握了动力学建模和高效算法后，本节探讨三个工程中的高级问题：**惯性参数辨识**、**约束与接触动力学**和**浮动基机器人动力学**。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解动力学参数的线性化形式 $\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q},\dot{\mathbf{q}},\ddot{\mathbf{q}})\boldsymbol{\theta}$
# - ⭐ 了解动力学参数辨识的基本流程
# - 📖 理解约束动力学和拉格朗日乘子 $\lambda$ 的含义
# - 📖 理解接触动力学中的摩擦锥线性化
# - 📚 浮动基机器人的 6 个虚关节

# %% [markdown]
# ## 3. 动力学参数线性化 ⭐

# %% [markdown]
# ### 3.1 每个连杆的 10 个线性惯性参数
#
# 标准线性惯性参数使用**关于连杆坐标系原点 O_i** 的惯性张量：
#
# | 参数 | 符号 | 说明 |
# |------|------|------|
# | 质量 | $m_i$ | 连杆质量 |
# | 一阶质量矩 | $\mathbf{h}_i = m_i\mathbf{c}_i = [mc_x, mc_y, mc_z]^T$ | 质量 × 质心位置 |
# | 惯性张量 | $\mathbf{I}_{O,i}$ (6 个独立分量) | 关于连杆坐标系原点 O_i 的惯性张量 |
#
# **注意：** 这里使用 $\mathbf{I}_O$（关于原点）而非 $\mathbf{I}_C$（关于质心）。
# 因为 $\boldsymbol{\tau} = \mathbf{Y}\boldsymbol{\theta}$ 要求 $\boldsymbol{\theta}$ 中所有元素线性独立；
# 若同时使用 $m\mathbf{c}$ 和 $\mathbf{I}_C$，平行轴定理 $\mathbf{I}_O = \mathbf{I}_C + m(|c|^2\mathbf{I} - \mathbf{c}\mathbf{c}^T)$
# 会在参数向量中引入 $m$ 的二次项，破坏线性性质。
#
# 标准 10 参数向量:
# $$\boldsymbol{\theta}_i = [m_i, mc_{x,i}, mc_{y,i}, mc_{z,i}, I_{O,xx,i}, I_{O,xy,i}, I_{O,xz,i}, I_{O,yy,i}, I_{O,yz,i}, I_{O,zz,i}]^T$$
#
# 共 $4 + 6 = 10$ 个线性无关参数/连杆。加上电机转子惯量则更多。

# %% [markdown]
# ### 3.2 参数线性化形式
#
# 逆动力学方程对惯性参数是**线性的**：
# $$\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) \boldsymbol{\theta}$$
#
# - $\boldsymbol{\theta} \in \mathbb{R}^{10n}$：所有连杆惯性参数的堆叠
# - $\mathbf{Y} \in \mathbb{R}^{n \times 10n}$：回归矩阵（regressor matrix），由运动学量构造
#
# 这是**自适应控制**（NB18 拓展）的基础——控制器通过 $\mathbf{Y}\hat{\boldsymbol{\theta}}$ 来在线更新动力学参数估计。

# %% [markdown]
# ### 3.3 参数可辨识性问题
#
# 不是所有 10n 个参数都可以从 $\boldsymbol{\tau}$ 的测量中辨识出来——有些参数组合对 $\boldsymbol{\tau}$ 没有影响，或总是以组合形式出现。
# - **基参数（Base Parameters）**：最小可辨识参数集
# - 使用 QR 分解 $\mathbf{Y}$ 矩阵来选择基参数

# %% [markdown]
# ## 4. 约束动力学 ⭐

# %% [markdown]
# ### 4.1 完整约束与拉格朗日乘子
#
# 约束方程 $\boldsymbol{\phi}(\mathbf{q}) = \mathbf{0}$（例如末端始终接触某个表面）。
#
# 带约束的动力学：
# $$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau} + \mathbf{J}_c^T(\mathbf{q})\boldsymbol{\lambda}$$
#
# 其中 $\mathbf{J}_c = \partial\boldsymbol{\phi}/\partial\mathbf{q}$ 是约束雅可比，$\boldsymbol{\lambda}$ 是**拉格朗日乘子**（即约束力）。
#
# 物理含义：
# - $\boldsymbol{\lambda}$ 是约束面施加在机器人上的反作用力
# - $\mathbf{J}_c^T\boldsymbol{\lambda}$ 是该反作用力对关节力矩的贡献

# %% [markdown]
# ### 4.2 摩擦锥与接触
#
# 库仑摩擦：接触力 $\mathbf{f}_c = [f_n, f_{t1}, f_{t2}]^T$ 需满足：
# $$\|[f_{t1}, f_{t2}]\| \leq \mu f_n$$
#
# 这是一个**二阶锥约束**。在 QP 中常将其线性化（用多面体近似圆锥）。

# %% [markdown]
# ## 5. Python 验证

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics
%matplotlib inline
print("✅ 导入完成")

# %% [markdown]
# ### 5.1 参数辨识仿真

# %%
# 使用"真"动力学参数生成数据，再加噪声模拟测量
dyn_true = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=1.0, l2=0.8, lc1=0.5, lc2=0.4, I1=0.083, I2=0.067, g=9.81)

# 2R 臂 Y 矩阵基于复合参数: θ = [α, β, δ, m1*lc1*g + m2*l1*g, m2*lc2*g]
# τ = Y(q, q̇, q̈) θ（简化版线性参数化）

def Y_matrix_2r(q, q_dot, q_ddot):
    """2R 臂简化的回归矩阵（用复合参数）"""
    c2 = np.cos(q[1]); s2 = np.sin(q[1])
    c1 = np.cos(q[0]); c12 = np.cos(q[0]+q[1])

    Y = np.zeros((2, 5))
    Y[0, 0] = q_ddot[0]  # α
    Y[0, 1] = 2*c2*q_ddot[0] + c2*q_ddot[1] - s2*q_dot[1]*(2*q_dot[0] + q_dot[1])
    Y[0, 2] = q_ddot[1]  # δ
    Y[0, 3] = c1  # gravity term for joint 1
    Y[0, 4] = c12

    Y[1, 0] = 0
    Y[1, 1] = c2*q_ddot[0] + s2*q_dot[0]**2
    Y[1, 2] = q_ddot[0] + q_ddot[1]
    Y[1, 3] = 0
    Y[1, 4] = c12
    return Y

# 生成激励数据
n_samples = 500
rng = np.random.RandomState(42)
q_data = rng.uniform(-np.pi/2, np.pi/2, (n_samples, 2))
q_dot_data = rng.uniform(-2, 2, (n_samples, 2))
q_ddot_data = rng.uniform(-5, 5, (n_samples, 2))

# 真参数（复合参数）
alpha_true = dyn_true.alpha; beta_true = dyn_true.beta; delta_true = dyn_true.delta
g1_coeff = dyn_true.m1*dyn_true.lc1*9.81 + dyn_true.m2*dyn_true.l1*9.81
g2_coeff = dyn_true.m2*dyn_true.lc2*9.81
theta_true = np.array([alpha_true, beta_true, delta_true, g1_coeff, g2_coeff])

# 生成测量 τ（加噪声）
Y_stack = np.zeros((2*n_samples, 5))
tau_meas = np.zeros(2*n_samples)
for i in range(n_samples):
    tau_true_i = dyn_true.inverse_dynamics(q_data[i], q_dot_data[i], q_ddot_data[i])
    tau_noisy = tau_true_i + rng.normal(0, 0.1, 2)
    tau_meas[2*i:2*i+2] = tau_noisy
    Y_stack[2*i:2*i+2] = Y_matrix_2r(q_data[i], q_dot_data[i], q_ddot_data[i])

# 最小二乘辨识
theta_est = np.linalg.lstsq(Y_stack, tau_meas, rcond=None)[0]

print("参数辨识结果:")
print(f"{'参数':<12} {'真值':>10} {'估计值':>10} {'误差%':>10}")
for name, true, est in zip(['α','β','δ','g1_coeff','g2_coeff'], theta_true, theta_est):
    err = abs(est - true) / (abs(true)+1e-10) * 100
    print(f"{name:<12} {true:>10.4f} {est:>10.4f} {err:>9.1f}%")

# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
x_pos = np.arange(len(theta_true))
ax.bar(x_pos - 0.2, theta_true, 0.4, label='True', color='blue', alpha=0.7)
ax.bar(x_pos + 0.2, theta_est, 0.4, label='Estimated', color='red', alpha=0.7)
ax.set_xticks(x_pos); ax.set_xticklabels(['α', 'β', 'δ', 'g1_coeff', 'g2_coeff'])
ax.set_ylabel('Parameter Value'); ax.set_title('Dynamics Parameter Identification')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/12_parameter_identification.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ 参数辨识成功！估计值与真值高度一致（受噪声影响的合理误差）")

# %% [markdown]
# ### 5.2 约束力仿真

# %%
# 模拟 2R 臂末端接触垂直墙壁（x = x_wall）
x_wall = 1.5
q_test = np.array([np.pi/4, np.pi/6])

# FK 求末端位置
dh_test = np.array([[1.0, 0, 0, q_test[0]], [0.8, 0, 0, q_test[1]]])
from src.robotics_learning.kinematics import forward_kinematics
T_end, _ = forward_kinematics(dh_test)
x_end = T_end[0, 3]

print(f"末端 x 位置: {x_end:.4f}")
print(f"墙体 x 位置: {x_wall}")
print(f"穿透? {x_end > x_wall}")

# 约束雅可比 J_c = ∂φ/∂q, φ(q) = x_end - x_wall
l1_c, l2_c = 1.0, 0.8
J_c = np.array([[-l1_c*np.sin(q_test[0]) - l2_c*np.sin(q_test[0]+q_test[1]),
                 -l2_c*np.sin(q_test[0]+q_test[1])]])

# 约束力 λ 需要通过接触动力学方程求解（此处省略完整仿真）
print(f"约束雅可比 J_c = {np.round(J_c, 4)}")
print("J_c^T λ 将在下一时间步影响关节力矩，约束末端不穿透墙壁")

# %% [markdown]
# ## 6. 常见错误
#
# 1. **所有参数都可以辨识**：不。只有基参数可以从运动数据中辨识。某些参数组合对 $\boldsymbol{\tau}$ 不可见。
# 2. **约束力方向**：$\mathbf{J}_c^T\boldsymbol{\lambda}$ 中 $\boldsymbol{\lambda}$ 的正负取决于约束法向的定义。$\lambda_n > 0$ 表示"推开"力。
# 3. **浮动基 vs 固定基**：浮动基机器人（人形、四足）基座位姿属于 SE(3)（6 个速度自由度）。常以 7 维位姿向量表示（位置 3 + 四元数 4，含单位范数约束），另加 n 个内部关节，**速度层面**总 DOF = n + 6。前 6 个速度自由度是欠驱动的（无直接力矩）。注意配置向量维度（7+n）与速度自由度（6+n）的区分。

# %% [markdown]
# ## 7. 练习题
#
# ### 概念题
# 1. 为什么不是所有惯性参数都可以辨识？
# 2. 拉格朗日乘子 $\lambda$ 的物理含义是什么？
#
# ### 编程题
# 1. 使用 QR 分解从 Y 矩阵中选出基参数。
# 2. 模拟 2R 臂末端接触墙壁的约束动力学（使用罚函数法或拉格朗日乘子法）。
#
# > 答案见 `solutions/12_solutions.ipynb`

# %% [markdown]
# ## 8. 本节总结
#
# | 概念 | 关键公式 | 用途 |
# |------|----------|------|
# | 参数线性化 | $\boldsymbol{\tau} = \mathbf{Y}\boldsymbol{\theta}$ | 辨识、自适应控制 |
# | 基参数 | 通过 QR(Y) 选取 | 最小可辨识集 |
# | 约束动力学 | $\mathbf{M}\ddot{\mathbf{q}} + \dots = \boldsymbol{\tau} + \mathbf{J}_c^T\boldsymbol{\lambda}$ | 接触、闭环 |
# | 摩擦锥 | $\|f_t\| \leq \mu f_n$ | 抓取、行走 |
