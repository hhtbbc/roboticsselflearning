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
# # Notebook 09b：刚体惯性 — 张量、主轴与平行轴定理

# %% [markdown]
# ## 1. 定位
#
# NB09-10 推导了 2R 臂的 $\mathbf{M}(\mathbf{q})$ 但未深入刚体惯性本身。本节补全惯性张量的物理定义、$I_C$ vs $I_O$、主惯量和平行轴定理——这些是正确建立动力学参数化的前提。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 理解惯性张量 $\mathbf{I}$ 的积分定义和物理含义
# - ⭐ 区分 $I_O$（关于原点）和 $I_C$（关于质心）
# - ⭐ 掌握平行轴定理（Steiner's Theorem）
# - ⭐ 理解主惯量和主轴
# - ⭐ 知道标准 10 参数 $\boldsymbol{\theta}$ 使用 $I_O$ 而非 $I_C$

# %% [markdown]
# ## 3. 惯性张量 ⭐

# %% [markdown]
# ### 3.1 积分定义
#
# 刚体关于参考点 $O$ 的惯性张量：
# $$\mathbf{I}_O = \int_{\mathcal{B}} \left( \|\mathbf{r}\|^2 \mathbf{I} - \mathbf{r}\mathbf{r}^T \right) \rho \, dV$$
#
# 其中 $\mathbf{r} = [x, y, z]^T$ 是从 $O$ 到体积元的位置向量。
#
# 矩阵形式：
# $$\mathbf{I}_O = \begin{bmatrix}
# I_{xx} & I_{xy} & I_{xz} \\
# I_{xy} & I_{yy} & I_{yz} \\
# I_{xz} & I_{yz} & I_{zz}
# \end{bmatrix}$$
#
# 对角元素 $I_{xx} = \int (y^2+z^2)\rho dV$ 是绕 X 轴的转动惯量。
# 非对角元素 $I_{xy} = -\int xy\rho dV$ 是惯性积（product of inertia）。

# %% [markdown]
# ### 3.2 主惯量和主轴
#
# $\mathbf{I}_O$ 是对称正定矩阵。其特征值 $\lambda_1, \lambda_2, \lambda_3$ 是**主惯量**，特征向量是**主轴**。
#
# 若坐标轴与主轴对齐，$\mathbf{I}_O$ 是对角矩阵：
# $$\mathbf{I}_O = \text{diag}(I_{xx}, I_{yy}, I_{zz})$$
#
# 工程中常通过设计使连杆的惯性张量近似对角化。

# %% [markdown]
# ## 4. 平行轴定理 ⭐

# %% [markdown]
# 已知关于质心 $C$ 的惯性张量 $\mathbf{I}_C$，求关于另一点 $O$ 的 $\mathbf{I}_O$（$O$ 和 $C$ 在刚体上以向量 $\mathbf{c}$ 相连）：
#
# $$\boxed{\mathbf{I}_O = \mathbf{I}_C + m\left( \|\mathbf{c}\|^2 \mathbf{I} - \mathbf{c}\mathbf{c}^T \right)}$$
#
# 其中 $\mathbf{c} = \mathbf{r}_{OC} = [c_x, c_y, c_z]^T$ 是从 $O$ 指向 $C$ 的向量。
#
# 展开：
# $$I_{O,xx} = I_{C,xx} + m(c_y^2 + c_z^2)$$
# $$I_{O,xy} = I_{C,xy} - m c_x c_y$$

# %% [markdown]
# ### 4.1 为什么标准参数用 $I_O$ 而非 $I_C$？

# %% [markdown]
# 动力学方程 $\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})\boldsymbol{\theta}$ 对 $I_O$ 是线性的，对 $I_C$ 不是。
#
# 因为 $I_O$ 直接出现在 RNEA 递推中（关节力矩贡献来自关于关节坐标系原点的惯性），而 $I_C$ 需要通过平行轴定理转回 $I_O$。使用 $I_O$ 避免了非线性参数组合。
#
# 标准 10 参数：
# $$\boldsymbol{\theta}_i = [m_i, mc_{x,i}, mc_{y,i}, mc_{z,i}, I_{O,xx,i}, I_{O,xy,i}, I_{O,xz,i}, I_{O,yy,i}, I_{O,yz,i}, I_{O,zz,i}]^T$$

# %% [markdown]
# ## 5. 转动动能

# %% [markdown]
# 角速度为 $\boldsymbol{\omega}$ 的刚体关于质心的转动动能：
# $$K_{rot} = \frac{1}{2} \boldsymbol{\omega}^T \mathbf{I}_C \boldsymbol{\omega}$$
#
# 这等价于：$K_{rot} = \frac{1}{2} \dot{\mathbf{q}}^T \mathbf{M}_{rot}(\mathbf{q}) \dot{\mathbf{q}}$，其中 $\mathbf{M}_{rot}$ 是质量矩阵中与转动相关的部分。

# %% [markdown]
# ## 6. Python 验证

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %%
def parallel_axis(I_C, m, c):
    """平行轴定理: I_O = I_C + m(|c|^2 I - c c^T)"""
    return I_C + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

# 均匀密度长方体 (a×b×c)，质量 m
a, b, c_box = 0.1, 0.2, 0.3
m_box = 2.0
I_C_box = (m_box / 12) * np.diag([b**2 + c_box**2, a**2 + c_box**2, a**2 + b**2])
print(f"I_C (质心惯量):\n{np.round(I_C_box, 4)}")

# 验证 I_O 正定性
c_vec = np.array([0.5, 0.3, 0.2])
I_O = parallel_axis(I_C_box, m_box, c_vec)
eigvals = np.linalg.eigvalsh(I_O)
print(f"\nI_O (关于 O):\n{np.round(I_O, 4)}")
print(f"I_O 特征值: {np.round(eigvals, 4)} (全部 > 0 = 正定 ✅)")

# 验证: 绕远离质心的轴旋转需要更大的惯量
print(f"\nI_O 的对角元素 > I_C 的对角元素:")
for i, ax in enumerate(['xx', 'yy', 'zz']):
    print(f"  I_O,{ax} = {I_O[i,i]:.4f}, I_C,{ax} = {I_C_box[i,i]:.4f}, 增量 = {I_O[i,i]-I_C_box[i,i]:.4f}")

# 验证转动动能不变性
omega = np.array([1.0, 0.5, -0.3])
K_C = 0.5 * omega @ I_C_box @ omega
# 绕 O 的转动动能 = ½ ω^T I_O ω  (因为绕质心的角速度和绕 O 的角速度相同)
K_O = 0.5 * omega @ I_O @ omega
print(f"\nK_C = {K_C:.4f}, K_O = {K_O:.4f} (K_O > K_C 因为远离质心转动需要更多能量)")

# %% [markdown]
# ### 主轴可视化

# %%
# 随机惯性张量的主轴
I_random = rng.randn(3, 3)
I_random = I_random.T @ I_random + np.eye(3)  # 对称正定
eigvals, eigvecs = np.linalg.eigh(I_random)

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(3):
    ax.arrow(0, 0, eigvecs[0, i]*eigvals[i], eigvecs[1, i]*eigvals[i],
             head_width=0.1, head_length=0.1, linewidth=2,
             color=['r','g','b'][i], label=f'λ{i+1}={eigvals[i]:.2f}')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Principal Axes of Inertia Tensor')
ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/09b_principal_axes.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 练习题
#
# ### 概念题
# 1. 为什么标准 10 参数使用 $I_O$ 而非 $I_C$？
# 2. 惯性积 $I_{xy}$ 什么时候为零？
#
# ### 手算题
# 1. 均匀杆（长 L，质量 m）绕一端点的转动惯量？先用平行轴定理，再用积分定义验证。
#
# ### 编程题
# 1. 实现通用刚体惯性类，支持任意参考点的惯性张量计算。
# 2. 验证 $I_O$ 的 6 个独立分量通过平行轴定理可生成 $I_C$。
