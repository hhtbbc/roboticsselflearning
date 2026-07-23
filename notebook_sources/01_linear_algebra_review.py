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
# # Notebook 01：线性代数与矩阵论复习
#
# ## 1. 本节在知识体系中的位置
#
# ```
# 本课程四条主线：
#   建模 → 规划 → 控制 → 感知与状态估计
#           ↑
#   本 Notebook 是以上所有主线的数学语言基础
# ```
#
# 机器人学中的几乎所有计算——从坐标变换到动力学、从优化到滤波——都使用线性代数。本 Notebook 不是"从头教线性代数"，而是**快速回顾机器人学中最高频使用的工具**。

# %% [markdown]
# ## 2. 学习目标
#
# - ⭐ 掌握向量和矩阵的基本运算及其几何意义
# - ⭐ 掌握特征值分解和奇异值分解（SVD）
# - ⭐ 理解伪逆（Moore-Penrose pseudoinverse）及其在 IK 和优化中的作用
# - ⭐ 能够用 NumPy 高效实现所有操作
# - 📖 理解正定性及其在动力学 M(q) 中的关键作用
# - 📚 了解 PCA 与其他矩阵分解

# %% [markdown]
# ## 3. 前置知识
#
# - 基础 Python 编程
# - 高中/大学低年级线性代数基本概念

# %% [markdown]
# ## 4. 内容分级
#
# - ⭐ **必须掌握**：向量/矩阵运算、特征值/SVD、伪逆、最小二乘
# - 📖 **需要理解**：Cholesky 分解、矩阵指数
# - 📚 **拓展**：PCA、数值稳定性

# %% [markdown]
# ## 5. 向量与矩阵基础
#
# ### 5.1 向量（Vector）
#
# 在机器人学中，向量通常表示：
# - **位置**：$\mathbf{p} = [p_x, p_y, p_z]^T \in \mathbb{R}^3$
# - **速度**：$\mathbf{v} = [v_x, v_y, v_z]^T$
# - **关节角**：$\mathbf{q} = [q_1, q_2, \dots, q_n]^T \in \mathbb{R}^n$
# - **力/力矩**：$\boldsymbol{\tau} = [\tau_1, \dots, \tau_n]^T$
#
# ### 5.2 内积（Dot Product / Inner Product）
#
# $$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T\mathbf{b} = \sum_{i=1}^n a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$$
#
# 几何意义：$\mathbf{a}$ 在 $\mathbf{b}$ 方向上的投影长度乘以 $\|\mathbf{b}\|$。
# 在机器人学中用于计算**力做功**（$\mathbf{F} \cdot \Delta\mathbf{x}$）和**投影**。

# %% [markdown]
# ### 5.3 外积 / 叉乘（Cross Product）
#
# $$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_y b_z - a_z b_y \\ a_z b_x - a_x b_z \\ a_x b_y - a_y b_x \end{bmatrix}$$
#
# 几何意义：垂直于 $\mathbf{a}$ 和 $\mathbf{b}$ 所张成平面的向量，大小 = $\|\mathbf{a}\|\|\mathbf{b}\|\sin\theta$。
# 在机器人学中用于：
# - 计算**力矩** $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$
# - 计算**角速度对线速度的贡献** $\mathbf{v} = \boldsymbol{\omega} \times \mathbf{r}$
# - 构造**叉乘矩阵** $[\boldsymbol{\omega}]_\times$（见 NB02）

# %% [markdown]
# ### 5.4 叉乘矩阵（Skew-Symmetric Matrix）
#
# 叉乘 $\mathbf{a} \times \mathbf{b}$ 可以写为矩阵乘法形式：
#
# $$[\mathbf{a}]_\times \mathbf{b} = \begin{bmatrix}
# 0 & -a_z & a_y \\
# a_z & 0 & -a_x \\
# -a_y & a_x & 0
# \end{bmatrix} \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix}$$
#
# $[\mathbf{a}]_\times$ 满足 $[\mathbf{a}]_\times^T = -[\mathbf{a}]_\times$（反对称），是 $\mathfrak{so}(3)$ 李代数的元素。

# %% [markdown]
# ## 6. 矩阵运算

# %% [markdown]
# ### 6.1 矩阵乘法
#
# 对于 $\mathbf{C} = \mathbf{A}\mathbf{B}$，其中 $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{B} \in \mathbb{R}^{n \times p}$：
#
# $$C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$
#
# **左乘**（$\mathbf{A}\mathbf{x}$）：对 $\mathbf{x}$ 的列空间进行线性变换（如旋转、缩放）。
# **右乘**（$\mathbf{x}^T\mathbf{A}$）：对 $\mathbf{x}$ 的行空间进行操作。
#
# 在机器人学中，**左乘 vs 右乘**的物理含义差���极大：
# - 左乘旋转变换矩阵：相对于固定参考系旋转
# - 右乘旋转变换矩阵：相对于当前物体自身坐标系旋转

# %% [markdown]
# ### 6.2 矩阵秩（Rank）
#
# $\text{rank}(\mathbf{A})$ = 线性无关的行数（或列数）= $\text{dim}(\text{col}(\mathbf{A}))$
#
# 在机器人学中的作用：
# - rank(J(q)) = 6：机械臂末端可实现任意 6 维速度（非奇异）
# - rank(J(q)) < 6：存在奇异性（Singularity，NB08）

# %% [markdown]
# ### 6.3 行列式（Determinant）
#
# $\det(\mathbf{A})$ = 矩阵 $\mathbf{A}$ 所代表的线性变换对体积的缩放因子。
#
# - $\det(\mathbf{R}) = 1$：旋转矩阵保持体积不变（SO(3) 的正交性）
# - $\det(\mathbf{J}) = 0$：雅可比的奇异性条件
# - $\det(\mathbf{M}) > 0$：质量矩阵的正定性

# %% [markdown]
# ## 7. 特征值分解与奇异值分解

# %% [markdown]
# ### 7.1 特征值分解（Eigendecomposition）
#
# $$\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$$
#
# 对于**对称矩阵** $\mathbf{A} = \mathbf{A}^T$：
#
# $$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T = \sum_{i=1}^n \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$
#
# 在机器人学中：
# - **惯性矩阵** $\mathbf{M}(\mathbf{q})$ 是对称正定的，其特征值决定了各方向上"有效惯性"的大小
# - **可操作度椭球**（NB08）的主轴方向和长度来自 $\mathbf{J}\mathbf{J}^T$ 的特征分解
# - **协方差矩阵** $\boldsymbol{\Sigma}$（NB21-23）的特征分解给出不确定性的主轴

# %% [markdown]
# ### 7.2 奇异值分解（Singular Value Decomposition, SVD）
#
# **SVD 是机器人学中最重要的矩阵分解之一。**
#
# $$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
#
# 其中：
# - $\mathbf{A} \in \mathbb{R}^{m \times n}$
# - $\mathbf{U} \in \mathbb{R}^{m \times m}$：左奇异向量（正交矩阵），$\mathbf{U}\mathbf{U}^T = \mathbf{I}$
# - $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$：奇异值对角矩阵，$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$
# - $\mathbf{V} \in \mathbb{R}^{n \times n}$：右奇异向量（正交矩阵），$\mathbf{V}\mathbf{V}^T = \mathbf{I}$
#
# **几何解释**：任意线性变换 $\mathbf{A}$ 可以分解为三个步骤：
# 1. $\mathbf{V}^T$：旋转（正交变换）
# 2. $\boldsymbol{\Sigma}$：沿坐标轴缩放
# 3. $\mathbf{U}$：再次旋转
#
# 这意味着 **$\mathbf{A}$ 将单位球映射为椭球**，其半轴长度 = 奇异值，方向 = $\mathbf{U}$ 的列向量。

# %% [markdown]
# ### 7.3 SVD 在机器人学中的核心应用
#
# 1. **伪逆（Pseudoinverse）**：用于求解 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 的最小范数最小二乘解
#    $$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$$
#    其中 $\boldsymbol{\Sigma}^+$ 是对 $\boldsymbol{\Sigma}$ 的非零奇异值取倒数后转置
#
# 2. **奇异性检测**：雅可比的最小奇异值 $\sigma_{\min}$ → 0 时机器人接近奇异
#
# 3. **可操作度**：$\mu = \prod \sigma_i = \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$
#
# 4. **数值 IK**：使用阻尼最小二乘（DLS）时
#    $$\Delta\mathbf{q} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda^2\mathbf{I})^{-1}\Delta\mathbf{x}$$
#    $\lambda$ 阻尼项使逆的奇异值平滑过渡：$\sigma_i \to \sigma_i/(\sigma_i^2 + \lambda^2)$

# %% [markdown]
# ## 8. 最小二乘与伪逆

# %% [markdown]
# ### 8.1 最小二乘问题
#
# 对于超定方程组 $\mathbf{A}\mathbf{x} = \mathbf{b}$（方程数 > 未知数），一般无精确解。
# 最小二乘解最小化残差平方和：
#
# $$\mathbf{x}^* = \arg\min_\mathbf{x} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$$
#
# **正规方程**（当 $\mathbf{A}^T\mathbf{A}$ 可逆时）：
# $$\mathbf{x}^* = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$
#
# **SVD 解法**（更数值稳定）：
# $$\mathbf{x}^* = \mathbf{A}^+\mathbf{b} = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T\mathbf{b}$$

# %% [markdown]
# ### 8.2 Moore-Penrose 伪逆
#
# 对于**任意**矩阵 $\mathbf{A}$，伪逆 $\mathbf{A}^+$ 满足四条 Penrose 条件：
#
# 1. $\mathbf{A}\mathbf{A}^+\mathbf{A} = \mathbf{A}$
# 2. $\mathbf{A}^+\mathbf{A}\mathbf{A}^+ = \mathbf{A}^+$
# 3. $(\mathbf{A}\mathbf{A}^+)^T = \mathbf{A}\mathbf{A}^+$
# 4. $(\mathbf{A}^+\mathbf{A})^T = \mathbf{A}^+\mathbf{A}$
#
# **在机器人学中的关键应用**：
# - 数值 IK：$\Delta\mathbf{q} = \mathbf{J}^+\Delta\mathbf{x}$（NB06）
# - 冗余分解：$\dot{\mathbf{q}} = \mathbf{J}^+\dot{\mathbf{x}} + (\mathbf{I} - \mathbf{J}^+\mathbf{J})\dot{\mathbf{q}}_0$（NB19）

# %% [markdown]
# ## 9. 正定性
#
# 矩阵 $\mathbf{A} \in \mathbb{R}^{n\times n}$ **正定（Positive Definite）**，如果对于所有 $\mathbf{x} \neq \mathbf{0}$：
# $$\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$$
#
# 判断方法：
# 1. 所有特征值 > 0
# 2. 所有主子式 > 0（Sylvester 判据）
# 3. Cholesky 分解 $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ 存在
#
# **在机器人学中的关键应用**：
# - **质量矩阵 $\mathbf{M}(\mathbf{q})$ 总是对称正定**，这保证了 $\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}} = 2K > 0$（动能总是正的）
# - 正定性是**李雅普诺夫稳定性分析**的基础（NB17-18）

# %% [markdown]
# ## 10. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
%matplotlib inline

print("✅ 环境就绪")

# %% [markdown]
# ### 10.1 向量与矩阵基础操作

# %%
# === 向量操作 ===
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 0.0, -1.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"内积 a·b = {np.dot(a, b):.4f}")  # 1×4 + 2×0 + 3×(-1) = 1
print(f"外积 a×b = {np.cross(a, b)}")
print(f"L2 范数 ||a|| = {np.linalg.norm(a):.4f}")

# === 矩阵操作 ===
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

print(f"\nA =\n{A}")
print(f"A @ B =\n{A @ B}")          # 矩阵乘法
print(f"A^T =\n{A.T}")              # 转置
print(f"A⁻¹ =\n{np.linalg.inv(A)}") # 逆
print(f"rank(A) = {np.linalg.matrix_rank(A)}")
print(f"det(A) = {np.linalg.det(A):.4f}")

# %% [markdown]
# ### 10.2 叉乘矩阵

# %%
def skew(v):
    """构造叉乘矩阵 [v]_×"""
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])

v = np.array([1.0, 2.0, 3.0])
S = skew(v)
print(f"[v]_× =\n{S}")

# 验证反对称性
print(f"\nS + S^T = 0? {np.allclose(S + S.T, np.zeros((3,3)))}")

# 验证：cross(v, w) = skew(v) @ w
w = np.array([4.0, 5.0, 6.0])
print(f"cross(v, w) = {np.cross(v, w)}")
print(f"[v]_× @ w  = {S @ w}")
print(f"两者相等? {np.allclose(np.cross(v, w), S @ w)}")

# %% [markdown]
# ### 10.3 特征值分解

# %%
# 对称矩阵的特征值分解
M = np.array([[3.0, 1.0], [1.0, 2.0]])
eigvals, eigvecs = np.linalg.eigh(M)

print(f"M =\n{M}")
print(f"特征值: λ₁={eigvals[0]:.4f}, λ₂={eigvals[1]:.4f}")
print(f"特征向量:\n{eigvecs}")
print(f"验证 M = VΛV^T: {np.allclose(M, eigvecs @ np.diag(eigvals) @ eigvecs.T)}")

# 正定性检验
print(f"\nM 正定? {np.all(eigvals > 0)} (所有特征值 > 0)")

# %% [markdown]
# ### 10.4 SVD 分解与伪逆

# %%
# SVD 分解
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0],
              [2.0, 3.0, 1.0]])

U, s, Vt = np.linalg.svd(A, full_matrices=False)
Sigma = np.diag(s)

print(f"A ({A.shape[0]}×{A.shape[1]}):\n{A}")
print(f"\n奇异值: {np.round(s, 4)}")
print(f"条件数 κ(A) = σmax/σmin = {s[0]/s[-1]:.4f}")
print(f"rank(A) ≈ {np.sum(s > 1e-10)} (非零奇异值数)")

# 验证 SVD 重构
A_reconstructed = U @ Sigma @ Vt
print(f"\nA = U·Σ·V^T? {np.allclose(A, A_reconstructed, atol=1e-10)}")

# %% [markdown]
# ### 10.5 伪逆求解最小二乘

# %%
# 超定方程组 (4 方程, 3 未知数)
b = np.array([2.0, 3.0, 1.0, 4.0])

# 方法 1：正规方程（可能数值不稳定）
x_normal = np.linalg.solve(A.T @ A, A.T @ b)
print(f"正规方程解: {np.round(x_normal, 4)}")

# 方法 2：伪逆 (SVD)
A_pinv = np.linalg.pinv(A)
x_pinv = A_pinv @ b
print(f"伪逆解:     {np.round(x_pinv, 4)}")

# 方法 3：np.linalg.lstsq
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print(f"最小二乘解: {np.round(x_lstsq, 4)}")

# 残差比较
r_normal = np.linalg.norm(A @ x_normal - b)
r_pinv = np.linalg.norm(A @ x_pinv - b)
print(f"\n残差 ||Ax-b||: 正规方程={r_normal:.6f}, 伪逆={r_pinv:.6f}")

# %% [markdown]
# ### 10.6 SVD 的几何可视化

# %%
# 用 SVD 展示 A 将单位圆映射为椭圆
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 前两个输入维度上的单位圆
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])  # 3D 向量在 z=0

# 左边：输入空间（单位圆）
ax1.fill(circle[0], circle[1], color='lightblue', alpha=0.5)
ax1.plot(circle[0], circle[1], 'b-', linewidth=2)
ax1.set_title('输入空间：单位圆 $\|x\|=1$')
ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')
ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2); ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# 右边：输出空间（椭圆）+ SVD 分析
output = A[:, :2] @ circle[:2]  # 只用前两列（与前两个输入维对应）
ax2.fill(output[0], output[1], color='lightcoral', alpha=0.5)
ax2.plot(output[0], output[1], 'r-', linewidth=2)

# 画奇异向量方向
for i in range(2):
    direction = s[i] * U[:, i]
    ax2.arrow(0, 0, direction[0], direction[1],
              head_width=0.3, head_length=0.3, fc='darkred', ec='darkred',
              linewidth=2, label=f'σ{i+1}={s[i]:.2f}')

ax2.set_title('输出空间：椭圆 (SVD: 半轴 = 奇异值)')
ax2.set_xlabel('$y_1$'); ax2.set_ylabel('$y_2$')
ax2.legend()
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/01_svd_geometry.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. 常见错误与易混淆概念
#
# 1. **矩阵逆与伪逆**：只有方阵且满秩才有真正的逆 $\mathbf{A}^{-1}$。任意矩阵都有伪逆 $\mathbf{A}^+$。在数值 IK 中应使用伪逆，不应假设 $\mathbf{J}$ 是方阵。
# 2. **$\mathbf{A}^T\mathbf{A}$ 求逆 vs SVD 求伪逆**：正规方程 $\mathbf{x}=(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$ 在 $\mathbf{A}^T\mathbf{A}$ 条件数大时数值不稳定。SVD 伪逆更鲁棒。
# 3. **奇异值 vs 特征值**：只有方阵有特征值；任何矩阵都有奇异值。对于对称矩阵，奇异值 = |特征值|。
# 4. **内积 vs 外积**：内积输出标量（做功、投影），外积输出向量（叉乘、力矩）。

# %% [markdown]
# ## 12. 工程应用
#
# - **SVD → 伪逆 → 数值 IK**（每次 IK 迭代都需要解 $\mathbf{J}\Delta\mathbf{q} = \Delta\mathbf{x}$）
# - **惯性矩阵的正定性 → 控制稳定性**（$\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}} > 0$ 保证了动能正定性）
# - **协方差矩阵的对称正定性 → 不确定性建模**（保证马氏距离有意义）
# - **SVD 的最小奇异值监测 → 奇异性预警**（工业机器人监控）
# - **最小二乘 → 参数辨识**（NB12 的动力学参数辨识）

# %% [markdown]
# ## 13. 面试常见问题
#
# 1. **SVD 是什么？在机器人学中有什么用？** → 见 INTERVIEW_CHECKLIST.md #1.1
# 2. **伪逆怎么求？什么时候需要？** → 见 1.2
# 3. **什么是正定矩阵？M(q) 为什么必须正定？** → 见 1.3

# %% [markdown]
# ## 14. 练习题

# %% [markdown]
# ### 概念题
#
# 1. 对于 $\mathbf{A} \in \mathbb{R}^{m\times n}$，$\mathbf{A}^T\mathbf{A}$ 和 $\mathbf{A}\mathbf{A}^T$ 分别是什么形状？哪个一定可逆？
# 2. SVD 中，$\mathbf{U}$ 和 $\mathbf{V}$ 的列向量分别张成什么空间？
# 3. 为什么 $\mathbf{J}^T\mathbf{J}$ 的奇异值是 $\mathbf{J}$ 奇异值的平方？

# %% [markdown]
# ### 手算题
#
# 1. 给定 $\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}$，求 $\mathbf{A}^+$（使用 SVD 手算）。
# 2. 验证 $\mathbf{A}^+$ 满足 Penrose 四条条件。

# %% [markdown]
# ### 编程题
#
# 1. 实现函数 `pseudoinverse_svd(A, tol)`，使用 SVD 计算伪逆，忽略小于 `tol` 的奇异值。
# 2. 对比正规方程和 SVD 伪逆在 Hilbert 矩阵（高度病态）上的表现。
#
# > 答案见 `solutions/01_solutions.ipynb`

# %% [markdown]
# ## 15. 本节总结
#
# | 概念 | 定义 | 机器人学应用 |
# |------|------|-------------|
# | 内积 | $\mathbf{a}^T\mathbf{b}$ | 做功、投影、方向余弦 |
# | 叉乘矩阵 | $[\mathbf{v}]_\times$ | 角速度→线速度，so(3) |
# | SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | 伪逆、奇异性检测、可操作度 |
# | 伪逆 | $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$ | 数值 IK、冗余分解 |
# | 正定性 | $\forall\mathbf{x}\neq0: \mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ | M(q)性质、稳定性分析 |
#
# ### 核心公式速查
#
# - 叉乘矩阵：$[\mathbf{v}]_\times = \begin{bmatrix}0&-v_z&v_y\\v_z&0&-v_x\\-v_y&v_x&0\end{bmatrix}$
# - SVD 伪逆：$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$
# - 阻尼最小二乘：$\Delta\mathbf{q} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda^2\mathbf{I})^{-1}\Delta\mathbf{x}$

# %% [markdown]
# ## 16. 与下一节的联系
#
# 下一节（NB02）将使用齐次变换矩阵来描述**刚体在空间中的位置和姿态**。我们将看到：
# - 旋转矩阵 $\mathbf{R} \in SO(3)$ 如何嵌入 $4\times4$ 齐次变换矩阵
# - 叉乘矩阵 $[\boldsymbol{\omega}]_\times$ 如何出现在旋转矩阵的导数中
# - SVD 和伪逆如何为后续的数值 IK（NB06）和奇异性分析（NB08）做准备
