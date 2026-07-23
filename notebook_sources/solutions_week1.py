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
# # 第 1 周练习解答 — 数学基础与运动学 (NB01-06)
#
# > 建议：先独立尝试再做这些解答。

# %% [markdown]
# ## NB01 线性代数 — 解答

# %% [markdown]
# ### 概念题 1: $\mathbf{A}^T\mathbf{A}$ vs $\mathbf{A}\mathbf{A}^T$
#
# - $\mathbf{A} \in \mathbb{R}^{m\times n}$：
#   - $\mathbf{A}^T\mathbf{A} \in \mathbb{R}^{n\times n}$（$n \times n$ 方阵）
#   - $\mathbf{A}\mathbf{A}^T \in \mathbb{R}^{m\times m}$（$m \times m$ 方阵）
# - 当 $\text{rank}(\mathbf{A}) = n$ 时 $\mathbf{A}^T\mathbf{A}$ 可逆（满列秩）
# - 当 $\text{rank}(\mathbf{A}) = m$ 时 $\mathbf{A}\mathbf{A}^T$ 可逆（满行秩）

# %% [markdown]
# ### 编程题: Hilbert 矩阵 SVD vs 正规方程对比

# %%
import numpy as np
import sys; sys.path.insert(0, '..')

def hilbert(n): return np.fromfunction(lambda i,j: 1/(i+j+1), (n,n))

for n in [5, 10]:
    H = hilbert(n); b = np.ones(n)
    x_true = np.ones(n)  # H @ ones = known solution
    b = H @ x_true
    x_normal = np.linalg.solve(H.T @ H, H.T @ b)
    x_svd = np.linalg.pinv(H) @ b
    print(f"n={n}: Normal Error={np.linalg.norm(x_normal-x_true):.2e}, SVD Error={np.linalg.norm(x_svd-x_true):.2e}")

# %% [markdown]
# ## NB02 坐标系与刚体运动 — 解答

# %% [markdown]
# ### 手算题: ${}^{A}_{B}\mathbf{R} = R_x(90°)$, ${}^{A}\mathbf{p}_B=[2,0,0]^T$, ${}^{B}\mathbf{p}=[0,1,0]^T$
#
# $$R_x(90°) = \begin{bmatrix}1&0&0\\0&0&-1\\0&1&0\end{bmatrix}$$
# $${}^{A}\mathbf{p} = R_x(90°)\begin{bmatrix}0\\1\\0\end{bmatrix} + \begin{bmatrix}2\\0\\0\end{bmatrix} = \begin{bmatrix}0\\0\\1\end{bmatrix} + \begin{bmatrix}2\\0\\0\end{bmatrix} = \begin{bmatrix}2\\0\\1\end{bmatrix}$$

# %%
from src.robotics_learning.transforms import rot_x, homogenous_transform
R = rot_x(np.pi/2)
T = homogenous_transform(R, np.array([2,0,0]))
p_B = np.array([0,1,0,1])
p_A = T @ p_B
print(f"p_A = {p_A[:3]}")  # [2, 0, 1]

# %% [markdown]
# ## NB03 旋转表示 — 解答

# %% [markdown]
# ### 手算题: 绕 $\mathbf{k}=[1,0,0]^T$ 转 90° → 罗德里格斯公式
# $$\mathbf{R} = \mathbf{I} + \sin(90°)[\mathbf{k}]_\times + (1-\cos(90°))[\mathbf{k}]_\times^2$$
# $$= \mathbf{I} + \begin{bmatrix}0&0&0\\0&0&-1\\0&1&0\end{bmatrix} + \begin{bmatrix}0&0&0\\0&-1&0\\0&0&-1\end{bmatrix} = \begin{bmatrix}1&0&0\\0&0&-1\\0&1&0\end{bmatrix}$$

# %%
k = np.array([1.,0,0]); theta = np.pi/2
R = np.eye(3) + np.sin(theta)*np.array([[0,0,0],[0,0,-1],[0,1,0]]) + (1-np.cos(theta))*np.array([[0,0,0],[0,0,-1],[0,1,0]])@np.array([[0,0,0],[0,0,-1],[0,1,0]])
print(R)

# %% [markdown]
# ## NB04 四元数 — 解答

# %% [markdown]
# ### 手算题: slerp(t=0.5) between q1=(1,0,0,0) and q2=(0.707, 0.707, 0, 0)
#
# $\Omega = \arccos(0.707) = 45°$
# $$\text{slerp}(0.5) = \frac{\sin(22.5°)}{\sin(45°)}\mathbf{q}_1 + \frac{\sin(22.5°)}{\sin(45°)}\mathbf{q}_2 = (0.924, 0.383, 0, 0)$$

# %%
from src.robotics_learning.transforms import Quaternion, slerp
import numpy as np
q1 = Quaternion(1,0,0,0); q2 = Quaternion(0.707,0.707,0,0)
qs = slerp(q1, q2, 0.5)
print(f"slerp(0.5) = {qs}")  # (0.924, 0.383, 0, 0)

# %% [markdown]
# ## NB05 正运动学 — 解答

# %% [markdown]
# ### 手算题: 2R 臂 DH (l1=1, l2=0.8), q1=30°, q2=45°
# $$x = 1\cdot\cos30° + 0.8\cdot\cos75° = 0.866 + 0.207 = 1.073$$
# $$y = 1\cdot\sin30° + 0.8\cdot\sin75° = 0.500 + 0.773 = 1.273$$

# %%
from src.robotics_learning.kinematics import forward_kinematics
dh = np.array([[1.,0,0,np.radians(30)], [0.8,0,0,np.radians(45)]])
T,_ = forward_kinematics(dh)
print(f"p = {np.round(T[:2,3], 3)}")  # [1.073, 1.273]

# %% [markdown]
# ## NB06 逆运动学 — 解答

# %% [markdown]
# ### 概念题 1: 2R 臂最多 2 个 IK 解 (elbow up / elbow down)
# 无解条件: $r < |l_1-l_2|$ (太近) 或 $r > l_1+l_2$ (太远)

# %% [markdown]
# ### 手算题: l1=1, l2=0.8, (x,y)=(1.2,0.6)
# $r = \sqrt{1.2^2+0.6^2} = 1.3416$
# $\cos q_2 = (1.3416^2 - 1 - 0.64)/(2\cdot1\cdot0.8) = 0.2$
# $q_2 = \pm \arccos(0.2) = \pm 78.46°$

# %%
from src.robotics_learning.kinematics import ik_2r_geometric
sols = ik_2r_geometric(1., 0.8, 1.2, 0.6)
for i,(q1,q2) in enumerate(sols):
    print(f"Sol {i+1}: q1={np.degrees(q1):.1f}°, q2={np.degrees(q2):.1f}°")
