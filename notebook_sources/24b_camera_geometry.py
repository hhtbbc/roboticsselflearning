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
# # Notebook 24b：相机几何与针孔模型

# ## 1. 定位
# 相机是机器人感知的核心传感器。本节建立从 3D 世界点到 2D 像素的完整投影链，包括内参、外参和畸变。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ 针孔相机模型：$\tilde{\mathbf{u}} = \mathbf{K}[\mathbf{R}|\mathbf{t}]\tilde{\mathbf{X}}$
# - ⭐ 内参矩阵 K (fx, fy, cx, cy) 的含义
# - ⭐ 径向畸变与切向畸变
# - ⭐ 重投影误差
# - 📖 双目几何、本质矩阵、基础矩阵

# %% [markdown]
# ## 3. 针孔模型 ⭐

# %% [markdown]
# 3D 点 $\mathbf{X} = [X, Y, Z]^T$（相机系）→ 像素 $[u, v]^T$：
# $$u = f_x\frac{X}{Z} + c_x, \quad v = f_y\frac{Y}{Z} + c_y$$
#
# 齐次形式：
# $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$
#
# 含外参（世界系→相机系）：
# $$\tilde{\mathbf{u}} = \mathbf{K}[\mathbf{R}|\mathbf{t}]\tilde{\mathbf{X}}_W$$

# %% [markdown]
# ## 4. Python 实现

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '..')
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %%
# 相机内参
fx, fy, cx, cy = 500, 500, 320, 240
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 外参：相机在 (2,1,3)，看向原点
t_vec = np.array([2.0, 1.0, 3.0])
R_cw = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # 简化：相机看向原点
T_cw = np.column_stack([R_cw, t_vec])

# 3D 世界点（立方体顶点 + 随机点）
points_3d_w = np.array([
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]
], dtype=float)
points_3d_w = np.vstack([points_3d_w, rng.uniform(-0.5, 1.5, (20, 3))])

# 投影
def project(X_w, K, R, t):
    X_c = R @ X_w + t
    uv_h = K @ X_c
    return uv_h[:2] / uv_h[2]

pixels = np.array([project(p, K, R_cw, t_vec) for p in points_3d_w])

# 加噪声模拟真实测量
pixels_noisy = pixels + rng.normal(0, 2, pixels.shape)

# 重投影误差
pixels_back = np.array([project(p, K, R_cw, t_vec) for p in points_3d_w])
errors = np.linalg.norm(pixels_noisy - pixels_back, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(pixels[:,0], pixels[:,1], c='blue', s=30, label='Projected (noiseless)')
axes[0].scatter(pixels_noisy[:,0], pixels_noisy[:,1], c='red', s=10, alpha=0.5, marker='x', label='Noisy')
axes[0].set_xlim([0, 640]); axes[0].set_ylim([480, 0])  # 图像坐标系 y 朝下
axes[0].set_xlabel('u (px)'); axes[0].set_ylabel('v (px)')
axes[0].set_title('Camera Projection — 640×480'); axes[0].set_aspect('equal')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].hist(errors, bins=15, color='blue', alpha=0.7, edgecolor='black')
axes[1].axvline(x=np.mean(errors), color='red', linestyle='--', label=f'Mean={np.mean(errors):.1f}px')
axes[1].set_xlabel('Reprojection Error (px)'); axes[1].set_ylabel('Count')
axes[1].set_title('Reprojection Error Distribution'); axes[1].legend()
plt.tight_layout()
plt.savefig('../outputs/24b_camera_projection.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.1 畸变模型

# %%
def distort_radial(uv, k1, k2, K):
    """径向畸变: u_distorted = u(1 + k1*r² + k2*r⁴)"""
    uv_norm = (uv - np.array([K[0,2], K[1,2]])) / np.array([K[0,0], K[1,1]])
    r2 = np.sum(uv_norm**2, axis=1)
    factor = 1 + k1 * r2 + k2 * r2**2
    uv_dist_norm = uv_norm * factor[:, None]
    return uv_dist_norm * np.array([K[0,0], K[1,1]]) + np.array([K[0,2], K[1,2]])

k1, k2 = -0.3, 0.1
pixels_distorted = distort_radial(pixels, k1, k2, K)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(pixels[:,0], pixels[:,1], c='blue', s=20, label='Ideal')
ax.scatter(pixels_distorted[:,0], pixels_distorted[:,1], c='red', s=20, label=f'Distorted (k1={k1}, k2={k2})')
for i in range(len(pixels)):
    ax.plot([pixels[i,0], pixels_distorted[i,0]], [pixels[i,1], pixels_distorted[i,1]], 'k-', alpha=0.2, linewidth=0.5)
ax.set_xlim([0, 640]); ax.set_ylim([480, 0])
ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')
ax.set_title('Radial Distortion Effect'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/24b_radial_distortion.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. 练习题
# 1. fx 和 fy 的物理含义？为什么它们通常不相等？
# 2. 重投影误差在 Bundle Adjustment 中的角色？
