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
# # Notebook 24c：PnP、已知对应点云配准与一维约束图演示

# ## 1. 定位
# 连接视觉几何与状态估计的入门演示：
# - **PnP**：从 2D-3D 对应求解相机位姿（Gauss-Newton，已知对应）
# - **已知对应点云配准**：从 3D-3D 对应一步求解刚体变换（SVD，无最近邻搜索）
# - **一维约束图**：线性最小二乘回环校正（不含因子图/鲁棒核）
#
# 本 Notebook 是教学预览。完整实现（point-to-plane ICP、KD-tree 最近邻、RANSAC、非线性因子图）计划在后续独立章节完成。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ PnP：最小化重投影误差求解 $\mathbf{T}_{cw}$（已知 2D-3D 对应）
# - ⭐ 已知对应 SVD 配准：一步求解最优 $\mathbf{R}, \mathbf{t}$
# - ⭐ 一维约束图：线性最小二乘回环校正
# - 📖 RANSAC / point-to-plane / 非线性因子图（后续独立章节）

# %% [markdown]
# ## 3. Python — PnP 简化实现

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys; sys.path.insert(0, '..')
from src.robotics_learning.transforms import so3_exp, so3_log, axis_angle_to_rot
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成")

# %%
# 真值相机位姿
R_true = axis_angle_to_rot(np.array([0.2, -0.1, 0.3]), 0.5)
t_true = np.array([1.0, 0.5, 2.0])
K_pnp = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

# 3D 点 + 投影
n_pts = 15
pts_3d = rng.uniform(-2, 2, (n_pts, 3)) + np.array([0, 0, 5])
pts_2d = np.array([K_pnp @ (R_true @ p + t_true) for p in pts_3d])
pts_2d = pts_2d[:, :2] / pts_2d[:, 2, None]
pts_2d += rng.normal(0, 2, pts_2d.shape)  # 噪声

# PnP: 从 2D-3D 对应求解 R, t
# 注意：此简化版假设所有点均在相机前方 (z_c > 0)；生产系统需检查并拒绝 z_c ≤ 0 的点
def reprojection_error(params, pts_3d, pts_2d, K):
    omega = params[:3]; t = params[3:]
    R = so3_exp(omega)
    errors = []
    for p3, p2 in zip(pts_3d, pts_2d):
        pc = R @ p3 + t
        if pc[2] <= 1e-6:
            # 深度非正：点在相机后方/平面上 — 返回大残差且保留梯度
            # 使用 z_c 的 sigmoid 惩罚: 当 z_c→0⁻ 时残差平滑增大
            z_penalty = np.exp(-pc[2]) * 1e4  # z_c≪0 时巨大，z_c≥0 时小
            errors.extend([z_penalty, z_penalty])
            continue
        uv = K @ pc; uv = uv[:2]/uv[2]
        errors.extend(uv - p2)
    return np.array(errors)

res = least_squares(reprojection_error, np.zeros(6), args=(pts_3d, pts_2d, K_pnp))
R_est = so3_exp(res.x[:3]); t_est = res.x[3:]

print(f"R_true:\n{np.round(R_true, 4)}")
print(f"R_est:\n{np.round(R_est, 4)}")
print(f"t_true: {np.round(t_true, 4)}")
print(f"t_est:  {np.round(t_est, 4)}")
print(f"Rotation error: {np.linalg.norm(so3_log(R_est.T @ R_true)):.4f} rad")
print(f"Translation error: {np.linalg.norm(t_est - t_true):.4f} m")

# 重投影可视化
fig_rp, ax_rp = plt.subplots(figsize=(8, 6))
for p, R, t, c, label in [(pts_3d, R_true, t_true, 'green', 'True'), (pts_3d, R_est, t_est, 'blue', 'Estimated')]:
    for p3 in p:
        pc = R @ p3 + t
        if pc[2] > 0:
            uv = K_pnp @ pc; uv = uv[:2] / uv[2]
            ax_rp.scatter(*uv, c=c, s=20, alpha=0.6, label=label)
ax_rp.scatter(pts_2d[:, 0], pts_2d[:, 1], c='red', s=30, marker='+',
              linewidths=2, label='Measured')
handles, labels = ax_rp.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_rp.legend(by_label.values(), by_label.keys())
ax_rp.set_xlabel('u (px)'); ax_rp.set_ylabel('v (px)')
ax_rp.set_title('PnP — Reprojection'); ax_rp.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/24c_pnp_reprojection.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 3.1 已知对应点云配准 — 一步 SVD
#
# 当对应关系已知时，最优刚体变换可通过单次 SVD 求得（不需迭代）：
# 1. 中心化: ā = a - mean(A), b̄ = b - mean(B)
# 2. H = Ā^T B̄ → SVD: H = U Σ V^T
# 3. R* = V U^T (需确保 det=+1), t* = mean(B) - R* mean(A)

# %%
R_svd_true = axis_angle_to_rot(np.array([0, 0, 1]), 0.3)
t_svd_true = np.array([0.5, 0.2, 0.0])
cloud_A = rng.uniform(-1, 1, (100, 3))
cloud_B = np.array([R_svd_true @ p + t_svd_true + rng.normal(0, 0.02, 3) for p in cloud_A])

# 一步 SVD 求解
centroid_A = np.mean(cloud_A, axis=0)
centroid_B = np.mean(cloud_B, axis=0)
A_centered = cloud_A - centroid_A
B_centered = cloud_B - centroid_B
H = A_centered.T @ B_centered
U, _, Vt = np.linalg.svd(H)
R_svd = Vt.T @ U.T
if np.linalg.det(R_svd) < 0:
    Vt[-1] *= -1; R_svd = Vt.T @ U.T
t_svd = centroid_B - R_svd @ centroid_A

# 对齐验证
A_aligned = np.array([R_svd @ p + t_svd for p in cloud_A])
print(f"一步 SVD 配准: R error = {np.linalg.norm(so3_log(R_svd.T @ R_svd_true)):.4f} rad, "
      f"t error = {np.linalg.norm(t_svd - t_svd_true):.4f} m")

# 对比迭代式 ICP (实际应用需 KD-tree 最近邻 + 距离门限 + 外点剔除)
R_icp = np.eye(3); t_icp = np.zeros(3)
err_icp_hist = []

for it in range(30):
    A_transformed = np.array([R_icp @ p + t_icp for p in cloud_A])
    # 这里使用已知对应（教学简化）；真正 ICP 需要最近邻关联
    residuals = cloud_B - A_transformed
    err_icp_hist.append(np.mean(np.linalg.norm(residuals, axis=1)))
    H_icp = (A_transformed - np.mean(A_transformed, 0)).T @ (cloud_B - np.mean(cloud_B, 0))
    U_i, _, Vt_i = np.linalg.svd(H_icp)
    R_delta = Vt_i.T @ U_i.T
    if np.linalg.det(R_delta) < 0:
        Vt_i[-1] *= -1; R_delta = Vt_i.T @ U_i.T
    t_delta = np.mean(cloud_B, 0) - R_delta @ np.mean(A_transformed, 0)
    R_icp = R_delta @ R_icp
    t_icp = R_delta @ t_icp + t_delta
    if it > 2 and abs(err_icp_hist[-1] - err_icp_hist[-2]) < 1e-8:
        break

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(cloud_A[:,0], cloud_A[:,1], c='blue', s=10, alpha=0.5, label='Source (A)')
axes[0].scatter(cloud_B[:,0], cloud_B[:,1], c='red', s=10, alpha=0.5, label='Target (B)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].set_aspect('equal')
axes[0].set_title('Point Clouds Before ICP'); axes[0].legend()
axes[1].semilogy(err_icp_hist, 'b-o', linewidth=2)
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Mean Residual (m)')
axes[1].set_title('ICP Convergence'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/24c_icp.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"ICP: R error = {np.linalg.norm(so3_log(R_icp.T @ R_icp_true)):.4f} rad, t error = {np.linalg.norm(t_icp - t_icp_true):.4f} m")

# %% [markdown]
# ### 因子图 SLAM — 位姿图优化

# %%
# 1D 环形位姿图：5 个节点，odometry 边 + 1 个回环边
n_poses = 5
poses_true = np.arange(n_poses, dtype=float)
poses_true = np.concatenate([poses_true, [poses_true[0]]])  # 回环

# Odometry 测量（带噪声）
odom = np.diff(poses_true[:n_poses]) + rng.normal(0, 0.1, n_poses-1)
# 优化前：里程计累加（漂移）
poses_odom = np.zeros(n_poses)
poses_odom[0] = 0
for i in range(1, n_poses):
    poses_odom[i] = poses_odom[i-1] + odom[i-1]

# 回环测量
loop_closure = poses_true[-1] - poses_true[n_poses-1] + rng.normal(0, 0.05)

# 因子图优化: min Σ_w · ||x_j - x_i - z_{ij}||²
# 以 ||Ax - b||² 形式: 对每条边, A 中 x_j 的系数为 +√w, x_i 的系数为 -√w
n_edges = (n_poses - 1) + 1  # odom edges + loop closure
A_pose = np.zeros((n_edges, n_poses))
b_pose = np.zeros(n_edges)
row = 0

# Odometry edges: x_{i+1} - x_i = odom_i (weight=1)
for i in range(n_poses - 1):
    A_pose[row, i] = -1.0; A_pose[row, i+1] = 1.0
    b_pose[row] = odom[i]
    row += 1

# Loop closure: x_0 - x_{n-1} = loop_closure (higher weight)
w_lc = 4.0  # 回环权重 > 里程计
A_pose[row, n_poses-1] = -np.sqrt(w_lc)
A_pose[row, 0] = np.sqrt(w_lc)
b_pose[row] = np.sqrt(w_lc) * loop_closure

# 固定 anchor: x_0 = 0 (硬约束，通过加一行大权重)
A_anchor = np.zeros((1, n_poses)); A_anchor[0, 0] = 1000.0
b_anchor = np.zeros(1)
A_aug = np.vstack([A_pose, A_anchor])
b_aug = np.concatenate([b_pose, b_anchor])

# Solve: A^T A x = A^T b
H = A_aug.T @ A_aug; rhs = A_aug.T @ b_aug
poses_opt = np.linalg.solve(H, rhs)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(poses_true[:n_poses], 'ko-', linewidth=2, markersize=10, label='Ground Truth')
ax.plot(poses_odom, 'r--o', linewidth=2, label='Odometry (drift)')
ax.plot(poses_opt, 'g-o', linewidth=2, label='Pose Graph Optimized')
ax.set_xlabel('Pose Index'); ax.set_ylabel('Position')
ax.set_title('1D Pose Graph SLAM — Loop Closure Correction'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/24c_pose_graph_slam.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"Odom 最大漂移: {np.max(np.abs(poses_odom-poses_true[:n_poses])):.2f}")
print(f"优化后最大误差: {np.max(np.abs(poses_opt-poses_true[:n_poses])):.4f}")

# %% [markdown]
# ## 4. 练习题
# 1. PnP 和 ICP 分别解决什么问题？输入输出各是什么？
# 2. 因子图优化中为什么需要 anchor 节点？
# 3. 回环检测对 SLAM 精度的贡献？
