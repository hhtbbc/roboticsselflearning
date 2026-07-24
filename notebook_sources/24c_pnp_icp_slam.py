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
# # Notebook 24c：PnP、ICP 与因子图 SLAM 基础

# ## 1. 定位
# 连接视觉几何与状态估计：PnP 从 2D-3D 对应求解相机位姿，ICP 从 3D-3D 对应求解刚体变换，因子图将它们统一为非线性最小二乘。

# %% [markdown]
# ## 2. 学习目标
# - ⭐ PnP：最小化重投影误差求解 $\mathbf{T}_{cw}$
# - ⭐ ICP：point-to-point 和 point-to-plane
# - ⭐ 因子图：位姿图优化 = 稀疏非线性最小二乘
# - 📖 RANSAC 外点剔除

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
def reprojection_error(params, pts_3d, pts_2d, K):
    omega = params[:3]; t = params[3:]
    R = so3_exp(omega)
    errors = []
    for p3, p2 in zip(pts_3d, pts_2d):
        pc = R @ p3 + t
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
for p, R, t, c, label in [(pts_3d, R_true, t_true, 'green', 'True'), (pts_3d, R_est, t_est, 'blue', 'Estimated')]:
    pass  # 略过在此

# %% [markdown]
# ### ICP — Point-to-Point

# %%
# 生成两个点云（真值变换 + 噪声）
R_icp_true = axis_angle_to_rot(np.array([0, 0, 1]), 0.3)
t_icp_true = np.array([0.5, 0.2, 0.0])
cloud_A = rng.uniform(-1, 1, (100, 3))
cloud_B = np.array([R_icp_true @ p + t_icp_true + rng.normal(0, 0.02, 3) for p in cloud_A])

# ICP 迭代
R_icp = np.eye(3); t_icp = np.zeros(3)
err_icp_hist = []
for it in range(20):
    # 1. 最近邻关联 (简化：已知对应)
    A_matched = cloud_A
    B_target = np.array([R_icp @ p + t_icp for p in cloud_A])
    residuals = cloud_B - B_target

    # 2. 最小化 Σ||Rp_i + t - q_i||² → SVD 求最优 R, t
    centroid_A = np.mean(cloud_A, axis=0)
    centroid_B = np.mean(cloud_B, axis=0)
    H = (cloud_A - centroid_A).T @ (cloud_B - centroid_B)
    U, _, Vt = np.linalg.svd(H)
    R_delta = Vt.T @ U.T
    if np.linalg.det(R_delta) < 0:
        Vt[-1] *= -1; R_delta = Vt.T @ U.T
    t_delta = centroid_B - R_delta @ centroid_A

    R_icp = R_delta @ R_icp
    t_icp = R_delta @ t_icp + t_delta
    err_icp_hist.append(np.mean(np.linalg.norm(residuals, axis=1)))

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

# 因子图优化：min Σ(pose_j - pose_i - meas_{ij})²
# 即 solve A x = b
A_pose = np.zeros((n_poses, n_poses)); b_pose = np.zeros(n_poses)
A_pose[0, 0] = 1; b_pose[0] = 0  # anchor

for i in range(n_poses - 1):
    A_pose[i, i] += 1; A_pose[i, i+1] += -1; b_pose[i] += odom[i]
    A_pose[i+1, i+1] += 1; A_pose[i+1, i] += -1; b_pose[i+1] += -odom[i]

# 回环边: pose[0] - pose[n-1] = loop_closure
A_pose[0, 0] += 1; A_pose[0, n_poses-1] += -1; b_pose[0] += loop_closure
A_pose[n_poses-1, n_poses-1] += 1; A_pose[n_poses-1, 0] += -1; b_pose[n_poses-1] += -loop_closure

# 固定第一个位姿
A_pose[0, :] = 0; A_pose[0, 0] = 1; b_pose[0] = 0

poses_opt = np.linalg.solve(A_pose, b_pose)

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
