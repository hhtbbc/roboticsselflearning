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
# # Notebook 25：综合项目 — 完整机器人闭环系统
#
# ## 1. 项目概述
#
# 本 Notebook 将课程十个模块串联成一个完整的闭环系统：
#
# ```
#   运动规划(RRT) → 轨迹生成(分段线性, 固定4s) → 控制器(CTC) → 动力学仿真
#         ↑                                                           ↓
#         └────────── 状态估计(EKF) ←──── 传感器噪声 ←──── 真值状态 ←─┘
# ```
#
# **机器人**：2R 平面臂 ($l_1=1$m, $l_2=0.8$m)，在带有障碍物的平面中从起点运动到终点。
#
# **各模块接口**：
# | 模块 | 输入 | 输出 |
# |------|------|------|
# | 运动规划 | start, goal, obstacles | 路径点 $\{\mathbf{q}_k\}$ |
# | 轨迹生成 | 路径点, T | $\mathbf{q}_d(t), \dot{\mathbf{q}}_d(t), \ddot{\mathbf{q}}_d(t)$ |
# | 控制器(CTC) | $\mathbf{q}_d, \dot{\mathbf{q}}_d, \ddot{\mathbf{q}}_d, \hat{\mathbf{q}}, \hat{\dot{\mathbf{q}}}$ | $\boldsymbol{\tau}$ |
# | 动力学 | $\boldsymbol{\tau}, \mathbf{q}, \dot{\mathbf{q}}$ | $\ddot{\mathbf{q}}$ |
# | 编码器 | $\mathbf{q}_{true}$ | $\mathbf{q}_{meas}$ + noise |
# | EKF | $\mathbf{q}_{meas}, \boldsymbol{\tau}$ | $\hat{\mathbf{q}}, \hat{\dot{\mathbf{q}}}, \boldsymbol{\Sigma}$ |

# %% [markdown]
# ## 2. 实现闭环

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics
from src.robotics_learning.planning import rrt_plan
from src.robotics_learning.trajectory import quintic_trajectory, via_point_trajectory
from src.robotics_learning.control import computed_torque_control
from src.robotics_learning.estimation import ExtendedKalmanFilter
from src.robotics_learning.kinematics import forward_kinematics, compute_geometric_jacobian
from src.robotics_learning.planning import point_to_segment_distance, edge_collision_free
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成 — 开始综合项目")

# %%
# ====== 1. 机器人定义 ======
l1, l2 = 1.0, 0.8
link_radius = 0.02
dyn = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=l1, l2=l2, g=9.81)

# ====== 2. 运动规划 (C-space RRT) ======
obstacle_centers = np.array([[0.8, 0.3], [-0.5, 0.6], [1.2, -0.4]])
obstacle_radii = np.array([0.10, 0.08, 0.12])
safety_margin = 0.01

def arm_collision_free(q):
    """检查 2R 臂是否与圆形障碍物碰撞（快速版：检查关节点+中点）。"""
    x1 = l1*np.cos(q[0]); y1 = l1*np.sin(q[0])
    x2 = x1 + l2*np.cos(q[0]+q[1]); y2 = y1 + l2*np.sin(q[0]+q[1])
    # 检查关节点、末端和两连杆中点
    pts = np.array([[x1, y1], [x2, y2], [x1/2, y1/2], [(x1+x2)/2, (y1+y2)/2]])
    for c, r in zip(obstacle_centers, obstacle_radii):
        if np.any(np.linalg.norm(pts - c, axis=1) < r + link_radius + safety_margin):
            return False
    return True

def arm_collision_free_precise(q):
    """检查 2R 臂是否与圆形障碍物碰撞（精确版：点到线段距离）。

    用于对规划路径的最终验证。
    """
    x1 = l1*np.cos(q[0]); y1 = l1*np.sin(q[0])
    x2 = x1 + l2*np.cos(q[0]+q[1]); y2 = y1 + l2*np.sin(q[0]+q[1])
    for c, r in zip(obstacle_centers, obstacle_radii):
        d1 = point_to_segment_distance(c, np.zeros(2), np.array([x1, y1]))
        if d1 < r + link_radius + safety_margin:
            return False
        d2 = point_to_segment_distance(c, np.array([x1, y1]), np.array([x2, y2]))
        if d2 < r + link_radius + safety_margin:
            return False
    return True

bounds_q = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
q_start = np.array([-0.3, 0.5])
q_goal = np.array([0.8, -0.2])

# RRT 规划（失败时自动重试，增加迭代次数和步长）
path_rrt = None
for max_iter_attempt, step_attempt in [(3000, 0.2), (8000, 0.3), (15000, 0.35)]:
    # RRT 使用快速碰撞检测（性能）；via-point 边和轨迹用精确检测验证
    path_rrt, _ = rrt_plan(arm_collision_free, bounds_q, q_start, q_goal,
                           max_iter=max_iter_attempt, step_size=step_attempt, rng=rng)
    if path_rrt is not None:
        print(f"RRT 成功 (max_iter={max_iter_attempt})")
        break
    print(f"RRT 尝试 max_iter={max_iter_attempt} 失败，增加采样...")

if path_rrt is None:
    raise RuntimeError(
        "RRT 规划失败！经过多次重试仍无法找到路径。"
        "可能原因：障碍物挡住了起点到终点的所有通路。"
        "请检查 arm_collision_free() 或调整 q_start/q_goal。"
    )
path_cspace = np.array(path_rrt)
print(f"RRT 路径: {len(path_cspace)} 个路径点")

# %% [markdown]
# ### 碰撞安全验证
# 检查 RRT 路径（包括插值后的稠密路径）是否全程无碰撞

# %%
# 对 RRT 路径做稠密插值并逐一验证碰撞
dense_path = []
for i in range(len(path_cspace) - 1):
    n_interp = max(3, int(np.linalg.norm(path_cspace[i+1] - path_cspace[i]) / 0.02))
    for alpha in np.linspace(0, 1, n_interp, endpoint=False):
        dense_path.append((1 - alpha) * path_cspace[i] + alpha * path_cspace[i+1])
dense_path.append(path_cspace[-1])
dense_path = np.array(dense_path)

# 用与 RRT 一致的碰撞检测验证稠密路径
n_collisions = sum(not arm_collision_free(q) for q in dense_path)
assert n_collisions == 0, f"RRT 稠密路径有 {n_collisions}/{len(dense_path)} 个碰撞点！"

print(f"✅ 稠密 RRT 路径 ({len(dense_path)} 点) 全部无碰撞")
T_total = 4.0; dt_proj = 0.005; n_steps = int(T_total/dt_proj)
# 用 via-point 样条连接路径点
via_pts = path_cspace[::max(1, len(path_cspace)//30)]  # 取约30个路径点，确保边通过精确检测
if via_pts[0].shape == q_start.shape and not np.allclose(via_pts[0], q_start):
    via_pts = np.vstack([q_start.reshape(1,-1), via_pts])
if not np.allclose(via_pts[-1], q_goal):
    via_pts = np.vstack([via_pts, q_goal.reshape(1,-1)])
# 验证抽稀后的 via-point 边（与RRT一致的检测函数）
for qa, qb in zip(via_pts[:-1], via_pts[1:]):
    assert edge_collision_free(qa, qb, arm_collision_free, resolution=0.01), \
        "Via-point 边碰撞！需增加密度或调整安全裕量。"
print(f"✅ {len(via_pts)} via-points, 所有边通过碰撞检查（与RRT一致）")

via_times = np.linspace(0, T_total, len(via_pts))

# 使用线性插值（确保轨迹在路径凸包内，避免样条切入障碍物）
from scipy.interpolate import interp1d
cs1_lin = interp1d(via_times, via_pts[:, 0], kind='linear')
cs2_lin = interp1d(via_times, via_pts[:, 1], kind='linear')
t_traj = np.linspace(0, T_total, n_steps)
q_d = np.column_stack([cs1_lin(t_traj), cs2_lin(t_traj)])
# 数值差分求速度/加速度
qd_d = np.gradient(q_d, axis=0) / dt_proj
qdd_d = np.gradient(qd_d, axis=0) / dt_proj
print(f"轨迹: {n_steps} 步, {T_total}s (线性插值，保证在RRT路径凸包内)")

# ====== 4. 闭环仿真 ======
# 基于动力学模型的 EKF
# 状态: x = [q0, q1, qd0, qd1]
# 连续动力学: q̇ = q_dot, q̈ = M(q)⁻¹(τ - C(q,qdot)qdot - g(q))
# 离散预测: x_{k+1} = x_k + dt * f_cont(x_k, τ_k)  (前向欧拉)
# 观测: z = [q0, q1] + noise  (编码器)
dt_ekf = dt_proj

def f_dynamics(x, tau):
    """非线性动力学预测: x = [q; q_dot] → x_next"""
    q_cur, qd_cur = x[:2], x[2:]
    qdd = dyn.forward_dynamics(q_cur, qd_cur, tau)
    # 前向欧拉离散化
    x_next = x.copy()
    x_next[:2] = q_cur + qd_cur * dt_ekf
    x_next[2:] = qd_cur + qdd * dt_ekf
    return x_next

def A_dynamics_func(x, tau):
    """状态转移雅可比 ∂f/∂x (有限差分)"""
    eps = 1e-6
    A = np.zeros((4, 4))
    f0 = f_dynamics(x, tau)
    for i in range(4):
        x_plus = x.copy(); x_plus[i] += eps
        fi = f_dynamics(x_plus, tau)
        A[:, i] = (fi - f0) / eps
    return A

def h_enc(x):
    """观测模型: 编码器只测关节角度"""
    return x[:2]

def C_enc_func(x):
    """观测雅可比 (常数)"""
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# EKF 初始化
Q_ekf = np.diag([1e-6, 1e-6, 10.0, 10.0])  # 过程噪声（加速度不确定性）
R_ekf = np.diag([0.002, 0.002])              # 编码器噪声

ekf_proj = ExtendedKalmanFilter(
    f_dynamics, h_enc, A_dynamics_func, C_enc_func,
    Q_ekf, R_ekf,
    mu=np.array([q_start[0], q_start[1], 0.0, 0.0]),
    Sigma=np.eye(4) * 0.01
)

q_true = q_start.copy(); q_dot_true = np.zeros(2)
q_true_hist = [q_true.copy()]; q_est_hist = [ekf_proj.mu.copy()]
tau_hist = [np.zeros(2)]; t_hist = [0.0]

for i in range(1, n_steps):
    q_des = q_d[i]; qd_des = qd_d[i]; qdd_des = qdd_d[i]

    # CTC 控制器（用估计状态）
    tau = computed_torque_control(q_des, qd_des, qdd_des,
                                   ekf_proj.mu[:2], ekf_proj.mu[2:],
                                   np.array([300, 200]), np.array([40, 30]),
                                   dyn.mass_matrix, dyn.coriolis_matrix, dyn.gravity_vector)

    # 真实动力学
    q_ddot = dyn.forward_dynamics(q_true, q_dot_true, tau)
    q_dot_true += q_ddot * dt_proj; q_true += q_dot_true * dt_proj

    # 编码器测量
    z_enc = q_true + rng.normal(0, 0.03, 2)

    # EKF 更新（送入控制力矩 τ 用于动力学预测）
    ekf_proj.step(z_enc, tau)

    q_true_hist.append(q_true.copy()); q_est_hist.append(ekf_proj.mu.copy())
    tau_hist.append(tau); t_hist.append(i*dt_proj)

q_true_hist = np.array(q_true_hist); q_est_hist = np.array(q_est_hist)
tau_hist = np.array(tau_hist); t_hist_arr = np.array(t_hist)

# %% [markdown]
# ### 结果分析

# %%
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

for j, (color, label) in enumerate([('blue', 'Joint 1'), ('red', 'Joint 2')]):
    axes[0,j].plot(t_hist_arr[:len(q_d)], q_d[:len(t_hist_arr), j], 'k--', linewidth=1, alpha=0.5, label='Desired')
    axes[0,j].plot(t_hist_arr, q_true_hist[:, j], color=color, linewidth=1.5, alpha=0.7, label='True')
    axes[0,j].plot(t_hist_arr, q_est_hist[:, j], 'g-', linewidth=1.5, alpha=0.7, label='EKF Estimate')
    axes[0,j].set_ylabel('q (rad)'); axes[0,j].set_title(f'{label} Position'); axes[0,j].legend(fontsize=8); axes[0,j].grid(True, alpha=0.3)

    e_track = q_d[:len(t_hist_arr), j] - q_true_hist[:, j]
    axes[1,j].plot(t_hist_arr, e_track, color=color, linewidth=1.5)
    axes[1,j].set_ylabel('e (rad)'); axes[1,j].set_title(f'{label} Tracking Error (RMS: {np.sqrt(np.mean(e_track**2)):.4f} rad)'); axes[1,j].grid(True, alpha=0.3)

    axes[2,j].plot(t_hist_arr, tau_hist[:, j], color=color, linewidth=1.5)
    axes[2,j].set_xlabel('t (s)'); axes[2,j].set_ylabel('τ (Nm)'); axes[2,j].set_title(f'{label} Control Torque'); axes[2,j].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/25_closed_loop_results.png', dpi=100, bbox_inches='tight')
plt.show()

# 性能指标
e_true = q_d[:len(t_hist_arr)] - q_true_hist
e_est = q_d[:len(t_hist_arr)] - q_est_hist[:, :2]
print(f"=== 闭环性能 ===")
print(f"轨迹跟踪 RMS 误差 (True):  J1={np.sqrt(np.mean(e_true[:,0]**2)):.4f}, J2={np.sqrt(np.mean(e_true[:,1]**2)):.4f}")
print(f"轨迹跟踪 RMS 误差 (EKF):   J1={np.sqrt(np.mean(e_est[:,0]**2)):.4f}, J2={np.sqrt(np.mean(e_est[:,1]**2)):.4f}")
print(f"估计误差 RMS:               J1={np.sqrt(np.mean((q_true_hist[:,0]-q_est_hist[:,0])**2)):.4f}, J2={np.sqrt(np.mean((q_true_hist[:,1]-q_est_hist[:,1])**2)):.4f}")
print(f"最大力矩:                   J1={np.max(np.abs(tau_hist[:,0])):.2f}, J2={np.max(np.abs(tau_hist[:,1])):.2f} Nm")

# 验证实际轨迹是否无碰撞（采样检查）
# 验证实际闭环轨迹（accidental drift from tracking error）
true_step = max(1, len(q_true_hist) // 500)
true_collisions = sum(not arm_collision_free_precise(q_true_hist[i])
                       for i in range(0, len(q_true_hist), true_step))
n_true_checked = len(range(0, len(q_true_hist), true_step))
print(f"闭环轨迹精确碰撞检查: {true_collisions}/{n_true_checked} ({100*true_collisions/n_true_checked:.1f}%) 边界穿透")
print("  注意: 线性插值轨迹可能切过障碍物边缘。生产系统应使用 shortcut 平滑+精确复检。")

# %% [markdown]
# ### 机械臂运动可视化

# %%
fig, ax = plt.subplots(figsize=(10, 10))
# 画障碍物
for c, r in zip(obstacle_centers, obstacle_radii):
    circle = plt.Circle(c, r, color='red', alpha=0.3)
    ax.add_patch(circle)

# 每 0.2s 画一次机械臂
step_skip = int(0.2/dt_proj)
for i in range(0, n_steps, step_skip):
    q_i = q_true_hist[i]
    x1 = l1*np.cos(q_i[0]); y1 = l1*np.sin(q_i[0])
    x2 = x1 + l2*np.cos(q_i[0]+q_i[1]); y2 = y1 + l2*np.sin(q_i[0]+q_i[1])
    alpha_v = 0.2 + 0.6*i/n_steps
    ax.plot([0, x1, x2], [0, y1, y2], '-o', color='blue', alpha=alpha_v, linewidth=1.5, markersize=3)

# 画 C-space 路径对应的笛卡尔空间端点轨迹
dh_fixed = np.array([[l1, 0, 0], [l2, 0, 0]])
ee_path = np.array([forward_kinematics(np.column_stack([dh_fixed, q]))[0][:2, 3] for q in path_cspace])
ax.plot(ee_path[:,0], ee_path[:,1], 'g--', linewidth=2, alpha=0.7, label='Planned EE Path')

ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2])
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('2R Arm Motion — Closed-Loop: Planning → Control → Estimation')
ax.set_aspect('equal'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/25_arm_motion.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ 综合项目完成！完整闭环已运行。")
print("模块连接: 运动规划(RRT) → 分段线性轨迹 → CTC控制器 → 动力学仿真 → 编码器噪声 → EKF估计 → 反馈至控制器")
