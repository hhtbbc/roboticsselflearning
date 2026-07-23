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
#   运动规划(RRT) → 轨迹生成(五次多项式) → 时间参数化 → 控制器(CTC) → 动力学仿真
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
from src.robotics_learning.estimation import KalmanFilter
from src.robotics_learning.kinematics import forward_kinematics
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成 — 开始综合项目")

# %%
# ====== 1. 机器人定义 ======
l1, l2 = 1.0, 0.8
dyn = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=l1, l2=l2, g=9.81)

# ====== 2. 运动规划 (C-space RRT) ======
obstacle_centers = np.array([[0.8, 0.3], [-0.5, 0.6], [1.2, -0.4]])
obstacle_radii = np.array([0.15, 0.12, 0.18])

def arm_collision_free(q):
    x1 = l1*np.cos(q[0]); y1 = l1*np.sin(q[0])
    x2 = x1 + l2*np.cos(q[0]+q[1]); y2 = y1 + l2*np.sin(q[0]+q[1])
    pts = np.array([[x1,y1],[x2,y2],[(x1+x2)/2,(y1+y2)/2]])
    for c, r in zip(obstacle_centers, obstacle_radii):
        if np.any(np.linalg.norm(pts - c, axis=1) < r + 0.08):
            return False
    return True

bounds_q = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
q_start = np.array([np.pi/4, -np.pi/3])
q_goal = np.array([-np.pi/3, np.pi/6])

path_rrt, _ = rrt_plan(arm_collision_free, bounds_q, q_start, q_goal,
                        max_iter=2000, step_size=0.15, rng=rng)
if path_rrt is None:
    path_rrt = [q_start, q_goal]  # fallback
path_cspace = np.array(path_rrt)
print(f"RRT 路径: {len(path_cspace)} 个路径点")

# ====== 3. 轨迹生成 ======
T_total = 4.0; dt_proj = 0.005; n_steps = int(T_total/dt_proj)

# 用 via-point 样条连接路径点
via_pts = path_cspace[::max(1, len(path_cspace)//5)]  # 取5个路径点
if via_pts[0].shape == q_start.shape and not np.allclose(via_pts[0], q_start):
    via_pts = np.vstack([q_start.reshape(1,-1), via_pts])
if not np.allclose(via_pts[-1], q_goal):
    via_pts = np.vstack([via_pts, q_goal.reshape(1,-1)])
via_times = np.linspace(0, T_total, len(via_pts))

t_traj, q_d, qd_d, qdd_d = via_point_trajectory(via_pts, via_times, dt_proj)
print(f"轨迹: {n_steps} 步, {T_total}s")

# ====== 4. 闭环仿真 ======
# EKF 状态: [q0, q1, qd0, qd1]
dt_ekf = dt_proj
A_ekf = np.eye(4)  # 简化的恒速模型（实际应包含动力学）
for i in range(2):
    A_ekf[i, i+2] = dt_ekf
C_ekf = np.array([[1,0,0,0],[0,1,0,0]])  # 只观测位置
Q_ekf = np.diag([1e-6, 1e-6, 0.05, 0.05])  # 速度过程噪声
R_ekf = np.diag([0.002, 0.002])  # 编码器噪声

ekf_proj = KalmanFilter(A_ekf, np.zeros((4,1)), C_ekf, Q_ekf, R_ekf,
                        mu=np.array([*q_start, 0, 0]), Sigma=np.eye(4)*0.01)

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

    # EKF 更新
    ekf_proj.predict(); ekf_proj.update(z_enc)

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
print("模块连接: 运动规划(RRT) → 五次样条轨迹 → CTC控制器 → 动力学仿真 → 编码器噪声 → EKF估计 → 反馈至控制器")
