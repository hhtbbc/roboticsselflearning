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
#   运动规划(RRT+精确碰撞) → shortcut → C²三次样条轨迹(固定4s) → CTC(力矩限幅) → 动力学仿真
#         ↑                                                                        ↓
#         └────────── 状态估计(EKF) ←──── 编码器噪声 ←──── 真值状态 ←──────────────┘
# ```
#
# **机器人**：2R 平面臂 ($l_1=1$m, $l_2=0.8$m)，在带有障碍物的平面中从起点运动到终点。
#
# **各模块接口**：
# | 模块 | 输入 | 输出 |
# |------|------|------|
# | 运动规划 | start, goal, obstacles | 路径点 ${\mathbf{q}_k}$ |
# | Shortcut | 路径点, collision_fn | 精简路径 |
# | 轨迹生成 | 精简路径, T | $\mathbf{q}_d(t), \dot{\mathbf{q}}_d(t), \ddot{\mathbf{q}}_d(t)$ |
# | 控制器(CTC) | $\mathbf{q}_d, \dot{\mathbf{q}}_d, \ddot{\mathbf{q}}_d, \hat{\mathbf{q}}, \hat{\dot{\mathbf{q}}}$ | $\boldsymbol{\tau}$（限幅） |
# | 动力学 | $\boldsymbol{\tau}, \mathbf{q}, \dot{\mathbf{q}}$ | $\ddot{\mathbf{q}}$ |
# | 编码器 | $\mathbf{q}_{true}$ | $\mathbf{q}_{meas}$ + noise |
# | EKF | $\mathbf{q}_{meas}, \boldsymbol{\tau}$ | $\hat{\mathbf{q}}, \hat{\dot{\mathbf{q}}}, \boldsymbol{\Sigma}$ |

# %% [markdown]
# ## 2. 实现闭环

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
import sys
sys.path.insert(0, '..')
from src.robotics_learning.dynamics import TwoLinkArmDynamics
from src.robotics_learning.planning import (
    rrt_plan, edge_collision_free, point_to_segment_distance,
    wrap_to_pi
)
from src.robotics_learning.trajectory import quintic_trajectory
from src.robotics_learning.control import computed_torque_control
from src.robotics_learning.estimation import ExtendedKalmanFilter
from src.robotics_learning.kinematics import forward_kinematics, compute_geometric_jacobian
%matplotlib inline
rng = np.random.RandomState(42)
print("✅ 导入完成 — 开始综合项目")

# %%
# ====== 1. 机器人定义 ======
l1, l2 = 1.0, 0.8
link_radius = 0.015
dyn = TwoLinkArmDynamics(m1=1.0, m2=1.0, l1=l1, l2=l2, g=9.81)

# ====== 2. 运动规划 (C-space RRT，使用精确碰撞) ======
obstacle_centers = np.array([[0.8, 0.3], [-0.5, 0.6], [1.2, -0.4]])
obstacle_radii = np.array([0.06, 0.05, 0.08])
safety_margin = 0.003

def arm_collision_free_precise(q):
    """精确碰撞检测：基于连杆-障碍物线段距离"""
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
q_start = np.array([0.5, 0.3])
q_goal = np.array([1.0, 1.0])

# RRT 使用精确碰撞函数（2R 臂开销低，无需快速预检版本）
path_rrt = None
for max_iter_attempt, step_attempt in [(3000, 0.2), (8000, 0.3), (15000, 0.35)]:
    path_rrt, _ = rrt_plan(arm_collision_free_precise, bounds_q, q_start, q_goal,
                           max_iter=max_iter_attempt, step_size=step_attempt,
                           rng=rng, joint_types=['revolute', 'revolute'])
    if path_rrt is not None:
        print(f"RRT 成功 (max_iter={max_iter_attempt})")
        break
    print(f"RRT 尝试 max_iter={max_iter_attempt} 失败，增加采样...")

if path_rrt is None:
    raise RuntimeError(
        "RRT 规划失败！经过多次重试仍无法找到路径。"
        "可能原因：障碍物挡住了起点到终点的所有通路。"
        "请检查 arm_collision_free_precise() 或调整 q_start/q_goal。"
    )
path_cspace = np.array(path_rrt)
print(f"RRT 路径: {len(path_cspace)} 个路径点（全程精确碰撞检测）")

# %% [markdown]
# ### 碰撞感知 Shortcut — 精简路径同时保持无碰撞

# %%
def shortcut_path(path, collision_fn, resolution=0.02, n_attempts=200,
                  joint_types=None, rng=None):
    """碰撞感知 shortcut：随机选两节点，若直线边无碰撞则删除中间节点"""
    if rng is None:
        rng = np.random.RandomState()
    result = list(path)
    for _ in range(n_attempts):
        if len(result) <= 2:
            break
        i, j = sorted(rng.choice(len(result), size=2, replace=False))
        if j <= i + 1:
            continue
        if edge_collision_free(result[i], result[j], collision_fn,
                               resolution=resolution, joint_types=joint_types):
            result = result[:i + 1] + result[j:]
    return result

path_short = shortcut_path(path_cspace, arm_collision_free_precise,
                           resolution=0.01, n_attempts=500,
                           joint_types=['revolute', 'revolute'], rng=rng)
print(f"Shortcut: {len(path_cspace)} → {len(path_short)} 路径点 "
      f"({100*(1-len(path_short)/len(path_cspace)):.0f}% 精简)")
# 验证 shortcut 所有边无碰撞
for qa, qb in zip(path_short[:-1], path_short[1:]):
    assert edge_collision_free(np.array(qa), np.array(qb),
                               arm_collision_free_precise, resolution=0.005,
                               joint_types=['revolute', 'revolute']), \
        "Shortcut 边碰撞！"
print(f"✅ 所有 shortcut 边通过精确碰撞检查")

# %% [markdown]
# ### C² 连续轨迹生成 — 三次样条 + 精确碰撞复检
# 注意：当前使用固定总时长 4s 的自然三次样条，尚未接入路径时间参数化。
# 自然样条的端点条件为零二阶导数，不保证零边界速度。
# 如从静止启动，建议使用 clamped spline 或独立轨迹段。

# %%
T_total = 4.0; dt_proj = 0.005; n_steps = int(T_total/dt_proj)
via_time = np.linspace(0, T_total, len(path_short))
via_arr = np.array(path_short)

# 使用自然三次样条 (C² 连续，固定 4s 总时长；从静止启动建议用 clamped spline)
cs1_q = CubicSpline(via_time, via_arr[:, 0], bc_type='natural')
cs2_q = CubicSpline(via_time, via_arr[:, 1], bc_type='natural')

t_traj = np.linspace(0, T_total, n_steps)
q_d = np.column_stack([cs1_q(t_traj), cs2_q(t_traj)])
qd_d = np.column_stack([cs1_q(t_traj, 1), cs2_q(t_traj, 1)])
qdd_d = np.column_stack([cs1_q(t_traj, 2), cs2_q(t_traj, 2)])

# 平滑后精确碰撞复检 — 对期望轨迹全量验证
assert all(arm_collision_free_precise(q) for q in q_d), \
    "平滑后期望轨迹存在碰撞！需增加 via-point 密度。"
print(f"✅ 期望轨迹 ({len(q_d)} 步, {T_total}s, C²三次样条) 全量精确碰撞复检通过")

# 速度/加速度约束 — 逐关节验证
q_dot_max = np.array([4.0, 5.0])
q_ddot_max = np.array([15.0, 18.0])
assert np.all(np.abs(qd_d) <= 1.05 * q_dot_max[None, :]), \
    f"期望速度超限: max|q̇₁|={np.max(np.abs(qd_d[:,0])):.3f}, max|q̇₂|={np.max(np.abs(qd_d[:,1])):.3f}"
assert np.all(np.abs(qdd_d) <= 1.05 * q_ddot_max[None, :]), \
    f"期望加速度超限: max|q̈₁|={np.max(np.abs(qdd_d[:,0])):.3f}, max|q̈₂|={np.max(np.abs(qdd_d[:,1])):.3f}"
print(f"✅ 逐关节速度/加速度约束全部满足")

# ====== 4. 闭环仿真 ======
# EKF 过程噪声模型说明:
#   连续噪声模型: q̈ = M⁻¹(τ-Cq̇-g) + w_a, w_a ~ N(0, σ_a² I)
#   白噪声加速度 → 离散协方差:
#     Q_d = [[Δt⁴/4·σ_a²·I,  Δt³/2·σ_a²·I],
#            [Δt³/2·σ_a²·I,  Δt²·σ_a²·I]]
#   取 σ_a = 10 rad/s² (加速度噪声标准差), Δt=0.005s:
#     Δt⁴/4 ≈ 1.6e-10, Δt³/2 ≈ 6.3e-8, Δt² ≈ 2.5e-5
#   这些 Q 元素自动覆盖建模误差（q 通道 ~1e-10 → q̇ 通道 ~2.5e-3）
dt_ekf = dt_proj
sigma_a = 10.0  # rad/s² 加速度噪声标准差 (单位: rad/s², 方差 σ_a² 单位: (rad/s²)²)
Q_ekf = np.zeros((4, 4))
Q_ekf[:2, :2] = np.eye(2) * (dt_ekf**4 / 4) * sigma_a**2    # q-q 耦合
Q_ekf[:2, 2:] = np.eye(2) * (dt_ekf**3 / 2) * sigma_a**2    # q-q̇ 耦合
Q_ekf[2:, :2] = np.eye(2) * (dt_ekf**3 / 2) * sigma_a**2    # q̇-q 耦合
Q_ekf[2:, 2:] = np.eye(2) * (dt_ekf**2) * sigma_a**2        # q̇-q̇ 耦合
# 编码器噪声: σ_enc = 0.03 rad, R = σ_enc² I
R_ekf = np.diag([0.0009, 0.0009])  # 0.03²

print(f"EKF Q 矩阵 (σ_a={sigma_a}, dt={dt_ekf}): "
      f"q-var={Q_ekf[0,0]:.2e}, q̇-var={Q_ekf[2,2]:.2e}")
print(f"EKF R 矩阵 (σ_enc=0.03): diag({R_ekf[0,0]:.4f}, {R_ekf[1,1]:.4f})")

# 动力学模型函数
def f_dynamics(x, tau):
    """非线性动力学预测: x = [q; q_dot] → x_next"""
    q_cur, qd_cur = x[:2], x[2:]
    qdd = dyn.forward_dynamics(q_cur, qd_cur, tau)
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

    # CTC 控制器（用估计状态） + 力矩限幅
    tau = computed_torque_control(q_des, qd_des, qdd_des,
                                   ekf_proj.mu[:2], ekf_proj.mu[2:],
                                   np.array([300, 200]), np.array([40, 30]),
                                   dyn.mass_matrix, dyn.coriolis_matrix, dyn.gravity_vector)
    tau_max = 50.0
    tau = np.clip(tau, -tau_max, tau_max)

    # 真实动力学
    q_ddot = dyn.forward_dynamics(q_true, q_dot_true, tau)
    q_dot_true += q_ddot * dt_proj; q_true += q_dot_true * dt_proj

    # 编码器测量 (σ = 0.03 rad)
    z_enc = q_true + rng.normal(0, 0.03, 2)

    # EKF 更新
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
print(f"最大力矩 (限幅 {tau_max} Nm): J1={np.max(np.abs(tau_hist[:,0])):.2f}, J2={np.max(np.abs(tau_hist[:,1])):.2f} Nm")

# ====== 安全断言 ======
print(f"\n=== 安全验证 ===")
# 1. 期望轨迹全量无碰撞
assert all(arm_collision_free_precise(q) for q in q_d), \
    "期望轨迹存在碰撞！"
print(f"✅ 期望轨迹 ({len(q_d)} 点) 全量无碰撞")

# 2. 实际闭环轨迹 — 全量逐状态验证
assert all(arm_collision_free_precise(q) for q in q_true_hist), \
    "实际闭环轨迹存在碰撞！"
print(f"✅ 实际闭环轨迹 ({len(q_true_hist)} 状态) 全量无碰撞")

# 3. 逐边碰撞验证 — 检查每一条相邻状态边
for qa, qb in zip(q_true_hist[:-1], q_true_hist[1:]):
    assert edge_collision_free(qa, qb, arm_collision_free_precise,
                               resolution=0.005,
                               joint_types=['continuous', 'continuous']), \
        f"闭环边碰撞: q={qa} → {qb}"
print(f"✅ 闭环轨迹 {len(q_true_hist)-1} 条边全量碰撞检查通过")

# 4. 力矩限幅生效
assert np.max(np.abs(tau_hist)) <= tau_max + 1e-6, \
    f"力矩限幅违反: max|τ|={np.max(np.abs(tau_hist)):.3f} > {tau_max}"
print(f"✅ 力矩限幅 (+/-{tau_max} Nm) 生效")

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

print("\n✅ 综合项目完成 — 所有安全检查通过")
print("流水线: RRT(精确碰撞) → Shortcut → C²三次样条(固定4s) → CTC(力矩限幅) → 动力学 → 编码器(σ=0.03) → EKF(Q_d离散化)")
print("安全验证: 期望轨迹 + 实际闭环轨迹 + 全部连续边碰撞 + 力矩限幅 全部通过")
print("注意: 当前使用固定总时长，未接入路径时间参数化。如需约束最优时间请使用 topp_forward_backward_parameterization().")
