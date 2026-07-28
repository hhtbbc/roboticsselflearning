"""
可视化工具 (Visualization Utilities)

所有函数返回 matplotlib figure/axes，不调用 plt.show()。
适合在 Jupyter Notebook 中使用 %matplotlib inline。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


# =============================================================================
# 坐标系与变换可视化
# =============================================================================

def draw_frame_3d(ax, T: np.ndarray, label: str = '',
                  scale: float = 1.0, alpha: float = 1.0):
    """在 3D 空间中画出坐标系（RGB=XYZ）"""
    origin = T[:3, 3]
    R = T[:3, :3]

    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        ax.quiver(origin[0], origin[1], origin[2],
                  R[0, i], R[1, i], R[2, i],
                  length=scale, color=color, alpha=alpha,
                  linewidth=2)

    if label:
        ax.text(origin[0], origin[1], origin[2], label, fontsize=10)


def draw_transform_chain(ax, transforms: List[np.ndarray],
                         labels: List[str] = None, scale: float = 0.5):
    """绘制变换链中的所有参考系"""
    if labels is None:
        labels = [str(i) for i in range(len(transforms))]

    for i, (T, label) in enumerate(zip(transforms, labels)):
        alpha = 1.0 - 0.3 * (len(transforms) - i) / len(transforms)
        draw_frame_3d(ax, T, label, scale=scale, alpha=alpha)


def setup_3d_axis(ax, xlim=(-2, 2), ylim=(-2, 2), zlim=(0, 3),
                  title='3D Coordinate Frames', equal_aspect=True):
    """设置 3D 轴的基本属性"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    if equal_aspect:
        ax.set_box_aspect([1, 1, 1])


# =============================================================================
# 机械臂可视化
# =============================================================================

def plot_2r_arm(ax, q: np.ndarray, l1: float, l2: float,
                color: str = 'blue', alpha: float = 1.0,
                show_joints: bool = True, show_ee: bool = True):
    """
    在 2D 平面上绘制 2R 机械臂

    参数:
        ax: matplotlib Axes
        q: (2,) 关节角 [θ1, θ2]
        l1, l2: 连杆长度
    """
    # 连杆 1
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    # 连杆 2
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])

    ax.plot([0, x1, x2], [0, y1, y2], '-o' if show_joints else '-',
            color=color, linewidth=3, alpha=alpha,
            markersize=8, markerfacecolor='white')

    if show_ee:
        ax.plot(x2, y2, 'r*', markersize=12, label='End-Effector')


def plot_3d_arm(ax, dh_table: np.ndarray, q: np.ndarray,
                color: str = 'blue', show_frames: bool = True):
    """
    在 3D 空间中绘制机械臂

    依赖 src.robotics_learning.kinematics.forward_kinematics
    """
    from .kinematics import forward_kinematics

    dh_full = np.column_stack([dh_table, q])
    _, transforms = forward_kinematics(dh_full)

    # 提取所有原点位置
    points = np.array([T[:3, 3] for T in transforms])

    # 画连杆
    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            '-o', color=color, linewidth=3, markersize=6)

    # 画基座
    ax.scatter([0], [0], [0], color='black', s=100, marker='s')

    # 画末端
    ax.scatter(*points[-1], color='red', s=100, marker='*')

    if show_frames:
        for i, T in enumerate(transforms):
            draw_frame_3d(ax, T, scale=0.3, alpha=0.5)


# =============================================================================
# 轨迹可视化
# =============================================================================

def plot_trajectory_1d(t, q, q_dot, q_ddot, q_jerk=None, title='Trajectory'):
    """单自由度轨迹的位移/速度/加速度/jerk 四合一图"""
    n_plots = 4 if q_jerk is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), sharex=True)

    axes[0].plot(t, q, 'b-', linewidth=2)
    axes[0].set_ylabel('Position $q$')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(t, q_dot, 'g-', linewidth=2)
    axes[1].set_ylabel('Velocity $\dot{q}$')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, q_ddot, 'r-', linewidth=2)
    axes[2].set_ylabel('Acceleration $\ddot{q}$')
    axes[2].set_xlabel('Time $t$ (s)')
    axes[2].grid(True, alpha=0.3)

    if q_jerk is not None:
        axes[3].plot(t, q_jerk, 'm-', linewidth=2)
        axes[3].set_ylabel('Jerk $\dddot{q}$')
        axes[3].set_xlabel('Time $t$ (s)')
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# =============================================================================
# 控制可视化
# =============================================================================

def plot_control_results(t, q_des, q_true, tau, labels=None):
    """控制结果对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    n_joints = q_des.shape[1]
    if labels is None:
        labels = [f'Joint {i+1}' for i in range(n_joints)]

    for j in range(n_joints):
        axes[0].plot(t, q_des[:, j], '--', linewidth=1.5, alpha=0.7)
        axes[0].plot(t, q_true[:, j], '-', linewidth=1.5,
                     label=f'{labels[j]} (actual)')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Trajectory Tracking')

    # 误差
    for j in range(n_joints):
        error = q_des[:, j] - q_true[:, j]
        axes[1].plot(t, error, linewidth=1.5, label=labels[j])
    axes[1].set_ylabel('Tracking Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 力矩
    for j in range(n_joints):
        axes[2].plot(t, tau[:, j], linewidth=1.5, label=labels[j])
    axes[2].set_ylabel('Torque $\\tau$ (Nm)')
    axes[2].set_xlabel('Time $t$ (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# =============================================================================
# 状态估计可视化
# =============================================================================

def plot_kf_results(t, x_true, z, x_est=None, Sigma=None, dim=0,
                    title='Kalman Filter Results'):
    """
    卡尔曼滤波结果图：真值 vs 测量 vs 估计 ±2σ
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t, x_true[:, dim], 'k-', linewidth=2, label='True')
    ax.plot(t, z[:, dim], 'r.', markersize=2, alpha=0.5, label='Measurement')

    if x_est is not None:
        ax.plot(t, x_est[:, dim], 'b-', linewidth=2, label='Estimate')

        if Sigma is not None:
            sigma_dim = np.sqrt(Sigma[:, dim, dim])
            ax.fill_between(t,
                            x_est[:, dim] - 2*sigma_dim,
                            x_est[:, dim] + 2*sigma_dim,
                            color='blue', alpha=0.15, label='$\\pm 2\\sigma$')

    ax.set_xlabel('Time')
    ax.set_ylabel(f'State dim {dim}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_particle_filter(ax, particles, weights, x_true=None,
                         goal=None, landmarks=None):
    """在 2D 平面绘制粒子滤波状态"""
    ax.clear()

    # 粒子（大小按权重）
    sizes = np.clip(weights * 5000, 1, 100)
    ax.scatter(particles[:, 0], particles[:, 1],
               s=sizes, c='blue', alpha=0.3, label='Particles')

    if x_true is not None:
        ax.scatter(*x_true[:2], c='green', s=200, marker='*',
                   label='True Position', zorder=5)

    if goal is not None:
        ax.scatter(*goal[:2], c='red', s=200, marker='x',
                   label='Goal', zorder=5)

    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 1],
                   c='orange', s=50, marker='s', label='Landmarks')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
