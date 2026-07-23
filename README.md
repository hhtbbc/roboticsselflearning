# 机器人学系统学习教程 (Robotics Self-Learning Course)

## 概述

本课程是一套以 **Jupyter Notebook** 为主要载体的机器人学系统学习教程，目标是在 **30 天**内建立完整、严谨、可用于工程实践和面试的机器人学知识体系。

## 课程设计理念

课程沿四条主线组织，覆盖十个核心模块：

| 主线 | 核心模块 |
|------|----------|
| **建模 (Modeling)** | 运动学数学基础、正/逆运动学、雅可比与静力学、动力学建模 |
| **规划 (Planning)** | 轨迹规划与时间参数化、运动规划 |
| **控制 (Control)** | 关节空间控制、操作空间控制与力控制 |
| **感知与状态估计 (Perception & Estimation)** | 状态估计与多传感器融合 |

## 适用人群

- 机器人学接近零基础，但希望系统建立知识体系
- 有 Python 编程基础
- 希望准备机器人算法岗位面试
- 希望在有限时间内掌握机器人学最硬核的内容

## 快速开始

### 1. 环境准备

```bash
cd /workspace/data/vggt-omega/roboticsselflearning
uv sync
uv run python -m ipykernel install --user --name robotics-learning --display-name "Python (robotics-learning)"
```

### 2. 启动 Jupyter

```bash
uv run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

然后在 VSCode 中打开 `notebooks/` 目录下的 `.ipynb` 文件，选择内核 `Python (robotics-learning)`。

### 3. 学习路径

按 Notebook 编号顺序学习：

- **Week 1 (Notebook 01-06)**：数学基础 + 运动学
- **Week 2 (Notebook 07-12)**：雅可比 + 奇异性 + 动力学
- **Week 3 (Notebook 13-18)**：轨迹规划 + 运动规划 + 关节控制
- **Week 4 (Notebook 19-27)**：操作空间控制 + 状态估计 + 综合项目 + 面试复习

## 内容分级

每个 Notebook 使用三级标记：

- **⭐ 必须掌握**：核心推导和代码，必须理解并能独立推导/实现
- **📖 需要理解**：重要的背景知识和工程实践，需理解但不必背诵
- **📚 拓展阅读**：进阶内容，初学可跳过，后续深入学习时参考

## 符号约定

本课程遵循以下符号约定：

- 标量：斜体小写 $x, y, \theta$
- 向量：粗体小写 $\mathbf{v}, \boldsymbol{\omega}$
- 矩阵：粗体大写 $\mathbf{R}, \mathbf{T}, \mathbf{J}$
- 参考系：上标表示，如 ${}^{A}\mathbf{v}$ 表示向量 $\mathbf{v}$ 在参考系 $\{A\}$ 中的表示
- 变换矩阵：${}^{A}_{B}\mathbf{T}$ 表示从系 $\{B\}$ 到系 $\{A\}$ 的齐次变换

## 推荐教材

1. **John J. Craig** - *Introduction to Robotics: Mechanics and Control* (经典入门)
2. **Siciliano et al.** - *Robotics: Modelling, Planning and Control* (综合参考)
3. **Lynch & Park** - *Modern Robotics: Mechanics, Planning, and Control* (现代视角，含代码)
4. **Sciavicco & Siciliano** - *Modelling and Control of Robot Manipulators* (控制深入)
5. **Thrun, Burgard & Fox** - *Probabilistic Robotics* (状态估计圣经)
6. **LaValle** - *Planning Algorithms* (运动规划参考)

## 在线资源

- [Modern Robotics 配套视频](https://modernrobotics.northwestern.edu/nu-gm-book/)
- [Peter Corke's Robotics Toolbox](https://github.com/petercorke/robotics-toolbox-python)
- [Underactuated Robotics (MIT)](http://underactuated.csail.mit.edu/)

## 项目结构

```
roboticsselflearning/
├── README.md                    ← 本文件
├── COURSE_PLAN.md               ← 30 天学习安排
├── KNOWLEDGE_MAP.md             ← 知识依赖关系图
├── PROGRESS.md                  ← 学习进度追踪
├── INTERVIEW_CHECKLIST.md       ← 面试题整理
├── notebooks/                   ← 学习用 .ipynb 文件
├── notebook_sources/            ← Jupytext .py 源文件
├── src/robotics_learning/       ← 可复用 Python 工具库
├── exercises/                   ← 练习题
├── solutions/                   ← 解答
├── outputs/                     ← 生成的图表和动画
├── projects/                    ← 综合项目
└── scripts/                     ← 验证和同步脚本
```

## Notebook 索引

| # | 文件名 | 标题 | 核心内容 | 天 |
|---|--------|------|----------|:--:|
| 00 | 00_course_guide | 课程导航指南 | 课程结构、使用说明、符号约定 | - |
| 01 | 01_linear_algebra_review | 线性代数复习 | 向量、矩阵、SVD、伪逆、最小二乘 | 1 |
| 02 | 02_coordinate_frames | 坐标系与刚体运动 | 齐次变换、基变换、主动/被动旋转 | 2 |
| 03 | 03_rotation_representations | 旋转的多种表示 | 欧拉角、轴角、万向锁、罗德里格斯公式 | 3 |
| 04 | 04_quaternions_lie | 四元数与李群基础 | 单位四元数、SO(3)/SE(3)、指数/对数映射 | 4 |
| 05 | 05_forward_kinematics | 正运动学 (FK) | DH 参数、FK 计算、ipywidgets 交互臂 | 5 |
| 06 | 06_inverse_kinematics | 逆运动学 (IK) | 几何法、代数法、数值法、DLS | 6 |
| 07 | 07_jacobian_statics | 雅可比与静力学 | 几何/解析雅可比、速度/力传播、虚功原理 | 8 |
| 08 | 08_singularity_manipulability | 奇异性与可操作度 | 可操作度椭球、条件数、奇异预警 | 9 |
| 09 | 09_10_dynamics | 动力学与拉格朗日法 | M/C/g 结构、2R 臂推导、SymPy 验证 | 10 |
| 10 | 11_newton_euler | 牛顿-欧拉动力学 | RNEA 递推、O(n) 复杂度、CRBA | 11 |
| 11 | 12_advanced_dynamics | 高级动力学专题 | 参数辨识 τ=Yθ、约束、接触、浮动基 | 12 |
| 12 | 13_trajectory_generation | 轨迹生成 | 梯形/五次多项式/样条、Path vs Trajectory | 15 |
| 13 | 14_time_parameterization | 时间参数化 | TOPP、(s,ṡ) 相平面、MVC | 16 |
| 14 | 15_motion_planning_basics | 运动规划基础 | C-空间、A*、Dijkstra、势场法 | 17 |
| 15 | 16_sampling_planning | 采样运动规划 | PRM、RRT、RRT*、2R 臂 C-space 规划 | 18 |
| 16 | 17_joint_control | 关节空间控制 | PID、重力补偿 PD、李雅普诺夫稳定性证明 | 19 |
| 17 | 18_computed_torque | 计算力矩控制 | CTC、反馈线性化、滑模控制 | 20 |
| 18 | 19_operational_space | 操作空间控制 | 零空间投影、RMRC、任务优先级 | 22 |
| 19 | 20_force_impedance | 力/阻抗/导纳控制 | 阻抗 vs 导纳、混合力控、无源性 | 23 |
| 20 | 21_state_estimation_basics | 概率与状态估计基础 | 贝叶斯滤波、高斯分布、协方差传播 | 24 |
| 21 | 22_kalman_filter | 卡尔曼滤波 (KF) | 预测-更新、卡尔曼增益、可观测性 | 25 |
| 22 | 23_ekf_particle_filter | EKF 与粒子滤波 | EKF 线性化、SIR 粒子滤波、全局定位 | 26 |
| 23 | 24_sensor_fusion | 传感器模型与融合 | IMU/编码器/视觉、多传感器异步 EKF | 27 |
| 24 | 25_integrated_project | 综合项目 | 完整闭环：RRT→轨迹→CTC→EKF→仿真 | 28-29 |
| 25 | 26_interview_review | 面试系统复习 | 50+ 题、核心公式速查表、自测框架 | 30 |
| — | solutions_week1~4 | 练习解答 (4 册) | 全部练习题的手算+编程答案 | — |

---

🤖 本课程由 Claude Code 辅助创建。欢迎反馈和改进建议。
