# CONVENTIONS.md — 数学和坐标系约定

本课程统一采用以下约定，除非在特定 Notebook 中另有说明。

## 1. 向量和矩阵

- 列向量优先：所有向量默认为列向量
- 标量：斜体 $x, y, \theta$
- 向量：粗体小写 $\mathbf{v}, \boldsymbol{\omega}$
- 矩阵：粗体大写 $\mathbf{R}, \mathbf{T}, \mathbf{J}$

## 2. 旋转和齐次变换

- ${}^{A}_{B}\mathbf{R}$：从 $\{B\}$ 到 $\{A\}$ 的旋转矩阵（列是 $\{B\}$ 的轴在 $\{A\}$ 中的表示）
- ${}^{A}_{B}\mathbf{T}$：从 $\{B\}$ 到 $\{A\}$ 的齐次变换
- 变换规则：${}^{A}\mathbf{p} = {}^{A}_{B}\mathbf{R}\,{}^{B}\mathbf{p} + {}^{A}\mathbf{p}_B$
- 左乘 = 相对于固定参考系的变换
- 右乘 = 相对于当前物体参考系的变换
- 主动旋转 (alibi) ≠ 被动旋转 (alias)

## 3. DH 参数约定

### 标准 DH (Standard DH / SDH)
$${}^{i-1}_{i}\mathbf{T} = R_z(\theta_i)\,T_z(d_i)\,T_x(a_i)\,R_x(\alpha_i)$$
参数下标：$a_i, \alpha_i, d_i, \theta_i$

### 改进 DH (Modified DH / MDH) — Khalil-Kleinfinger
$${}^{i-1}_{i}\mathbf{T} = R_x(\alpha_{i-1})\,T_x(a_{i-1})\,R_z(\theta_i)\,T_z(d_i)$$
参数下标：$a_{i-1}, \alpha_{i-1}, d_i, \theta_i$

## 4. Twist 排列

$$\mathcal{V} = \begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix}$$

线速度在前，角速度在后。这是 Lynch & Park (Modern Robotics) 的约定。

## 5. 四元数

- 顺序：$(w, x, y, z)$，标量在前（Hamilton 约定）
- 乘法：$\mathbf{q}_1 \otimes \mathbf{q}_2$ 表示先 $\mathbf{q}_2$ 后 $\mathbf{q}_1$ 的旋转组合
- $\mathbf{q}$ 和 $-\mathbf{q}$ 表示同一旋转

## 6. 欧拉角

- 默认使用 ZYX Tait-Bryan 角 (yaw, pitch, roll)
- 内旋 (intrinsic)：每次绕当前轴旋转

## 7. 重力方向

- 世界系中重力沿 $-z$ 方向：$\mathbf{g} = [0, 0, -9.81]^T$ m/s²
- 部分平面 2R 臂示例使用 $-y$ 方向以便在 XY 平面显示

## 8. 关节和力矩

- 旋转关节变量：$\theta_i$ (rad)
- 移动关节变量：$d_i$ (m)
- 关节力矩：$\tau_i$ (N·m 或 N)
- 末端力：$\mathbf{f}$ (N)，末端力矩：$\mathbf{n}$ (N·m)

## 9. 惯性参数

标准惯性参数（每个连杆 10 个，关于连杆坐标系原点 $O_i$）：
$$\boldsymbol{\theta}_i = [m_i, mc_{x,i}, mc_{y,i}, mc_{z,i}, I_{O,xx,i}, I_{O,xy,i}, I_{O,xz,i}, I_{O,yy,i}, I_{O,yz,i}, I_{O,zz,i}]^T$$

$I_O$ 和 $I_C$ 的关系（平行轴定理）：
$$I_O = I_C + m\left(|c|^2\mathbf{I} - \mathbf{c}\mathbf{c}^T\right)$$

## 10. 离散化

- 默认使用 RK4 进行动力学积分
- KF 使用精确离散化（零阶保持 ZOH）

## 11. 主要参考资料

1. Lynch & Park — *Modern Robotics* (PoE, twist, spatial algebra)
2. Craig — *Introduction to Robotics* (DH parameters)
3. Siciliano et al. — *Robotics: Modelling, Planning and Control*
4. Featherstone — *Rigid Body Dynamics Algorithms* (spatial vectors)
5. Thrun, Burgard, Fox — *Probabilistic Robotics* (KF, EKF, PF)
6. Barfoot — *State Estimation for Robotics* (Lie groups, SE(3))
