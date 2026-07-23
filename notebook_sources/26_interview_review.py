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
# # Notebook 26：机器人学面试系统复习
#
# ## 1. 概述
#
# 本 Notebook 按七大类别整理 50+ 道机器人算法岗位高频面试题。
# 每道题包含四层回答框架：
# - **一句话回答**：面试开场 30 秒内的简洁回应
# - **展开分析**：数学推导与原理说明
# - **工程实例**：工业中的应用场景
# - **追问预期**：面试官可能的后续追问
#
# 完整题库见 `INTERVIEW_CHECKLIST.md`。

# %% [markdown]
# ## 2. 数学与变换（8 题）

# %% [markdown]
# ### Q1: 旋转矩阵的性质？怎么验证 3×3 矩阵是合法旋转矩阵？
#
# **一句话回答**：旋转矩阵是行列式为 +1 的正交矩阵（即 $\mathbf{R}^T\mathbf{R} = \mathbf{I}$，$\det(\mathbf{R}) = +1$），属于 SO(3) 群。
#
# **展开分析**：
# - **正交性**：$\mathbf{R}^T\mathbf{R} = \mathbf{I} \implies \mathbf{R}^{-1} = \mathbf{R}^T$。这 9 个约束条件将 9 个元素的自由度降至 3。
#   - 列向量标准正交：$\mathbf{r}_i \cdot \mathbf{r}_j = \delta_{ij}$
#   - 行向量标准正交：同上
# - **行列式**：$\det(\mathbf{R}) = +1$ 排除反射（$\det = -1$）。数学上 $O(3) = SO(3) \cup \{\mathbf{R}: \det(\mathbf{R}) = -1\}$。
# - **验证方法**：`np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-6`
#
# **工程实例**：从传感器融合模块输出的旋转矩阵需周期性地重新归一化（投影回 SO(3)），防止浮点误差累积导致漂移。
#
# **面试官追问**：SO(3) 的维度为什么是 3？

# %% [markdown]
# ### Q2: 欧拉角和四元数各有什么优缺点？
#
# **一句话回答**：欧拉角直观（3 个角度），但有万向锁；四元数无奇异性、插值光滑（slerp），但不直观。
#
# | | 欧拉角 | 四元数 |
# |---|---|---|
# | 参数数 | 3 | 4（约束 $\|q\|=1$） |
# | 奇异性 | 万向锁（pitch=±90°） | **无** |
# | 插值 | 线性插值不光滑 | slerp 常数角速度 |
# | 人读 | ✓直观 | ✗不直观 |
# | 组合 | 需转回矩阵 | 四元数乘法 $O(1)$ |
#
# **工程实例**：无人机飞控界面显示欧拉角；姿态估计 EKF 内部用四元数误差状态。
#
# **面试官追问**：四元数为什么没有万向锁？

# %% [markdown]
# ### Q3-Q8: 见 INTERVIEW_CHECKLIST.md #1.3-1.8

# %% [markdown]
# ## 3. 运动学（8 题）

# %% [markdown]
# ### Q1: DH 参数是什么？标准 DH 和改进 DH 的区别？
#
# **一句话回答**：DH 参数用 4 个量 $(a, \alpha, d, \theta)$ 描述相邻连杆的齐次变换，标准 DH 变换顺序为 $R_z(\theta)T_z(d)T_x(a)R_x(\alpha)$，改进 DH 为 $R_x(\alpha)T_x(a)R_z(\theta)T_z(d)$。
#
# **展开分析**：
# - $a_i$：连杆长度（沿 $X_i$ 轴量，$Z_{i-1}$ 到 $Z_i$ 的最近距离）
# - $\alpha_i$：连杆扭转角（绕 $X_i$ 轴，$Z_{i-1}$ 到 $Z_i$ 的转角）
# - $d_i$：连杆偏置（沿 $Z_{i-1}$ 轴，$X_{i-1}$ 到 $X_i$ 的最近距离）
# - $\theta_i$：关节角（绕 $Z_{i-1}$ 轴，$X_{i-1}$ 到 $X_i$ 的转角）
# - 4 个参数足以描述：两个约束来自 $X_i$ 垂直于 $Z_{i-1}$ 且与 $Z_i$ 相交
#
# **面试官追问**：同一个机械臂用两种约定的 DH 表给出的末端位姿一样吗？

# %% [markdown]
# ### Q2: 为什么大多数 6 自由度机械臂有解析 IK 解？（Pieper 准则）
#
# **一句话回答**：Pieper 准则指出，当 6 自由度机械臂的最后三个关节轴交于一点（球形腕）时，IK 有闭式解——前三个关节决定位置，后三个关节决定姿态，位置和姿态解耦。
#
# **面试官追问**：不满足 Pieper 准则时怎么办？数值 IK 为什么需要 DLS？

# %% [markdown]
# ## 4. 雅可比与动力学（10 题）

# %% [markdown]
# ### Q1: 推导 $\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$
#
# **一句话回答**：通过虚功原理——末端虚位移 $\delta\mathbf{x}$ 做的功等于关节虚位移 $\delta\mathbf{q}$ 做的功：$\mathbf{F}^T\delta\mathbf{x} = \boldsymbol{\tau}^T\delta\mathbf{q}$，代入 $\delta\mathbf{x} = \mathbf{J}\delta\mathbf{q}$ 即得。
#
# **展开分析**：这是**静力学对偶原理**——速度域 $\dot{\mathbf{x}} = \mathbf{J}\dot{\mathbf{q}}$ 和力域 $\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$ 是对偶的。$\mathbf{J}^T$ 的每一行给出了"末端 1N 力在各关节产生多少力矩"。
#
# **面试官追问**：如果机器人处于奇异构型，这个力映射有什么问题？

# %% [markdown]
# ### Q2: Ṁ − 2C 的反对称性有什么工程意义？
#
# **一句话回答**：它意味着科氏力/离心力不做功（$\dot{\mathbf{q}}^T\mathbf{C}\dot{\mathbf{q}} = \frac{1}{2}\dot{\mathbf{q}}^T\dot{\mathbf{M}}\dot{\mathbf{q}}$），是重力补偿 PD 和阻抗控制中李雅普诺夫稳定性证明的核心。
#
# **展开分析**：
# - 总动能 $\mathcal{K} = \frac{1}{2}\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}}$
# - $\dot{\mathcal{K}} = \dot{\mathbf{q}}^T\mathbf{M}\ddot{\mathbf{q}} + \frac{1}{2}\dot{\mathbf{q}}^T\dot{\mathbf{M}}\dot{\mathbf{q}}$
# - 代入动力学：$\dot{\mathcal{K}} = \dot{\mathbf{q}}^T(\boldsymbol{\tau} - \mathbf{C}\dot{\mathbf{q}} - \mathbf{g}) + \frac{1}{2}\dot{\mathbf{q}}^T\dot{\mathbf{M}}\dot{\mathbf{q}} = \dot{\mathbf{q}}^T\boldsymbol{\tau} - \dot{\mathbf{q}}^T\mathbf{g} + \frac{1}{2}\dot{\mathbf{q}}^T(\dot{\mathbf{M}} - 2\mathbf{C})\dot{\mathbf{q}}$
# - 最后一项 = 0 → 功率平衡不包含 $\mathbf{C}$ 的贡献
#
# **面试官追问**：如果 C 矩阵不以 Christoffel 符号构造，Ṁ-2C 还反对称吗？

# %% [markdown]
# ## 5. 轨迹与运动规划（10 题）

# %% [markdown]
# ### Q1: Path 和 Trajectory 的区别？
#
# **一句话回答**：Path（路径）是纯几何的空间曲线 $\mathbf{q}(s)$，Trajectory（轨迹）是赋予了时间律的空间曲线 $\mathbf{q}(t)$。
#
# **展开分析**：
# - 路径 = 运动规划器（NB15-16）的输出
# - 轨迹 = 时间参数化器（NB13-14）的输出
# - 同一个路径可以用不同时间律（快/慢、梯形/五次多项式）
#
# **面试官追问**：时间最优沿路径规划（TOPP）在 (s, ṡ) 相平面中怎么求解？

# %% [markdown]
# ### Q2: RRT 为什么在高维空间中有效？
#
# **一句话回答**：RRT 利用 Voronoi 偏置自然偏向未探索区域（大 Voronoi 区域），不需要显式构建 C_{obs}，避免了高维空间的维度灾难。
#
# **面试官追问**：RRT* 的 rewire 步骤为什么不破坏概率完备性？

# %% [markdown]
# ## 6. 控制（10 题）

# %% [markdown]
# ### Q1: 为什么重力补偿 PD 可以消除稳态误差？用李雅普诺夫函数证明。
#
# **一句话回答**：选择 $V = \frac{1}{2}\dot{\mathbf{q}}^T\mathbf{M}\dot{\mathbf{q}} + \frac{1}{2}\mathbf{e}^T\mathbf{K}_p\mathbf{e}$，利用 $\dot{\mathbf{M}}-2\mathbf{C}$ 反对称性可得 $\dot{V} = -\dot{\mathbf{q}}^T\mathbf{K}_d\dot{\mathbf{q}} \leq 0$。
#
# **展开分析**：完整证明见 NB17 第 4.2 节。关键步骤：代入 CTC 消去 $\mathbf{C}$ 和 $\mathbf{g}$、$\dot{\mathbf{M}}-2\mathbf{C}$ 反对称导致二次型为零。
#
# **面试官追问**：$\dot{V}$ 是半负定的（$\leq 0$ 而非 $<0$），如何证明渐近稳定（而不仅仅是稳定）？——需要 LaSalle 不变原理。

# %% [markdown]
# ### Q2: 阻抗控制和导纳控制的区别？什么场景选哪个？
#
# **一句话回答**：阻抗控制是"测量运动→输出力"（适合轻质、高反驱性机器人），导纳控制是"测量力→输出运动"（适合重质、高减速比工业机器人）。按环境刚度选择。
#
# **面试官追问**：耦合稳定性的物理直觉是什么？

# %% [markdown]
# ## 7. 状态估计（10 题）

# %% [markdown]
# ### Q1: 卡尔曼增益 K 的意义？
#
# **一句话回答**：$K$ 在预测和观测之间做最优加权——观测噪声大（$\mathbf{R}$ 大）时 $K$ 小（更信预测），过程噪声大（$\mathbf{Q}$ 大）时 $K$ 大（更信观测）。
#
# **展开分析**：
# - 一维：$K = \sigma^2_{prior} / (\sigma^2_{prior} + \sigma^2_{obs}) \in [0, 1]$
# - 多维：$\mathbf{K} = \hat{\boldsymbol{\Sigma}}\mathbf{C}^T(\mathbf{C}\hat{\boldsymbol{\Sigma}}\mathbf{C}^T + \mathbf{R})^{-1}$
# - K 的各行给出了"每个测量维度的新息在状态更新中的权重"
#
# **面试官追问**： $\boldsymbol{\Sigma}_t$ 为什么不依赖观测 $\mathbf{z}_t$？

# %% [markdown]
# ### Q2: EKF 的核心局限是什么？
#
# **一句话回答**：EKF 使用一阶泰勒展开线性化非线性函数——在高度非线性的区域，一阶近似很差；且 EKF 倾向于低估协方差（不一致性）。
#
# **面试官追问**：UKF 如何改善这个问题？

# %% [markdown]
# ## 8. 工程与系统设计（5 题）

# %% [markdown]
# ### Q1: 如果要设计一个机械臂捡货系统，你会怎么分解？
#
# **一句话回答**：感知（相机检测物体位姿）→ IK（求目标关节角）→ 规划（检查碰撞+生成路径）→ 轨迹（给路径加时间律）→ 控制（跟踪轨迹+力控抓取）。
#
# **展开分析**：这是整个课程的完整闭环——NB06 IK → NB15-16 规划 → NB13-14 轨迹 → NB17-18 控制 → NB20 力控 → NB24 传感器状态估计。
#
# **面试官追问**：实时性要求（如 1kHz 控制回路）各模块怎么分配计算资源？

# %% [markdown]
# ## 9. 核心公式速查表

# %% [markdown]
# | 类别 | 公式 | 所在 NB |
# |------|------|:------:|
# | SO(3) 性质 | $\mathbf{R}^T\mathbf{R}=\mathbf{I}, \det(\mathbf{R})=+1$ | 02 |
# | 罗德里格斯 | $\mathbf{R} = \mathbf{I} + s_\theta[\mathbf{k}]_\times + (1-c_\theta)[\mathbf{k}]_\times^2$ | 03 |
# | 四元数旋转 | $\mathbf{q} = (\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\mathbf{k})$ | 04 |
# | so(3) exp | $\exp([\boldsymbol{\omega}]_\times) =$ 罗德里格斯 | 04 |
# | DH 变换 | $^{i-1}_i\mathbf{T} = R_z(\theta) T_z(d) T_x(a) R_x(\alpha)$ | 05 |
# | FK | $^{0}_n\mathbf{T} = \prod_{i=1}^n {}^{i-1}_i\mathbf{T}$ | 05 |
# | 几何雅可比列 | $\mathbf{J}_i = [\mathbf{z}\times(\mathbf{p}_e-\mathbf{p}); \mathbf{z}]$ | 07 |
# | 静力学 | $\boldsymbol{\tau} = \mathbf{J}^T\mathbf{F}$ | 07 |
| 可操作度 | $\mu = \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$ | 08 |
| 动力学标准形式 | $\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g} = \boldsymbol{\tau}$ | 09 |
| Ṁ−2C 反对称 | $\dot{\mathbf{q}}^T(\dot{\mathbf{M}}-2\mathbf{C})\dot{\mathbf{q}} = 0$ | 09 |
| 五次多项式 | 6 个边界条件 → $a_0\dots a_5$ | 13 |
| A* | $f(n) = g(n) + h(n)$ | 15 |
| 重力补偿 PD | $\boldsymbol{\tau} = \mathbf{K}_p\mathbf{e} + \mathbf{K}_d\dot{\mathbf{e}} + \mathbf{g}(\mathbf{q})$ | 17 |
| CTC | $\boldsymbol{\tau} = \mathbf{M}(\ddot{\mathbf{q}}_d + \mathbf{K}_v\dot{\mathbf{e}} + \mathbf{K}_p\mathbf{e}) + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g}$ | 18 |
| 零空间投影 | $\dot{\mathbf{q}}_{null} = (\mathbf{I} - \mathbf{J}^+\mathbf{J})\dot{\mathbf{q}}_0$ | 19 |
| 阻抗控制 | $\mathbf{M}_d\ddot{\tilde{\mathbf{x}}} + \mathbf{D}_d\dot{\tilde{\mathbf{x}}} + \mathbf{K}_d\tilde{\mathbf{x}} = \mathbf{F}_{ext}$ | 20 |
| KF 预测 | $\hat{\boldsymbol{\mu}} = \mathbf{A}\boldsymbol{\mu} + \mathbf{B}\mathbf{u}, \hat{\boldsymbol{\Sigma}} = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T + \mathbf{Q}$ | 22 |
| KF 更新 | $\boldsymbol{\mu} = \hat{\boldsymbol{\mu}} + \mathbf{K}(\mathbf{z} - \mathbf{C}\hat{\boldsymbol{\mu}})$ | 22 |
| 卡尔曼增益 | $\mathbf{K} = \hat{\boldsymbol{\Sigma}}\mathbf{C}^T(\mathbf{C}\hat{\boldsymbol{\Sigma}}\mathbf{C}^T + \mathbf{R})^{-1}$ | 22 |
| $N_{eff}$ | $1 / \sum_i (w^{(i)})^2$ | 23 |

# %% [markdown]
# ## 10. 快速自测
#
# 每天随机抽 5 题，对照 `INTERVIEW_CHECKLIST.md` 检查答案完整性。
#
# **评分标准**：
# - 3/3：一句话回答 + 展开推导 + 工程例子 都能流畅回答
# - 2/3：能回答但推导需要看书
# - 1/3：只能说概念
# - 0/3：完全不会 → 回去看对应 Notebook
#
# > 祝面试顺利！
