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
# # Notebook 00：课程导航指南
#
# ## 1. 本节在知识体系中的位置
#
# 本 Notebook 是整个课程的**入口和地图**。它不是正式教学内容，但请务必通读一遍——它能帮助你理解：
#
# - 课程的整体结构和学习路径
# - 每个 Notebook 的定位和依赖关系
# - 符号约定和术语规范
# - 如何有效使用本课程
#
# ```
# 本课程的四条主线：
#
#   建模 (Modeling) ────→ 规划 (Planning)
#        │                      │
#        ↓                      ↓
#   控制 (Control) ←──── 感知与状态估计 (Perception & Estimation)
#        │                      │
#        └──────────────────────┘
#               闭环系统
# ```

# %% [markdown]
# ## 2. 学习目标
#
# - 理解课程的四条主线和十个核心模块
# - 掌握内容三级分类系统（必须掌握 / 需要理解 / 拓展阅读）
# - 熟悉数学符号约定
# - 知道如何使用 Jupyter Notebook 和 uv 环境
# - 了解推荐教材和学习策略

# %% [markdown]
# ## 3. 课程设计理念
#
# ### 3.1 四条主线 × 十个模块
#
# | 主线 | 模块 | Notebook 范围 |
# |------|------|---------------|
# | **建模 (Modeling)** | ① 数学基础 ② 机械臂运动学 ③ 速度运动学/雅可比/静力学 ④ 动力学建模 ⑤ 高级动力学 | NB01-NB12 |
# | **规划 (Planning)** | ⑥ 轨迹规划与时间参数化 ⑦ 运动规划 | NB13-NB16 |
# | **控制 (Control)** | ⑧ 关节空间控制 ⑨ 操作空间控制与力控制 | NB17-NB20 |
# | **感知与状态估计 (Perception & Estimation)** | ⑩ 状态估计与多传感器融合 | NB21-NB24 |

# %% [markdown]
# ## 4. 内容三级分类
#
# 每个 Notebook 使用以下标记：
#
# - **⭐ 必须掌握**：核心推导和代码，必须理解并能独立推导/实现。面试和工程必备。
# - **📖 需要理解**：重要的背景知识和工程实践。需理解但不必背诵所有细节。
# - **📚 拓展阅读**：进阶内容，初学可跳过。深入学习时回头阅读。
#
# **关键原则**：不因为课程只有 30 天就把理论简单化。通过三级分类来压缩学习范围，而不是牺牲正确性。

# %% [markdown]
# ## 5. 数学符号约定
#
# 本课程统一使用以下符号约定（与 Craig、Siciliano 等教材一致）：
#
# | 符号 | 含义 | 示例 |
# |------|------|------|
# | 斜体小写 | 标量 | $x, y, \theta, t$ |
# | 粗体小写 | 向量 | $\mathbf{v}, \boldsymbol{\omega}, \mathbf{q}$ |
# | 粗体大写 | 矩阵 | $\mathbf{R}, \mathbf{T}, \mathbf{J}, \mathbf{M}$ |
# | 上标 | 所属参考系 | ${}^{A}\mathbf{v}$：向量 $\mathbf{v}$ 在系 $\{A\}$ 中的表示 |
# | 下标 | 对象标识 | $\mathbf{p}_{i}$：第 $i$ 个点的位置 |
# | 双下标变换 | 从某系到某系 | ${}^{A}_{B}\mathbf{T}$：从 $\{B\}$ 到 $\{A\}$ 的齐次变换 |
# | $\times$ | 向量叉乘 | $\mathbf{a} \times \mathbf{b}$ |
# | $[\cdot]_\times$ | 叉乘矩阵 | $[\mathbf{v}]_\times$ = skew-symmetric matrix |
# | $\otimes$ | 四元数乘法 | $\mathbf{q}_1 \otimes \mathbf{q}_2$ |
# | $\|\cdot\|$ | 欧几里得范数 | $\|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2 + v_z^2}$ |

# %% [markdown]
# ## 6. 环境与工具

# %% [markdown]
# ### 6.1 Python 环境 (uv)
#
# ```bash
# # 安装依赖
# cd roboticsselflearning
# uv sync
#
# # 注册 Jupyter 内核
# uv run python -m ipykernel install --user --name robotics-learning
# ```
#
# 在 VSCode 中打开 `.ipynb` 文件时，选择内核 `Python (robotics-learning)`。

# %% [markdown]
# ### 6.2 Jupyter Notebook 常用快捷键
#
# | 快捷键 | 功能 |
# |--------|------|
# | `Shift+Enter` | 执行当前单元格，移到下一个 |
# | `Ctrl+Enter` | 执行当前单元格，保持在原位 |
# | `Esc + A` | 在当前单元格上方插入新单元格 |
# | `Esc + B` | 在当前单元格下方插入新单元格 |
# | `Esc + M` | 切换为 Markdown 单元格 |
# | `Esc + Y` | 切换为代码单元格 |
# | `Esc + D + D` | 删除单元格 |
# | `Shift+Tab` | 显示函数文档 |

# %% [markdown]
# ### 6.2 常用魔法命令
#
# ```python
# %matplotlib inline     # 图像嵌入 Notebook（已默认）
# %load_ext autoreload
# %autoreload 2          # 自动重载修改后的模块
# %timeit some_function() # 计时
# %%time                  # 单元格计时
# ```

# %%
# 测试环境
import sys
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx

print(f"Python: {sys.version}")
print(f"NumPy:  {np.__version__}")
print(f"SciPy:  {sp.__version__}")
print(f"SymPy:  {sym.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"NetworkX: {nx.__version__}")

# 确保图像内嵌
%matplotlib inline
print("\n✅ 环境就绪！")

# %% [markdown]
# ## 7. Notebook 索引与依赖关系

# %% [markdown]
# ### 第 1 周：数学基础与运动学
#
# | # | Notebook | 内容 | 前置 | 必须掌握的核心 |
# |---|----------|------|:--:|---------------|
# | 01 | 线性代数复习 | 向量/矩阵/SVD/伪逆 | — | SVD、伪逆、正定性 |
# | 02 | 坐标系与刚体运动 | 齐次变换/基变换/SO(3)SE(3) | 01 | 齐次变换的构造与逆 |
# | 03 | 旋转的多种表示 | 欧拉角/轴角/万向锁 | 02 | 所有表示之间的转换 |
# | 04 | 四元数与李群基础 | 四元数/SO(3)exp/log | 03 | 四元数运算、slerp |
# | 05 | 正运动学 | DH参数/FK/工作空间 | 02,04 | DH→T→末端位姿 |
# | 06 | 逆运动学 | 几何法/代数法/数值法 | 05 | 2R几何法、数值IK |
#
# ### 第 2 周：雅可比、静力学与动力学
#
# | # | Notebook | 内容 | 前置 | 必须掌握的核心 |
# |---|----------|------|:--:|---------------|
# | 07 | 雅可比与静力学 | 几何vs解析雅可比/速度/力传播 | 05,06 | J构造、τ=J^TF |
# | 08 | 奇异性与可操作度 | 奇异位型/可操作度椭球/条件数 | 07 | det(J)=0、椭球含义 |
# | 09-10 | 拉格朗日动力学 | M/C/g结构/2R臂完整推导 | 01,05 | 拉格朗日推导M/C/g |
# | 11 | 牛顿-欧拉动力学 | RNEA/CRBA/O(n)递推 | 09-10 | RNEA递推流程 |
# | 12 | 高级动力学 | 参数辨识/约束/浮动基 | 10-11 | τ=Yθ线性参数化 |
#
# ### 第 3 周：规划与控制
#
# | # | Notebook | 内容 | 前置 | 必须掌握的核心 |
# |---|----------|------|:--:|---------------|
# | 13 | 轨迹生成 | Path vs Trajectory/梯形/多项式/样条 | 05-06 | 五次多项式、梯形曲线 |
# | 14 | 时间参数化 | TOPP/(s,ṡ)相平面 | 13,07 | 链式分解、MVC概念 |
# | 15 | 运动规划基础 | C-space/A*/Dijkstra/势场法 | 01,05 | A*算法、C-space概念 |
# | 16 | 采样运动规划 | PRM/RRT/RRT*/碰撞检测 | 15 | RRT/RRT*实现 |
# | 17 | 关节空间控制 | PID/重力补偿PD/稳定性 | 09-10,13 | 重力补偿PD + Lyapunov |
# | 18 | 计算力矩控制 | CTC/反馈线性化/SMC | 17,09-11 | CTC控制律推导 |
#
# ### 第 4 周：高级控制、状态估计与综合
#
# | # | Notebook | 内容 | 前置 | 必须掌握的核心 |
# |---|----------|------|:--:|---------------|
# | 19 | 操作空间控制 | 任务空间动力学/冗余分辨率 | 18,07 | Λ/μ/p、零空间投影 |
# | 20 | 力/阻抗/导纳控制 | 混合力控/阻抗vs导纳/无源性 | 19,17 | 阻抗控制律 |
# | 21 | 概率与状态估计 | 贝叶斯滤波/高斯分布 | 01 | 预测-更新框架 |
# | 22 | 卡尔曼滤波 | 线性KF/卡尔曼增益/可观测性 | 21 | KF两步公式 |
# | 23 | EKF与粒子滤波 | EKF线性化/UKF/SIR粒子滤波 | 22 | EKF雅可比线性化 |
# | 24 | 传感器融合 | IMU/编码器/视觉/激光雷达/tf树 | 23,02 | 多传感器EKF |
# | 25 | 综合项目 | 完整闭环系统 | 全部 | 建模→规划→控制→估计 |
# | 26 | 面试复习 | 50+道面试题 | 全部 | 高频题回答框架 |

# %% [markdown]
# ## 8. 推荐教材与资源
#
# 1. **John J. Craig** — *Introduction to Robotics: Mechanics and Control*（经典入门，DH 参数章节特别清晰）
# 2. **Siciliano, Sciavicco, Villani, Oriolo** — *Robotics: Modelling, Planning and Control*（综合参考，深度和广度兼顾）
# 3. **Lynch & Park** — *Modern Robotics*（现代视角，配套视频和代码。PoE 表示法）
# 4. **Featherstone** — *Rigid Body Dynamics Algorithms*（动力学算法圣经）
# 5. **Thrun, Burgard, Fox** — *Probabilistic Robotics*（状态估计必读）
# 6. **LaValle** — *Planning Algorithms*（运动规划大全，免费在线）
# 7. **Murray, Li, Sastry** — *A Mathematical Introduction to Robotic Manipulation*（李群方法）
#
# 在线资源：
# - [Modern Robotics 视频课程](https://modernrobotics.northwestern.edu/)
# - [Peter Corke's Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)
# - [Underactuated Robotics (MIT 6.832)](http://underactuated.csail.mit.edu/)

# %% [markdown]
# ## 9. 学习策略建议
#
# 1. **每天三步走**：先看理论 + 推导 → 跑代码动手验证 → 做自测题
# 2. **手推比阅读重要**：即使看起来复杂，人工推导 1-2 遍会极大加深理解
# 3. **代码动手比看代码重要**：每个编程练习都要亲自实现，不要直接复制粘贴
# 4. **保持知识连接**：经常回顾 KNOWLEDGE_MAP.md，理解模块间的数据流
# 5. **善用弹性日**：如果某天内容太难，不必挣扎——标记后继续，弹性日回来补
# 6. **面试题天天练**：每天从 INTERVIEW_CHECKLIST.md 随机抽 3-5 题自测
# 7. **善用三级分类**：30 天不需要掌握所有内容，优先保证"必须掌握"的部分

# %% [markdown]
# ## 10. 下一步
#
# 如果你还没有配置好环境，请先运行上面的环境测试单元格。
#
# 确认环境就绪后，请进入 **Notebook 01：线性代数与矩阵论复习**，开始你的机器人学之旅！
#
# ```
# 从数学工具 → 坐标变换 → 旋转表示 → 运动学 → ...
# 每一步都建立在前一步的基础上。
# 祝学习顺利！
# ```
