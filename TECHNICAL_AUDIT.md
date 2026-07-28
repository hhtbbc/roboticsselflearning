# TECHNICAL_AUDIT.md — 技术审校基线

## 最新审校日期：2026-07-28

## 审校范围

45 个 Notebook（含扩展章节）、9 个 Python 源码模块、5 个设计文档、3 个脚本。

## 状态标记

| 标记 | 含义 |
|:----:|------|
| ✅ | 已修复 + 测试通过 |
| 🔧 | 代码已修改，待 Notebook 执行验证 |
| ⏳ | 未处理 |
| ❌ | 待处理（P0） |

---

## P0 错误（产生错误结果或误导学习）

| # | 文件 | 错误描述 | 修复方向 | 状态 |
|---|------|----------|----------|:----:|
| 1 | `src/robotics_learning/kinematics.py:dh_transform` | 标准DH/改进DH命名混乱 | 统一命名为 SDH/MDH | ✅ |
| 2 | `src/robotics_learning/kinematics.py:ik_numerical` | 姿态误差用反对称部分提取，在180°附近退化 | 改用 SO(3) Log map + IKResult | ✅ |
| 3 | `notebook_sources/14_time_parameterization.py` | TOPP不可行传播不完整、命名'时间最优'但只是近似、约束验证只打印百分比 | s_dot²递推 + 显式不可行检测 + 断言验证 + 统一命名为'教学版近似' | 🔧 |
| 4 | `src/robotics_learning/planning.py` | 边碰撞检测仅检查新节点和中点 | `edge_collision_free` 全边插值 | ✅ |
| 5 | `src/robotics_learning/planning.py:rrt_star_plan` | RRT* rewire后未递归更新子树代价 | `_update_subtree_cost` 递归更新 | ✅ |
| 6 | `notebook_sources/25_integrated_project.py` | 快速碰撞漏检、线性轨迹不C²、无碰撞断言、EKF Q未离散化 | 精确碰撞 + shortcut + C²样条 + 全量安全断言 + Q_d离散化 | 🔧 |
| 7 | `notebook_sources/24_sensor_fusion.py` | 用"加速度计"名称但实际测量角速度 | 改为陀螺仪+编码器融合 | ⏳ |
| 8 | `notebook_sources/23_ekf_particle_filter.py` | mu_efk历史数组覆盖、协方差椭圆用终态、角度创新未归一化 | 修复历史记录和atan2归一化 | ⏳ |
| 9 | `src/robotics_learning/estimation.py:ParticleFilter` | 双重噪声；似然未用log-sum-exp | 统一约定 + log-sum-exp | ✅ |
| 10 | `notebook_sources/12_advanced_dynamics.py` | 惯性参数定义混淆 I_C vs I_O | 明确标准参数定义 | ⏳ |

## 新增 P0 修复 (2026-07-28)

| # | 文件 | 错误描述 | 修复 | 状态 |
|---|------|----------|------|:----:|
| 11 | `src/robotics_learning/transforms.py:axis_angle_to_rot` | 零轴配非零角度静默返回单位旋转 | raise ValueError + 测试 | ✅ |
| 12 | `src/robotics_learning/transforms.py:slerp` | 非单位四元数输入未归一化 | 开头 normalize() + 5 项测试 | ✅ |
| 13 | `src/robotics_learning/planning.py:RRT/RRT*` | 周期关节仍用欧氏距离 | ConfigurationSpace 类 + 集成 | ✅ |
| 14 | `src/robotics_learning/planning.py:RRT*` | 缺少输入验证、邻域公式注释不准确 | validate_planning_problem + 注释修正 | ✅ |
| 15 | `src/robotics_learning/kinematics.py:compute_analytical_jacobian` | 名称不准确（实际是有穷差分欧拉角雅可比） | 重命名 + 准确注释 + 兼容别名 | ✅ |
| 16 | `src/robotics_learning/kinematics.py:ik_numerical` | 位置/姿态同容差，无任务权重 | IKResult + 分开容差 | ✅ |
| 17 | `src/robotics_learning/estimation.py:EKF` | 无角度/流形状态的自定义残差 | residual_fn + state_injection_fn | ✅ |
| 18 | `notebook_sources/24c_pnp_icp_slam.py` | 标题/目标超过实际内容 | 对齐为'演示'，加 z_c>0 检查 | 🔧 |
| 19 | `tests/test_trajectory.py` | 无 TOPP 测试 | 新增 TestTOPP 类 (3 tests) | ✅ |
| 20 | `notebook_sources/14_time_parameterization.py` | 约束验证只打印百分比 | assert 强制验证 + 区间加速度公式 | 🔧 |
| 21 | `notebook_sources/25_integrated_project.py` | RRT用快速碰撞、无shortcut、线性轨迹、EKF噪声缺少离散化依据 | 精确碰撞 + shortcut_path + C²样条 + Q_d公式 | 🔧 |

---

## P1 错误（理论链条不完整或算法名不副实）

| # | 描述 | 状态 |
|---|------|:----:|
| 1 | SE(3)/twist/wrench/Adjoint/PoE 缺失 | ⏳ |
| 2 | 加速度运动学(ḊJ q̇)缺失 | ⏳ |
| 3 | 空间向量动力学(RNEA/CRBA/ABA完整实现)缺失 | ⏳ |
| 4 | 接触动力学和抓取缺失 | ⏳ |
| 5 | 数值优化基础(Gauss-Newton/LM/KKT/QP)缺失 | ⏳ |
| 6 | 轨迹优化和kinodynamic规划缺失 | ⏳ |
| 7 | LQR/LQG/MPC缺失 | ⏳ |
| 8 | ESKF和IMU预积分缺失 | ⏳ |
| 9 | 相机几何/PnP/ICP/SLAM缺失 | ⏳ |
| 10 | 移动机器人运动学缺失 | ⏳ |
| 11 | 自适应控制/ILC缺少完整推导和可运行代码 | ⏳ |
| 12 | 执行器/减速器/摩擦模型缺失 | ⏳ |
| 13 | 标定和时间同步缺失 | ⏳ |
| 14 | URDF/ROS2接口缺失 | ⏳ |

---

## P2 错误（工程实现和可维护性）

| # | 描述 | 状态 |
|---|------|:----:|
| 1 | 无测试体系 | ✅ 83 tests, 5 测试文件 |
| 2 | check_notebooks.py 原地覆盖原始Notebook | ✅ 已改用 nbclient + 输出至 outputs/ |
| 3 | 无 CONVENTIONS.md 统一符号约定 | ✅ 已创建 |
| 4 | 无 CI/CD | ✅ .github/workflows/ci.yml |
| 5 | 部分Notebook答案链接指向不存在的文件 | ⏳ |
| 6 | pyproject.toml 缺少dev依赖组 | ✅ 已添加 |
| 7 | 随机实验缺少种子设置 | ⏳ (部分 Notebook 已使用 `RandomState(42)`) |

---

## P3 错误（排版、链接、命名和教学体验）

| # | 描述 | 状态 |
|---|------|:----:|
| 1 | PROGRESS.md 职责混淆 | ⏳ |
| 2 | 部分 Notebook 缺少"本节约定"和"参考资料"节 | ⏳ |
| 3 | 解答链接格式不统一 | ⏳ |

---

## 仍待处理（2026-07-28）

| # | 问题 | 优先级 |
|---|------|:------:|
| 1 | NB04b/05b 代码仍使用局部定义的旧版 adjoint/se3_hat | P2 |
| 2 | README 索引未更新至 45 个 Notebook | P3 |
| 3 | time_optimal_parameterization 重命名为 velocity_mvc_from_joint_limits (别名保留) | P1 |
| 4 | NB14 Notebook + 公共 TOPP 接口衔接（NB 使用独立实现,应统一调用轨迹模块） | P1 |
| 5 | Jupytext 双向同步未配置 | P3 |
| 6 | 估计模块缺少完整 UKF predict/update 实现 | P1 |
