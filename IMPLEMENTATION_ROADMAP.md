# IMPLEMENTATION_ROADMAP.md — 实施路线图

## 批次概览

| 批次 | 内容 | 预计工作量 | 独立测试 |
|:----:|------|:--------:|:--------:|
| 1 | 修复 P0 错误（DH/IK/TOPP/RRT/碰撞/EKF/IMU/PF/惯性参数） | 高 | ✅ |
| 2 | SE(3)/twist/Adjoint/PoE + 加速度运动学 | 高 | ✅ |
| 3 | 空间向量动力学(RNEA/CRBA/ABA) + 刚体惯性 | 高 | ✅ |
| 4 | 数值优化 + 轨迹优化 + kinodynamic planning | 高 | ✅ |
| 5 | LQR/LQG/MPC + 自适应控制/ILC | 高 | ✅ |
| 6 | 接触动力学 + 抓取 + ESKF + IMU预积分 | 高 | ✅ |
| 7 | 相机/PnP/ICP/SLAM因子图 | 高 | ✅ |
| 8 | 移动机器人 + 标定 + URDF/ROS2 | 中 | ✅ |
| 9 | 测试体系 + CI + 综合项目重构 | 高 | ✅ |

## 批次 1 详细任务

### 1.1 DH 修复
- [ ] 重命名: 'standard'→'MDH', 'modified'→'SDH'，或直接命名 sdh/mdh
- [ ] 修复矩阵乘法顺序和参数下标
- [ ] 更新 NB05 中的文字说明
- [ ] 增加同一物理模型的 SDH vs MDH 对比测试

### 1.2 IK 姿态误差修复
- [ ] 实现基于 SO(3) Log 的姿态误差
- [ ] 加权阻尼最小二乘
- [ ] 自适应阻尼 + 最大步长限制
- [ ] 180° 附近姿态误差测试

### 1.3 TOPP 修复
- [ ] 修复离散积分公式 (s_dot² 递推)
- [ ] 补齐一般路径 f'(s) 和 f''(s)
- [ ] 速度/加速度约束的正确形成
- [ ] 前向最大加速 + 后向最大减速
- [ ] MVC 计算和切换点检测
- [ ] 或准确命名为"简化路径速度参数化"

### 1.4 RRT/RRT* 修复
- [ ] 全边碰撞检测函数
- [ ] 周期性关节距离/插值/steer
- [ ] RRT 目标连接边完整检测
- [ ] RRT* 子树代价递归更新
- [ ] RRT* 迭代结束后选最优
- [ ] 邻域半径公式
- [ ] 规划失败原因返回

### 1.5 综合项目修复
- [ ] 完整连杆-障碍物距离检测
- [ ] RRT 失败时 assert 报错
- [ ] 真实动力学 EKF
- [ ] 样条平滑后碰撞复检

### 1.6 IMU 修复
- [ ] 改为陀螺仪+编码器融合
- [ ] 增加偏置估计
- [ ] 增加真实加速度计物理模型示例

### 1.7 EKF 历史数组修复
- [ ] 正确保存每时刻 mu 和 Sigma
- [ ] 角度残差 atan2 归一化
- [ ] 协方差椭圆用对应时刻的 Sigma

### 1.8 PF 双重噪声修复
- [ ] 统一噪声注入约定
- [ ] log-sum-exp 似然计算
- [ ] 补充 roughening 和 kidnapped robot

### 1.9 惯性参数修复
- [ ] 明确 I_O vs I_C
- [ ] 平行轴定理说明
- [ ] 标准 10 参数定义修正

## 批次 1 验证步骤

```bash
uv sync --all-groups
uv run pytest tests/ -v
uv run python scripts/check_notebooks.py notebooks/05_forward_kinematics.ipynb
uv run python scripts/check_notebooks.py notebooks/06_inverse_kinematics.ipynb
uv run python scripts/check_notebooks.py notebooks/14_time_parameterization.ipynb
uv run python scripts/check_notebooks.py notebooks/16_sampling_planning.ipynb
uv run python scripts/check_notebooks.py notebooks/23_ekf_particle_filter.ipynb
uv run python scripts/check_notebooks.py notebooks/24_sensor_fusion.ipynb
uv run python scripts/check_notebooks.py notebooks/25_integrated_project.ipynb
```
