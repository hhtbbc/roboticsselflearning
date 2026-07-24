# CHANGELOG_TECHNICAL.md — 技术变更记录

## 2026-07-23 — 批次 1: P0 错误修复 + 测试体系建立

### 修改的源文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/robotics_learning/kinematics.py` | 修复 | DH 命名统一 (sdh/mdh)、FK default、IK 姿态误差改用 SO(3) Log、步长限制 |
| `src/robotics_learning/transforms.py` | 修复 | 万向锁回解 sign bug、rot_to_axis_angle 180° 退化处理 |
| `src/robotics_learning/planning.py` | 修复+新增 | 全边碰撞检测、周期关节处理、RRT* 子树代价+最优选路、势场法记录 |
| `src/robotics_learning/estimation.py` | 修复 | PF log-sum-exp、统一噪声约定、EKF 接口完善 |
| `pyproject.toml` | 新增 | dev 依赖组 (pytest, ruff, nbqa) |

### 新增文件

| 文件 | 说明 |
|------|------|
| `CONVENTIONS.md` | 全局数学和坐标系约定 |
| `TECHNICAL_AUDIT.md` | 技术审校基线 |
| `IMPLEMENTATION_ROADMAP.md` | 实施路线图 |
| `CHANGELOG_TECHNICAL.md` | 本文档 |
| `tests/__init__.py` | 测试套件入口 |
| `tests/test_transforms.py` | SO(3)/SE(3)/四元数/李群测试 (20 tests) |
| `tests/test_kinematics.py` | DH/FK/IK/Jacobian 测试 (9 tests) |

### 修复的具体错误

1. **DH 命名**: `convention='standard'`→`'sdh'`, `'modified'`→`'mdh'`，明确矩阵乘法顺序
2. **IK 姿态误差**: 反对称部分 `(R-R^T)/2` → `so3_log(R_d R_curr^T)`，支持 180° 误差
3. **PRM/RRT 边碰撞**: 单中点检查 → 全边插值检测 `edge_collision_free()`
4. **RRT*  rewire**: 无子树代价更新 → `_update_subtree_cost()`，迭代终选最优
5. **PF 似然**: `weights *= exp(log_lik)` → log-sum-exp 稳定版
6. **万向锁**: 修复 pitch=+π/2 分支的 sign error
7. **axis_angle 180°**: 修复除零警告，改用 R+I 提取轴

### 测试结果

```
35 passed in 0.31s
```

### 待处理（批次 1 剩余）

- 修复 NB05/06/14/16/23/24/25 中引用旧约定的代码
- 运行受影响 Notebook 验证
- 补充动力学测试
- 补充估计测试

### 批次 1 补充 — Notebook 修复 (2026-07-23)

| 文件 | 变更 | 说明 |
|------|------|------|
| NB05 | 修复 | convention 'standard'/'modified' → 'sdh'/'mdh' |
| NB06 | 修复 | 同上 |
| NB23 | 修复 | 移除重复 mu_efk; Sigma_history 保存; 椭圆用对应时刻协方差 |
| NB24 | 修复 | IMU 陀螺仪命名 (C_acc→C_gyro, R_acc→R_gyro) |
| NB25 | 重大修复 | 碰撞检测全链路; RRT 无静默回退; 线性KF→动力学EKF; 轨迹碰撞验证 |

### 最终测试结果

```
35 tests passed in 0.23s
NB05 ✅ NB06 ✅ NB23 ✅ NB24 ✅ NB25 ✅
```

### 已知剩余问题

- NB14 TOPP 仍只处理直线路径（f''=0），离散积分未修复 — 标记为"简化路径速度参数化"
- tests/ 缺少动力学和估计测试
- .github/workflows/ci.yml 未创建
- BUILD_STATUS.md / LEARNING_LOG.md 未拆分
