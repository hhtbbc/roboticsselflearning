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
# # 第 2 周练习解答 — 雅可比、奇异性与动力学 (NB07-12)

# %%
import numpy as np
import sys; sys.path.insert(0, '..')

# %% [markdown]
# ## NB07 雅可比与静力学

# %% [markdown]
# ### 概念题 1: 几何雅可比 vs 解析雅可比
# - 几何雅可比 $\mathbf{J}$: $\dot{\mathbf{x}} = [\mathbf{v}; \boldsymbol{\omega}] = \mathbf{J}\dot{\mathbf{q}}$ — 映射到物理角速度
# - 解析雅可比 $\mathbf{J}_A$: $\dot{\boldsymbol{\phi}} = \mathbf{J}_A\dot{\mathbf{q}}$ — 映射到姿态参数导数
# - 关系: $\mathbf{J} = \mathbf{B}(\boldsymbol{\phi})\mathbf{J}_A$

# %% [markdown]
# ### 手算题: 2R 臂 (l1=1, l2=0.8), q1=30°, q2=45°
# $$\mathbf{J} = \begin{bmatrix}
# -l_1s_1 - l_2s_{12} & -l_2s_{12} \\
# l_1c_1 + l_2c_{12} & l_2c_{12}
# \end{bmatrix} = \begin{bmatrix}
# -1.273 & -0.773 \\ 1.073 & 0.207
# \end{bmatrix}$$
# 其中 $s_{12} = \sin(75°)=0.966$, $c_{12}=\cos(75°)=0.259$

# %%
import numpy as np
l1,l2=1.,0.8; q1,q2=np.radians(30),np.radians(45)
s12=np.sin(q1+q2);c12=np.cos(q1+q2)
J=np.array([[-l1*np.sin(q1)-l2*s12,-l2*s12],[l1*np.cos(q1)+l2*c12,l2*c12]])
print(f"J=\n{np.round(J,3)}")

# %% [markdown]
# ## NB08 奇异性与可操作度

# %% [markdown]
# ### 概念题 1: 2R 臂奇异性条件
# $\det(\mathbf{J}) = l_1l_2\sin(q_2) = 0$ → $q_2 = 0°$ 或 $q_2 = 180°$（边界奇异）

# %%
from src.robotics_learning.kinematics import compute_geometric_jacobian
dh_2r=np.array([[1.,0,0],[0.8,0,0]])
for q2 in [0.01, np.pi/2, np.pi-0.01]:
    J=compute_geometric_jacobian(dh_2r, np.array([0.5, q2]))[:2,:2]
    s=np.linalg.svd(J, compute_uv=False)
    print(f"q2={np.degrees(q2):.0f}°: σ=[{s[0]:.3f},{s[1]:.4f}], κ={s[0]/s[1]:.0f}")

# %% [markdown]
# ## NB09-10 拉格朗日动力学

# %% [markdown]
# ### 概念题 2: Ṁ-2C 反对称性的意义
# 意味着科氏力/离心力不做功。在李雅普诺夫稳定性证明中:
# $\dot{V} = -\dot{\mathbf{q}}^T\mathbf{K}_d\dot{\mathbf{q}} + \frac{1}{2}\dot{\mathbf{q}}^T(\dot{\mathbf{M}}-2\mathbf{C})\dot{\mathbf{q}}$
# 第二项恒为 0 → $\dot{V} \leq 0$ 成立。

# %% [markdown]
# ### 手算题: 1R 单摆 (m, l, I)
# 动能: $\mathcal{K} = \frac{1}{2}I\dot{\theta}^2$（绕支点转动惯量含平动贡献: $I_{pivot} = I + ml^2$）
# 势能: $\mathcal{P} = mgl(1-\cos\theta)$（取最低点为 0 势能面）
# 拉格朗日方程: $I_{pivot}\ddot{\theta} + mgl\sin\theta = \tau$
# 若 I 是绕质心的惯量，则 $I_{pivot} = I + ml^2$

# %%
from src.robotics_learning.dynamics import TwoLinkArmDynamics
dyn=TwoLinkArmDynamics(m1=1.,m2=1.,l1=1.,l2=0.8)
q=np.array([np.pi/4, np.pi/6]); qd=np.array([0.5,-0.3])
M=dyn.mass_matrix(q); C=dyn.coriolis_matrix(q,qd)
import numpy as np
eps=1e-5; M_plus=dyn.mass_matrix(q+eps*qd); M_dot_num=(M_plus-M)/eps
N=M_dot_num-2*C
print(f"q̇^T(Ṁ-2C)q̇ = {qd@N@qd:.2e}")  # ≈ 0

# %% [markdown]
# ## NB11 牛顿-欧拉

# %% [markdown]
# ### 概念题 1: RNEA 两遍递推
# - 向外 (1→n): 计算 ω, ω̇, v̇, a_c（运动学量）
# - 向内 (n→1): 计算 f, n，投影得 τ（力和力矩）
# - 重力处理: 设置 v̇₀ = -g

# %% [markdown]
# ## NB12 高级动力学

# %% [markdown]
# ### 概念题 1: 参数可辨识性
# 不是所有 10n 个惯性参数都可以从 τ 测量中辨识。基参数是通过 QR(Y) 选择的最小可辨识子集。
# 例如 2R 臂只有 5 个基参数（α,β,δ 和两个重力系数），而非 20 个。

# %%
# 验证参数线性化形式
from src.robotics_learning.dynamics import TwoLinkArmDynamics
dyn=TwoLinkArmDynamics(m1=1.,m2=1.,l1=1.,l2=0.8,g=9.81)
q=np.array([0.3,0.5]);qd=np.array([1.2,-0.8]);qdd=np.array([2.,-1.5])
tau=dyn.inverse_dynamics(q,qd,qdd)
# 验证: 手动构造 Yθ 应等于 τ
alpha=dyn.alpha;beta=dyn.beta;delta=dyn.delta
c2=np.cos(q[1]);s2=np.sin(q[1]);c1=np.cos(q[0]);c12=np.cos(q[0]+q[1])
g1_coeff=dyn.m1*dyn.lc1*9.81+dyn.m2*dyn.l1*9.81; g2_coeff=dyn.m2*dyn.lc2*9.81
theta=np.array([alpha,beta,delta,g1_coeff,g2_coeff])
Y=np.zeros((2,5))
Y[0]=[qdd[0], 2*c2*qdd[0]+c2*qdd[1]-s2*qd[1]*(2*qd[0]+qd[1]), qdd[1], c1, c12]
Y[1]=[0, c2*qdd[0]+s2*qd[0]**2, qdd[0]+qdd[1], 0, c12]
tau_check=Y@theta
print(f"τ=Yθ 成立? {np.allclose(tau, tau_check, atol=1e-10)}")
