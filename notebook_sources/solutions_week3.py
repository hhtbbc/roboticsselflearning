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

# %%
import numpy as np; import sys; sys.path.insert(0, '..')
from src.robotics_learning.trajectory import quintic_trajectory, trapezoidal_trajectory
from src.robotics_learning.planning import create_grid_map, astar
from src.robotics_learning.dynamics import TwoLinkArmDynamics
from src.robotics_learning.control import computed_torque_control

# %% [markdown]
# # 第 3 周练习解答 — 轨迹、规划与控制 (NB13-18)

# %% [markdown]
# ## NB13: 五次多项式 q0=0, qf=π/2, T=1s → a₀=a₁=a₂=0, a₃=5π, a₄=−7.5π, a₅=3π

# %%
T=1.0; t,q,_,_,_=quintic_trajectory(0,np.pi/2,0,0,0,0,T,0.01)
print(f"q(0)={q[0]:.4f}, q(T)={q[-1]:.4f}")

# %% [markdown]
# ## NB13: 梯形剖面判断 — v_max=2, a_max=3, dq=1.0 → Δq_triangle=4/3>1.0 → 三角剖面

# %%
t,q,_,_=trapezoidal_trajectory(0,1.0,2.0,3.0,0.01)
print(f"三角剖面 T={t[-1]:.3f}s")

# %% [markdown]
# ## NB15: A* 8-连通 — admissible 条件: h(n) ≤ h*(n)

# %%
grid=create_grid_map(20,20,[(8,4,4,12)])
path,_=astar(grid,(2,10),(17,10),heuristic=lambda a,b:np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))
print(f"A* path: {len(path) if path else 0} steps")

# %% [markdown]
# ## NB18: CTC 闭环验证 — ë+K_vė+K_pe=0

# %%
dyn=TwoLinkArmDynamics(m1=1.,m2=1.,l1=1.,l2=0.8,g=9.81)
q=np.array([0.3,0.4]);qd=np.array([0.5,-0.3])
qd_d=np.array([0.5,0.2]);qdd_d=np.array([1.,0.5]);qddd_d=np.zeros(2)
tau=computed_torque_control(qd_d,qdd_d,qddd_d,q,qd,np.array([100,80]),np.array([20,16]),
                             dyn.mass_matrix,dyn.coriolis_matrix,dyn.gravity_vector)
qdd=dyn.forward_dynamics(q,qd,tau)
print(f"ë+K_vė+K_pe = {np.linalg.norm(qdd-qddd_d-np.diag([20,16])@(qdd_d-qd)-np.diag([100,80])@(qd_d-q)):.2e}")
