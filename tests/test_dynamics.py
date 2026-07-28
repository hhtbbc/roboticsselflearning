"""动力学测试 — M(q) 性质, RNEA vs Lagrange, 能量守恒"""
import numpy as np
import sys; sys.path.insert(0, '.')
from src.robotics_learning.dynamics import TwoLinkArmDynamics, simulate_dynamics

RNG = np.random.RandomState(42)
ATOL = 1e-10


class TestMassMatrix:
    """M(q) 对称正定性测试"""

    def setup_method(self):
        self.dyn = TwoLinkArmDynamics(m1=1., m2=1., l1=1., l2=0.8, g=9.81)

    def test_symmetry(self):
        for _ in range(10):
            q = RNG.uniform(-np.pi, np.pi, 2)
            M = self.dyn.mass_matrix(q)
            assert np.allclose(M, M.T, atol=ATOL), f"M not symmetric at q={q}"

    def test_positive_definite(self):
        for _ in range(10):
            q = RNG.uniform(-np.pi, np.pi, 2)
            M = self.dyn.mass_matrix(q)
            eigvals = np.linalg.eigvalsh(M)
            assert np.all(eigvals > 0), f"M not PD at q={q}, eig={eigvals}"

    def test_Mdot_minus_2C_skew(self):
        """Ṁ - 2C 反对称性"""
        for _ in range(10):
            q = RNG.uniform(-np.pi, np.pi, 2)
            qd = RNG.uniform(-2, 2, 2)
            M = self.dyn.mass_matrix(q)
            C = self.dyn.coriolis_matrix(q, qd)
            # 数值差分 M_dot
            eps = 1e-6
            M_plus = self.dyn.mass_matrix(q + eps * qd)
            M_dot_num = (M_plus - M) / eps
            N = M_dot_num - 2 * C
            assert np.allclose(N + N.T, np.zeros((2,2)), atol=1e-2), \
                f"N not skew: {np.round(N+N.T, 6)}"
            # 二次型 q̇^T(Ṁ-2C)q̇ 应 ≈ 0
            assert abs(qd @ N @ qd) < 1e-4

    def test_gravity_equals_potential_gradient(self):
        """g(q) = ∂P/∂q (数值验证)"""
        eps = 1e-6
        for _ in range(5):
            q = RNG.uniform(-np.pi/2, np.pi/2, 2)
            g = self.dyn.gravity_vector(q)

            # 数值势能梯度
            P = lambda qv: (self.dyn.m1 * 9.81 * self.dyn.lc1 * np.sin(qv[0])
                          + self.dyn.m2 * 9.81 * (self.dyn.l1 * np.sin(qv[0])
                          + self.dyn.lc2 * np.sin(qv[0] + qv[1])))
            P0 = P(q)
            g_num = np.zeros(2)
            for i in range(2):
                qp = q.copy(); qp[i] += eps
                g_num[i] = (P(qp) - P0) / eps
            assert np.allclose(g, g_num, atol=1e-4), \
                f"g={g}, g_num={g_num}"


class TestInverseDynamics:
    """逆动力学一致性测试"""

    def setup_method(self):
        self.dyn = TwoLinkArmDynamics(m1=1., m2=1., l1=1., l2=0.8, g=9.81)

    def test_id_fd_roundtrip(self):
        """ID → FD → q̈ 应一致"""
        q = np.array([0.5, 0.3])
        qd = np.array([1.0, -0.5])
        qdd = np.array([2.0, -1.0])
        tau = self.dyn.inverse_dynamics(q, qd, qdd)
        qdd_fd = self.dyn.forward_dynamics(q, qd, tau)
        assert np.allclose(qdd, qdd_fd, atol=1e-10)

    def test_static_gravity(self):
        """静止时 (q̇=0, q̈=0): τ = g(q)"""
        q = np.array([np.pi/3, np.pi/6])
        tau = self.dyn.inverse_dynamics(q, np.zeros(2), np.zeros(2))
        g = self.dyn.gravity_vector(q)
        assert np.allclose(tau, g, atol=1e-10)


class TestEnergyConservation:
    """无外力时能量守恒测试"""

    def test_free_swing_energy(self):
        """无控制力矩 + 无阻尼 → 总能量应 (近似) 守恒"""
        dyn = TwoLinkArmDynamics(m1=1., m2=1., l1=1., l2=0.8, g=9.81)
        q0 = np.array([np.pi/3, -np.pi/4])
        qd0 = np.zeros(2)

        def zero_torque(t, q, qd):
            return np.zeros(2)

        t, q_hist, qd_hist, _ = simulate_dynamics(dyn, q0, qd0, zero_torque, 2.0, 0.005, 'rk4')

        # 总能量 E = K + P
        E = np.zeros(len(t))
        for i in range(len(t)):
            K = 0.5 * qd_hist[i] @ dyn.mass_matrix(q_hist[i]) @ qd_hist[i]
            P = (dyn.m1 * 9.81 * dyn.lc1 * np.sin(q_hist[i,0])
               + dyn.m2 * 9.81 * (dyn.l1 * np.sin(q_hist[i,0])
               + dyn.lc2 * np.sin(q_hist[i,0] + q_hist[i,1])))
            E[i] = K + P

        # 能量波动应在 1% 以内
        E_std = np.std(E) / (np.mean(np.abs(E)) + 1e-10)
        assert E_std < 0.02, f"Energy std/mean ratio = {E_std:.4f} (期望 < 0.02)"
