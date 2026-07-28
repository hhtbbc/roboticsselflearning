"""
状态估计 (State Estimation)

包含：
- 卡尔曼滤波 (KF)
- 扩展卡尔曼滤波 (EKF)
- 无迹卡尔曼滤波 (UKF)
- 粒子滤波 (Particle Filter / SIR)
"""

import numpy as np
from typing import Callable, Tuple, List


# =============================================================================
# 卡尔曼滤波 (Kalman Filter)
# =============================================================================

class KalmanFilter:
    """
    线性卡尔曼滤波

    系统模型:
        x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
        z_t     = C x_t + v_t,           v_t ~ N(0, R)

    用法:
        kf = KalmanFilter(A, B, C, Q, R)
        kf.predict(u)    # 预测步
        kf.update(z)     # 更新步
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 mu: np.ndarray = None, Sigma: np.ndarray = None):
        self.A, self.B, self.C = A, B, C
        self.Q, self.R = Q, R
        self.n = A.shape[0]
        self.m = C.shape[0]

        self.mu = mu if mu is not None else np.zeros(self.n)
        self.Sigma = Sigma if Sigma is not None else np.eye(self.n)

    def predict(self, u: np.ndarray = None):
        """预测步"""
        if u is None:
            u = np.zeros(self.B.shape[1] if self.B.ndim > 1 else 1)
        self.mu = self.A @ self.mu + self.B @ u
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q
        return self.mu, self.Sigma

    def update(self, z: np.ndarray):
        """更新步（Joseph 形式协方差，数值稳定性更好）"""
        y = z - self.C @ self.mu  # innovation
        S = self.C @ self.Sigma @ self.C.T + self.R

        # 卡尔曼增益（用 solve 替代 inv）
        K = np.linalg.solve(S, self.C @ self.Sigma).T  # K = Σ C^T S^{-1}

        # 均值更新
        self.mu = self.mu + K @ y

        # Joseph 形式协方差: P = (I-KC)P(I-KC)^T + KRK^T
        I_KH = np.eye(self.n) - K @ self.C
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ self.R @ K.T
        # 对称化
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        return self.mu, self.Sigma, K

    def step(self, z: np.ndarray, u: np.ndarray = None):
        """预测 + 更新 一步完成"""
        self.predict(u)
        return self.update(z)


# =============================================================================
# 扩展卡尔曼滤波 (Extended Kalman Filter)
# =============================================================================

class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波

    非线性系统:
        x_{t+1} = f(x_t, u_t) + w_t,   w_t ~ N(0, Q)
        z_t     = h(x_t) + v_t,         v_t ~ N(0, R)

    A_t = ∂f/∂x|_{mu_{t-1}, u_t}  (状态转移雅可比)
    C_t = ∂h/∂x|_{mû_t}           (观测雅可比)
    """

    def __init__(self, f: Callable, h: Callable,
                 A_func: Callable, C_func: Callable,
                 Q: np.ndarray, R: np.ndarray,
                 mu: np.ndarray = None, Sigma: np.ndarray = None,
                 residual_fn: Callable = None,
                 state_injection_fn: Callable = None,
                 state_normalization_fn: Callable = None):
        self.f, self.h = f, h
        self.A_func, self.C_func = A_func, C_func
        self.Q, self.R = Q, R
        self.n = Q.shape[0]
        self.m = R.shape[0]
        # 可选：角度/流形状态的自定义残差、注入和归一化函数
        self.residual_fn = residual_fn or (lambda z, z_pred: z - z_pred)
        self.state_injection_fn = state_injection_fn or (lambda mu, dx: mu + dx)
        self.state_normalization_fn = state_normalization_fn or (lambda x: x)

        self.mu = mu if mu is not None else np.zeros(self.n)
        self.Sigma = Sigma if Sigma is not None else np.eye(self.n)

    def predict(self, u: np.ndarray = None):
        """预测步：在 μ_{t-1} 处线性化 f"""
        if u is None:
            u = np.zeros(1)

        A_t = self.A_func(self.mu, u)
        self.mu = self.state_normalization_fn(self.f(self.mu, u))
        self.Sigma = A_t @ self.Sigma @ A_t.T + self.Q

        return self.mu, self.Sigma

    def update(self, z: np.ndarray):
        """更新步：在 μ̂_t 处线性化 h（Joseph 形式）"""
        C_t = self.C_func(self.mu)

        y = self.residual_fn(z, self.h(self.mu))
        S = C_t @ self.Sigma @ C_t.T + self.R
        K = np.linalg.solve(S, C_t @ self.Sigma).T  # K = Σ C^T S^{-1}

        self.mu = self.state_normalization_fn(
            self.state_injection_fn(self.mu, K @ y)
        )
        # Joseph 形式: P = (I-KC)P(I-KC)^T + KRK^T
        I_KH = np.eye(self.n) - K @ C_t
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ self.R @ K.T
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        return self.mu, self.Sigma, K

    def step(self, z: np.ndarray, u: np.ndarray = None):
        """一步预测+更新"""
        self.predict(u)
        return self.update(z)


# =============================================================================
# 粒子滤波 (Particle Filter / SIR)
# =============================================================================

class ParticleFilter:
    """
    序贯重要性重采样 (SIR) 粒子滤波

    用法:
        pf = ParticleFilter(n_particles, dim, f, h, Q, R)
        pf.predict(u)
        pf.update(z)
        pf.resample()
    """

    def __init__(self, n_particles: int, dim: int,
                 f: Callable, h: Callable,
                 process_noise_std: np.ndarray,
                 obs_noise_std: np.ndarray,
                 bounds: np.ndarray = None,
                 rng: np.random.RandomState = None,
                 periodic_dims: List[int] = None):
        self.N = n_particles
        self.dim = dim
        self.f, self.h = f, h
        self.proc_std = np.atleast_1d(process_noise_std)
        self.obs_std = np.atleast_1d(obs_noise_std)
        self.bounds = bounds
        self.rng = rng if rng is not None else np.random.RandomState()
        self.periodic_dims = set(periodic_dims or [])

        # 初始化粒子
        self.particles = np.zeros((n_particles, dim))
        self.weights = np.ones(n_particles) / n_particles

    def initialize(self, mean: np.ndarray, cov: np.ndarray):
        """从高斯分布初始化粒子"""
        self.particles = self.rng.multivariate_normal(mean, cov, self.N)
        self.weights = np.ones(self.N) / self.N

    def initialize_uniform(self, bounds: np.ndarray):
        """均匀初始化（用于全局定位）"""
        for d in range(self.dim):
            self.particles[:, d] = self.rng.uniform(
                bounds[d, 0], bounds[d, 1], self.N)
        self.weights = np.ones(self.N) / self.N

    def predict(self, u: np.ndarray = None):
        """预测步：对每个粒子施加过程模型。

        约定：过程模型 f(x, u) 是确定性函数。
        过程噪声由 ParticleFilter 在此方法中添加。
        不要在 f(x, u) 内部添加随机噪声。
        """
        if u is None:
            u = np.zeros(1)

        for i in range(self.N):
            self.particles[i] = self.f(self.particles[i], u)
            self.particles[i] += self.rng.normal(0, self.proc_std)

        # 周期维度 wrap
        for d in self.periodic_dims:
            self.particles[:, d] = np.arctan2(
                np.sin(self.particles[:, d]),
                np.cos(self.particles[:, d])
            )

        # 裁剪到边界（非周期维度）
        if self.bounds is not None:
            for d in range(self.dim):
                if d not in self.periodic_dims:
                    self.particles[:, d] = np.clip(
                        self.particles[:, d],
                        self.bounds[d, 0], self.bounds[d, 1]
                    )

    def update(self, z: np.ndarray):
        """更新步：用观测似然更新权重（log-sum-exp 稳定版）。

        使用对数似然避免数值下溢：
            w_i = w_i * exp(log_lik_i)
        然后归一化。
        """
        log_weights = np.log(np.maximum(self.weights, 1e-300))

        for i in range(self.N):
            z_pred = self.h(self.particles[i])
            residual = z - z_pred
            # 高斯对数似然（忽略常数项，因为归一化时会消去）
            log_likelihood = -0.5 * np.sum((residual / self.obs_std) ** 2)
            log_weights[i] += log_likelihood

        # log-sum-exp 归一化
        log_max = np.max(log_weights)
        w = np.exp(log_weights - log_max)
        w_sum = np.sum(w)
        if w_sum > 1e-300:
            self.weights = w / w_sum
        else:
            self.weights = np.ones(self.N) / self.N

    def neff(self) -> float:
        """有效粒子数"""
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        """系统重采样（低方差重采样）"""
        N = self.N
        w = self.weights

        # 系统重采样
        r = self.rng.uniform(0, 1/N)
        c = np.cumsum(w)
        idx = np.zeros(N, dtype=int)

        i, j = 0, 0
        while i < N:
            u = r + i / N
            while u > c[j]:
                j += 1
            idx[i] = j
            i += 1

        self.particles = self.particles[idx].copy()
        self.weights = np.ones(N) / N

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """加权均值与协方差估计。

        周期维度使用 circular mean（atan2 方法），
        非周期维度使用普通加权均值。
        """
        mean = np.average(self.particles, axis=0, weights=self.weights)
        # 周期维度使用 circular mean
        for d in self.periodic_dims:
            cos_sum = np.sum(self.weights * np.cos(self.particles[:, d]))
            sin_sum = np.sum(self.weights * np.sin(self.particles[:, d]))
            mean[d] = np.arctan2(sin_sum, cos_sum)
        diff = self.particles - mean
        # 周期维度 wrap 差值
        for d in self.periodic_dims:
            diff[:, d] = np.arctan2(np.sin(diff[:, d]), np.cos(diff[:, d]))
        cov = np.cov(diff.T, aweights=self.weights)
        return mean, cov

    def step(self, z: np.ndarray, u: np.ndarray = None):
        """一步：预测 → 更新 → 重采样"""
        self.predict(u)
        self.update(z)
        if self.neff() < self.N / 2:
            self.resample()


# =============================================================================
# UKF 辅助函数
# =============================================================================
# 注意：以下仅为 Sigma 点生成器（辅助函数），不含完整 predict/update 循环。
# 完整 UKF 需要额外实现权重计算、加权均值和协方差传播。

def ukf_sigma_points(mu: np.ndarray, Sigma: np.ndarray,
                     kappa: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 UKF sigma 点 (2n+1 个)，使用 Cholesky 分解。

    参数:
        mu: (n,) 均值
        Sigma: (n,n) 协方差 (正定)
        kappa: 缩放参数 (通常 3-n 或 0)

    返回:
        sigma_points: (2n+1, n) sigma 点
        w_mean: (2n+1,) 均值权重
        w_cov: (2n+1,) 协方差权重
    """
    n = len(mu)
    lam = kappa

    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = mu

    # 权重
    w_mean = np.full(2*n + 1, 0.5 / (n + lam))
    w_mean[0] = lam / (n + lam)
    w_cov = w_mean.copy()
    w_cov[0] += (1.0 - 0.0**2 + 2.0)  # default: alpha=1, beta=2
    # 简化: 标准 UKF w_cov[0] = lam/(n+lam) + (1 - α² + β)

    # 使用 Cholesky 分解 (数值稳定性优于 sqrtm)
    try:
        L = np.linalg.cholesky((n + lam) * Sigma)
    except np.linalg.LinAlgError:
        # 协方差非正定时，使用对称化 + 小正则化
        Sigma_reg = 0.5 * (Sigma + Sigma.T) + np.eye(n) * 1e-8
        L = np.linalg.cholesky((n + lam) * Sigma_reg)

    for i in range(n):
        sigma_points[i + 1] = mu + L[:, i]
        sigma_points[i + 1 + n] = mu - L[:, i]

    return sigma_points, w_mean, w_cov
