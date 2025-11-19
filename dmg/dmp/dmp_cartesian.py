from typing import Tuple

import numpy as np
import quaternion

from dmg.dmp import CanonicalSystem, PositionDMP, QuaternionDMP


class CartesianDMP:
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None, roto_dilatation=False):
        self.n_bfs = n_bfs
        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else alpha/2)

        # Centres of the Gaussian basis functions
        self.position_dmp = PositionDMP(n_bfs, alpha, beta, cs=self.cs, roto_dilatation=roto_dilatation)
        self.quaternion_dmp = QuaternionDMP(n_bfs, alpha, beta, cs=self.cs, roto_dilatation=roto_dilatation)

        self.reset()

    def step(self, x, dt, tau, force_disturbance=np.array([0, 0, 0]), torque_disturbance=np.array([0, 0, 0])) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        quaternion.quaternion, np.ndarray, np.ndarray]:

        p, dp, ddp = self.position_dmp.step(x, dt, tau, force_disturbance)
        q, omega, d_omega = self.quaternion_dmp.step(x, dt, tau, torque_disturbance)

        return p, dp, ddp, q, omega, d_omega

    def rollout(self, ts, tau):
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts)  # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, 3))
        dp = np.empty((n_steps, 3))
        ddp = np.empty((n_steps, 3))

        q = np.empty((n_steps,), dtype=np.quaternion)
        omega = np.empty((n_steps, 3))
        d_omega = np.empty((n_steps, 3))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.position_dmp.step(x[i], dt[i], tau[i])
            q[i], omega[i], d_omega[i] = self.quaternion_dmp.step(x[i], dt[i], tau[i])

        return p, dp, ddp, q, omega, d_omega

    def reset(self):
        self.position_dmp.reset()
        self.quaternion_dmp.reset()
        self.cs.reset()

    def train(self, positions, quaternions, ts, tau):
        self.position_dmp.train(positions, ts, tau)
        self.quaternion_dmp.train(quaternions, ts, tau)
