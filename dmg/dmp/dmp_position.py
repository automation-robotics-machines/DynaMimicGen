from __future__ import division, print_function

from typing import Tuple

import numpy as np

from dmg.dmp import CanonicalSystem
from dmg.utils import math_util


class PositionDMP:
    def __init__(self, n_bfs=10, alpha: float = 48.0, beta: float = None, cs_alpha=None, cs=None, roto_dilatation=False):
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha / 2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c) ** 2
        # self.h = 0.0025
        # phi_c = 0.5
        # x_min = 0.01
        # k = - 4 * np.log(phi_c) / (np.log(x_min) ** 2)
        # self.h = k * n_bfs ** 2 / (self.c ** 2)

        # Scaling factor
        self.Dp = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial- and goal positions
        self._p0 = np.zeros(3)
        self._gp = np.zeros(3)

        self._p0_train = np.zeros(3)
        self._gp_train = np.zeros(3)

        self._R_fx = np.identity(3)

        # Reset
        self.p = self._p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)
        self.train_p = None
        self.train_d_p = None
        self.train_dd_p = None

        self._roto_dilatation = roto_dilatation

    def step(self, x, dt, tau, force_disturbance=np.array([0, 0, 0])) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        self.ddp = (self.alpha * (self.beta * (self._gp - self.p) - tau * self.dp) + self._R_fx @ fp(x) + force_disturbance) / tau ** 2

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt
        # print(f"p: {self.p}")
        return self.p, self.dp, self.ddp

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

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i])

        return p, dp, ddp

    def reset(self):
        self.p = self._p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)

    def train(self, positions, ts, tau):
        p = positions

        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")

        # Initial- and goal positions
        self._p0 = p[0]
        self._gp = p[-1]

        self._p0_train = p[0]
        self._gp_train = p[-1]

        # Differential time vector
        dt = np.gradient(ts)[:, np.newaxis]

        # Scaling factor
        #self.Dp = np.diag(self.gp - self.p0)
        #Dp_inv = np.linalg.inv(self.Dp)
        Dp_inv = np.identity(3)

        # Desired velocities and accelerations
        d_p = np.gradient(p, axis=0) / dt
        dd_p = np.gradient(d_p, axis=0) / dt

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Dp_inv.dot(tau ** 2 * dd_p[j] - self.alpha * (self.beta * (self._gp - p[j]) - tau * d_p[j]))

        A = np.stack([features(xj) for xj in x])
        f = np.stack([forcing(j) for j in range(len(ts))])

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p

    def set_trained(self, w: np.ndarray, c: np.ndarray, h: np.ndarray, y0: np.ndarray, g: np.ndarray) -> None:
        self.w = w
        self.c = c
        self.h = h
        self._p0 = y0
        self._gp = g

        # Scaling factor
        self.Dp = np.diag(self._gp - self._p0)

    def _update_goal_change_parameters(self):
        # print("Updating goal rotation parameters")
        self._sg = np.linalg.norm(self._gp_train - self._p0_train) / np.linalg.norm(self._gp - self._p0)

        v_new = np.array(self._gp) - np.array(self._p0)
        v_new = math_util.normalize_vector(v_new)
        v_old = np.array(self._gp_train) - np.array(self._p0_train)
        v_old = math_util.normalize_vector(v_old)
        self._R_fx = math_util.calculate_rotation_between_vectors(v_old, v_new)
        # print("Position fx rotation: ", self._R_fx)

    @property
    def gp(self):
        return self._gp

    @gp.setter
    def gp(self, value):
        self._gp = value
        if self._roto_dilatation:
            self._update_goal_change_parameters()

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, value):
        self._p0 = value
        if self._roto_dilatation:
            self._update_goal_change_parameters()
