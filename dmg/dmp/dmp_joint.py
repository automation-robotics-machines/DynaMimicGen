from typing import List, Union, Tuple

import numpy as np

from dmg.dmp import CanonicalSystem


class JointDMP:
    def __init__(self, NDOF: int = 7, n_bfs: int = 10, alpha: float = 48.0, beta: float = Union[float, None],
                 cs_alpha: Union[float, None] = None, cs: Union[CanonicalSystem, None] = None):
        self.NDOF = NDOF
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha/2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2

        # Scaling factor
        self.Dp = np.identity(self.NDOF)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((self.NDOF, self.n_bfs))

        # Initial- and goal positions
        self.p0 = np.zeros(self.NDOF)
        self.gp = np.zeros(self.NDOF)

        # Reset
        self.p = self.p0.copy()
        self.dp = np.zeros(self.NDOF)
        self.ddp = np.zeros(self.NDOF)
        self.train_p = None
        self.train_d_p = None
        self.train_dd_p = None

    def step(self, x, dt: float, tau: float, FX) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration main equation 
        if FX == True:
            self.ddp = (self.alpha * (self.beta * (self.gp - self.p) - tau * self.dp) + fp(x)) / tau**2
        else:
            self.ddp = (self.alpha * (self.beta * (self.gp - self.p) - tau * self.dp)) / tau**2
        
        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp

    def rollout(self, ts: List[float], tau: Union[float, List[float]], FX: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts)  # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, self.NDOF))
        dp = np.empty((n_steps, self.NDOF))
        ddp = np.empty((n_steps, self.NDOF))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i], FX)

        return p, dp, ddp

    def reset(self):
        self.p = self.p0.copy()
        self.dp = np.zeros(self.NDOF)
        self.ddp = np.zeros(self.NDOF)

    def train(self, positions: np.ndarray, ts: List[float], tau: Union[float, List[float]]) -> None:
        p = positions

        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")

        # Initial- and goal positions
        self.p0 = p[0]
        self.gp = p[-1]

        # Differential time vector
        dt = np.gradient(ts)[:, np.newaxis]

        # Scaling factor
        self.Dp = np.diag(self.gp - self.p0)
        # Dp_inv = np.linalg.inv(self.Dp)
        Dp_inv = np.linalg.pinv(self.Dp)

        # Desired velocities and accelerations
        d_p = np.gradient(p, axis=0) / dt
        dd_p = np.gradient(d_p, axis=0) / dt

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Dp_inv.dot(tau**2 * dd_p[j] - self.alpha * (self.beta * (self.gp - p[j]) - tau * d_p[j]))

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
        self.p0 = y0
        self.gp = g

        # Scaling factor
        self.Dp = np.diag(self.gp - self.p0)
