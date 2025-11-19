class CUSUM:
    def __init__(self, mu0, mu1, std, threshold):
        self.mu0 = mu0
        self.mu1 = mu1
        self.std = std
        self.threshold = threshold
        self._Sk = 0
        self._mk = float("inf")

    def reset(self):
        self._Sk = 0
        self._mk = float("inf")

    def addPoint(self, zk):
        sk = ((zk - self.mu0)**2 - (zk - self.mu1)**2) / (2 * (self.std**2))
        self._Sk += sk
        self._mk = min(self._Sk, self._mk)
        gk = self._Sk - self._mk
        return gk > self.threshold
