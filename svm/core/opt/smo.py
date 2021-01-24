import numpy as np

from svm.core.kernels import get_kernel_function


class SequentialMinimalOptimizer:
    def __init__(self, X, y, C, kernel, tol=1e-3):
        self.X = X
        self.y = y
        self.C = C
        self.tol = tol
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(self.n).T
        self.e_cache = np.zeros(self.m).T

        # self.kernel = get_kernel_function(kernel)
        self.kernel = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                self.kernel[i, j] = kernel(X[i], X[j])

    def take_step(self, i1, i2):
        if i1 == i2:
            return 0
        alpha1 = self.alphas[i1]
        y1 = self.y[i1]

        if (alpha1 > 0) and (alpha1 < self.C):
            E1 = self.e_cache[i1]
        else:
            # E1 = self.X[i1] @ self.w + self.b - y1
            E1 = self.w.T @ self.X[i1].T + self.b - y1

        alpha2 = self.alphas[i2]
        y2 = self.y[i2]
        E2 = self.e_cache[i2]

        s = y1 * y2
        if y1 == y2:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)

        if L == H:
            return 0

        eta = (
            2 * self.kernel[i1, i2] - self.kernel[i1, i1] - self.kernel[i2, i2]
        )
        if eta < 0:
            a2 = alpha2 - y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            c1 = -eta / 2
            c2 = y2 * (E1 - E2) + eta * alpha2
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if Lobj > Hobj + self.tol:
                a2 = L
            elif Lobj < Hobj - self.tol:
                a2 = H
            else:
                a2 = alpha2

        if a2 < 1e-8:
            a2 = 0
        elif a2 > self.C - 1e-18:
            a2 = self.C

        if abs(a2 - alpha2) < self.tol * (a2 + alpha2 + self.tol):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        if (a1 > 0) and (a1 < self.C):
            b_new = (
                self.b
                + E1
                + y1 * (a1 - alpha1) * self.kernel[i1, i1]
                + y2 * (a2 - alpha2) * self.kernel[i1, i2]
            )
        elif (a2 > 0) and (a2 < self.C):
            b_new = (
                self.b
                + E2
                + y1 * (a1 - alpha1) * self.kernel[i1, i2]
                + y2 * (a2 - alpha2) * self.kernel[i2, i2]
            )
        else:
            b1 = (
                self.b
                + E1
                + y1 * (a1 - alpha1) * self.kernel[i1, i1]
                + y2 * (a2 - alpha2) * self.kernel[i1, i2]
            )
            b2 = (
                self.b
                + E2
                + y1 * (a1 - alpha1) * self.kernel[i1, i2]
                + y2 * (a2 - alpha2) * self.kernel[i2, i2]
            )
            b_new = (b1 + b2) / 2

        self.b = b_new

        # self.w = self.w + y1*(a1 - alpha1)*self.X[i1] + y2*(a2 - alpha2)*self.X[i2]

        self.alphas[i1] = a1
        self.alphas[i2] = a2

        self.w = (self.alphas * self.y) @ self.X

        for i in range(self.m):
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                # self.e_cache[i] = self.X[i] @ self.w + self.b - self.y[i]
                self.e_cache[i] = self.w.T @ self.X[i].T + self.b - self.y[i]

        return 1

    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        if (alpha2 > 0) and (alpha2 < self.C):
            E2 = self.e_cache[i2]
        else:
            # E2 = self.X[i2] @ self.w + self.b - y2
            E2 = self.w.T @ self.X[i2].T + self.b - y2
            self.e_cache[i2] = E2

        r2 = E2 * y2

        # Second choice heuristic
        max_delta_E = 0
        i1 = -1

        if (r2 < -self.tol and alpha2 < self.C) or (
            r2 > self.tol and alpha2 > 0
        ):
            for i in range(self.m):
                if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                    if i == i2:
                        continue
                    E1 = self.e_cache[i]
                    delta_E = abs(E1 - E2)
                    if delta_E > max_delta_E:
                        max_delta_E = delta_E
                        i1 = i

        if i1 >= 0:
            if self.take_step(i1, i2):
                return 1

        # Random border i1
        random_index = np.random.permutation(self.m)
        for i in random_index:
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                if i == i2:
                    continue
                if self.take_step(i, i2):
                    return 1

        random_index = np.random.permutation(self.m)
        for i in random_index:
            if i == i2:
                continue
            if self.take_step(i, i2):
                return 1

        return 0

    def solve(self, max_iter=5):
        num_changed = 0
        examine_all = 1
        iter = 0

        while iter < max_iter:
            num_changed = 0
            if examine_all:
                for i in range(self.m):
                    num_changed += self.examine_example(i)
            else:
                for i in range(self.m):
                    if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                        num_changed += self.examine_example(i)

            if num_changed == 0:
                iter += 1

            if examine_all:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

        return self.alphas, self.w, self.b
