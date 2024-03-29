import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from svm.core.kernels import get_kernel_function


def calc_gamma_value(x, gamma="scale"):
    if x.ndim == 1:
        n_features = 1
    else:
        _, n_features = x.shape

    if gamma == "scale":
        return 1 / (n_features * np.var(x))
    elif gamma == "auto":
        return 1 / n_features
    else:
        return float(gamma)


class SVM_SMO:
    def __init__(self, kernel, C, gamma, coef0, tol, degree, **param):
        self.C = C
        self.tol = tol
        self.kernel_str = kernel
        self.gamma = gamma
        self.coef = coef0
        self.eps = 0.01
        self.degree = degree

    def init_params(self, X, y):
        self.X = X
        self.y = y

        if self.kernel_str != "linear":
            self.gamma = calc_gamma_value(X, self.gamma)
        self.m, self.n = X.shape[0], X.shape[1]
        self.alphas = np.zeros(self.m)
        self.b = 0
        kernel = get_kernel_function(
            self.kernel_str, self.degree, self.gamma, self.coef
        )
        if self.kernel_str == "rbf":
            self.kernel = lambda x, z: rbf_kernel(x, z.T)
        else:
            self.kernel = lambda x, z: kernel(x.T, z)

        self.K = self.kernel(X, X.T)
        self.e_cache = np.full((self.m), np.inf)

    def project(self, Xtrain, Ytrain, Xtest):
        return (
            np.dot(self.alphas * Ytrain, self.kernel(Xtrain, Xtest.T)) - self.b
        )

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        self.init_params(X, y)

        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i2 in range(self.m):
                    num_changed += self.examine_example(i2)
            else:
                satisfied_idx = np.where(
                    (self.alphas != 0) & (self.alphas != self.C)
                )[0]
                for i2 in satisfied_idx:
                    num_changed += self.examine_example(i2)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        self.set_support_vectors()
        self.X = None
        self.y = None

    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        if self.e_cache[i2] != np.inf:
            E2 = self.e_cache[i2]
        else:
            E2 = self.decision_function(np.asarray([self.X[i2]])) - y2
        r2 = E2 * y2
        if ((r2 < -self.tol) and (alpha2 < self.C)) or (
            (r2 > self.tol) and (alpha2 > 0)
        ):
            if (
                len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)])
                > 1
            ):
                if self.e_cache[i2] > 0:
                    i1 = np.argmin(self.e_cache)
                else:
                    i1 = np.argmax(self.e_cache)
                if self.take_step(i1, i2):
                    return 1
            # H2: find suitable i1 inside the boundaries
            random_index = np.random.permutation(self.m)
            for i1 in random_index:
                if self.alphas[i1] > 0 and self.alphas[i1] < self.C:
                    if i1 == i2:
                        continue
                    if self.take_step(i1, i2):
                        return 1
            # H3: find suitable i1 on all alphas
            random_index = np.random.permutation(self.m)
            for i1 in random_index:
                if i1 == i2:
                    continue
                if self.take_step(i1, i2):
                    return 1
        return 0

    def take_step(self, i1, i2):
        alpha1 = self.alphas[i1]
        y1 = self.y[i1]
        if self.e_cache[i1] != np.inf:
            E1 = self.e_cache[i1]
        else:
            E1 = self.decision_function(np.asarray([self.X[i1]])) - y1
        # E1 = self.decision_function(self.X[i1, :]) - y1
        alpha2 = self.alphas[i2]
        y2 = self.y[i2]
        if self.e_cache[i2] != np.inf:
            E2 = self.e_cache[i2]
        else:
            E2 = self.decision_function(np.asarray([self.X[i2]])) - y2
        # E2 = self.decision_function(self.X[i2, :]) - y2

        s = y1 * y2
        if y1 == y2:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        if L == H:
            return 0

        eta = 2 * self.K[i1, i2] - self.K[i1, i1] - self.K[i2, i2]
        if eta < 0:
            a2 = alpha2 - y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            print("eta > 0.")
            # c1 = eta / 2.0
            # c2 = y2 * (E1 - E2) - eta * alpha2
            # Lobj = c1 * L * L + c2 * L
            # Hobj = c1 * H * H + c2 * L
            alphas_adj = self.alphas.copy()
            alphas_adj[i2] = L
            # objective function output with a2 = L
            Lobj = self.objective(alphas_adj, self.X, self.y)
            alphas_adj[i2] = H
            # objective function output with a2 = H
            Hobj = self.objective(alphas_adj, self.X, self.y)

            if Lobj > (Hobj + self.eps):
                a2 = L
            elif Lobj < (Hobj - self.eps):
                a2 = H
            else:
                a2 = alpha2

        if np.abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        b1 = (
            E1
            + y1 * (a1 - alpha1) * self.K[i1, i1]
            + y2 * (a2 - alpha2) * self.K[i1, i2]
            + self.b
        )
        b2 = (
            E2
            + y1 * (a1 - alpha1) * self.K[i1, i2]
            + y2 * (a2 - alpha2) * self.K[i2, i2]
            + self.b
        )

        if a1 > 0 and a1 < self.C:
            bnew = b1
        elif a2 > 0 and a2 < self.C:
            bnew = b2
        else:
            bnew = (b1 + b2) / 2.0

        self.alphas[i1] = a1
        self.alphas[i2] = a2

        self.b = bnew
        for i in range(self.m):
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                self.e_cache[i] = (
                    self.decision_function(np.asarray([self.X[i]])) - self.y[i]
                )

        return 1

    def decision_function(self, X):
        if not hasattr(self, "support_vectors_"):
            return (
                np.dot(
                    self.alphas * self.y,
                    self.kernel(self.X, X.T),
                )
                - self.b
            )
        else:
            return (
                np.dot(
                    self.support_vectors_alphas_ * self.support_vectors_ys_,
                    self.kernel(self.support_vectors_, X.T),
                )
                - self.b
            )

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def set_support_vectors(self, support_threshold=1e-5):
        mask = self.alphas > support_threshold
        self.support_vectors_alphas_ = self.alphas[mask]
        self.support_vectors_ = self.X[mask]
        self.support_vectors_ys_ = self.y[mask]

    def objective(self, alpha, X, Y):
        return np.sum(alpha) - 0.5 * np.sum(
            np.outer(Y, Y) * self.kernel(X, X) * np.outer(alpha, alpha)
        )
