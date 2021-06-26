import numpy as np
from scipy import optimize
import time


class KernelSvmClassifier:

    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.supportVectors = None

    def fit(self, X, y, DEBUG=False):
        start_time = time.time()
        if DEBUG:
            print('[KernelSvmClassifier] DEBUG: fit() started --------------------')

        N = len(y)

        # Create Gram matrix of k(x) y:
        gramXX = np.apply_along_axis(lambda x1: np.apply_along_axis(lambda x2: self.kernel(x1, x2), 1, X), 1, X)  # 2 for loops.
        yp = y.reshape(-1, 1)
        yy = yp @ yp.T
        gramXXyy = gramXX * yy
        if DEBUG:
            print(f'[KernelSvmClassifier] DEBUG: X shape: {X.shape}, y shape: {y.shape}')
            print(f'[KernelSvmClassifier] DEBUG: gramXXyy shape: {gramXXyy.shape}')

        # Lagrange dual objective function (to maximize!)
        def ld_obj(gram, alpha):
            return alpha.sum() - 0.5 * alpha @ (alpha @ gram)

        # Partial derivative of ld_obj on alpha
        def d_ldobj_d_alpha(gram, alpha):
            return np.ones_like(alpha) - alpha @ gram

        # cost function and its gradient - to minimizes
        def cost_func(a):
            return -ld_obj(gramXXyy, a)

        def cost_grad(a):
            return -d_ldobj_d_alpha(gramXXyy, a)

        # Constraints:
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))
        constraints = ({'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},           # y @ alpha = 0
                       {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})    # 0 <= alpha <= C

        # Minimizing the **negative** dual
        results = optimize.minimize(fun=cost_func,
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=cost_grad,
                                   constraints=constraints)
        self.alpha = results.x

        # store support vectors and y*alpha vals to use in inference
        epsilon = 1e-8  # Max distance to become a support vector
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportAlphaY = y[supportIndices] * self.alpha[supportIndices]

        if DEBUG:
            print(f'[KernelSvmClassifier] DEBUG: supportVectors shape: {self.supportVectors.shape}')
            print(f'[KernelSvmClassifier] DEBUG: alpha shape: {self.alpha.shape}')

        print(f'[KernelSvmClassifier] Info: fit() finished, time elapsed: {time.time() - start_time}')
        return


    def predict(self, X, DEBUG=False):
        """ Predict y values in {-1, 1} """
        if DEBUG:
            print(f'[KernelSvmClassifier] Debug: predict() started --------------------')
            print(f'[KernelSvmClassifier] Debug: X.shape: {X.shape}')

        def predict_sample(x):
            x1 = np.apply_along_axis(lambda s: self.kernel(s, x), 1, self.supportVectors)
            # Calc kernel of the sample with each support vector, return vectorized results
            x2 = x1 * self.supportAlphaY # Multiply by alpha*y of each vector
            return np.sum(x2)

        d = np.apply_along_axis(predict_sample, 1, X) # for each vector in X, run predict_sample
        return 2 * (d > 0) - 1
