import numpy as np

class Simulation():
# Simulation studies
# Input:
#   parameters for the factor model
#   m: number of subjects
#   n: number of time points
#   p: number of features
#   r: number of latent factors
#   factor_design: form of the factor model, choose from:
#           1. 'linear': linear factor model
#           2. 'logistic': logistic factor model
#   rho: parameters in AR(1) covariance structure to sample model errors
#   parameters for the regression model
#   s: number of true features (level of sparsity)
#   A: signal amplitude
#   link_design: form of the link function, choose from:
#           1. 'linear': linear model
#           2. 'nonlinear': noninear sin*exp model
    def __init__(self, m=1, n=1000, p=500, r=3, factor_design='linear', rho=0.9, s=10, A=10, link_design='linear'):
        self.m = m
        self.n = n
        self.p = p
        self.r = r
        self.factor_design = factor_design
        self.rho = rho
        self.s = s
        self.A = A
        self.link_design = link_design

    @staticmethod
    def ar1_cov(rho, size, sigma):
        if rho == 0:
            return (sigma ** 2) * np.identity(size)
        cov = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                cov[i][j] = (sigma ** 2) * (rho ** abs(i - j))
                cov[j][i] = (sigma ** 2) * (rho ** abs(i - j))
        return cov

    # Design matrix generation from factor model
    # Output:
    #   [factor matrix, loading matrix, design matrix]
    def dmg(self):
    
        Fa_raw = np.transpose(np.random.multivariate_normal(mean=[0] * self.n, cov=self.ar1_cov(self.rho, self.n, 1), size=self.r))
        Fa = np.zeros((self.n, self.r))
        Fa[0] = Fa_raw[0]
        Fa[-1] = 0.3 * Fa_raw[-2] + 0.7 * Fa_raw[-1]
        for i in range(1, self.n - 1):
            Fa[i] = 0.3 * Fa_raw[i - 1] + 0.7 * Fa_raw[i]
    
        if self.factor_design == 'linear':
            Lam = np.random.randn(self.r, self.p)
            E = np.random.randn(self.n, self.p)
            X = Fa @ Lam + E
        elif self.factor_design == 'logistic':
            def logistic(f, lam):
                # f : r-dim
                # lam : (r + 2)-dim
                return lam[0] / (1 + np.exp(lam[1] - np.dot(lam[2:], f)))
    
            Lam = np.random.randn(self.r + 2, self.p)
            FL = np.array([[logistic(f, lam) for lam in Lam.T] for f in Fa])
            E = np.random.randn(self.n, self.p)
            X = FL + E
    
        return [Fa, Lam, X]

    # Response vector generation for multiple subjects
    # Output:
    #   [coefficient vector, response vector, noiseless response]
    def rvg(self, X_ms, Fa_ms):
        m, n, p = X_ms.shape
        btrue = np.random.choice([self.A, -self.A], size=self.s)
        bfalse = np.repeat(0, p - self.s)
        beta = np.concatenate((btrue, bfalse)).reshape((p, 1))
        np.random.shuffle(beta)
    
        y_ms, nl_y_ms = [], []
        for X, Fa in zip(X_ms, Fa_ms):
            epsilon = np.random.multivariate_normal(mean=[0] * n, cov=self.ar1_cov(self.rho, n, 1)).reshape((n, 1))
            if self.link_design == 'linear':
                nl_y = X @ beta
                y = nl_y + epsilon
            elif self.link_design == 'nonlinear':
                nl_y = np.sin(X @ beta) * np.exp(X @ beta)
                y = nl_y + epsilon
            y_ms.append(y)
            nl_y_ms.append(nl_y)
        return [beta, np.array(y_ms), np.array(nl_y_ms)]

    def simulate(self):
        X_ms, Fa_ms = [], []
        for _ in range(self.m):
            GX = self.dmg()
            Xi = GX[2]
            Xi /= np.sqrt(np.sum(Xi ** 2, axis = 0))
            X_ms.append(Xi)
            Fa_ms.append(GX[0])
        X_ms, Fa_ms = np.array(X_ms), np.array(Fa_ms)
        Gy = self.rvg(X_ms, Fa_ms)
        X = X_ms
        y = Gy[1]
        true_beta = Gy[0]
        return X, y, true_beta

if __name__ == '__main__':
    import os
    import json
    import argparse
    
    import numpy as np
    import pandas as pd

    from collections import defaultdict
    from DeepLINK_T import DeepLINK_T
    from utility import pow, fdp
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--x_design", dest="x_design", default='logistic', help='factor model design')
    argparser.add_argument("--y_design", dest="y_design", default='linear', help='link function design')
    argparser.add_argument("--r", dest="r", default=3, help='number of factors', type=int)
    argparser.add_argument("--m", dest="m", default=1, help='number of subjects', type=int)
    argparser.add_argument("--n", dest="n", default=1000, help='number of time points', type=int)
    argparser.add_argument("--p", dest="p", default=500, help='number of features', type=int)
    argparser.add_argument("--s", dest="s", default=10, help='number of true signals', type=int)
    argparser.add_argument("--rho", dest="rho", default=0.9, help='parameter in the AR(1) covariance structure', type=float)
    argparser.add_argument("--amplitude", dest="amplitude", default=10, help='amplitude of the true signals', type=float)
    argparser.add_argument("--q", dest="q", default=0.2, help='targeted FDR level', type=float)
    argparser.add_argument("--it", dest="it", default=50, help='number of iterations for running DeepLINK-T', type=int)
    argparser.add_argument("--n_bottleneck", dest="n_bottleneck", default=15, help='number of bottleneck dimension in the autoencoder', type=int)
    argparser.add_argument("--aut_epoch", dest="aut_epoch", default=1000, help='number of autoencoder training epochs', type=int)
    argparser.add_argument("--aut_lr", dest="aut_lr", default=0.001, help='learning rate for the autoencoder', type=float)
    argparser.add_argument("--aut_norm", dest="aut_norm", default='', help='normalization for the autoencoder (either bn or ln)')
    argparser.add_argument("--mlp_epoch", dest="mlp_epoch", default=1000, help='number of prediction training epochs', type=int)
    argparser.add_argument("--mlp_lr", dest="mlp_lr", default=0.001, help='learning rate for the prediction network', type=float)
    argparser.add_argument("--output_path", dest="output_path")
    
    args = argparser.parse_args()
    print(
        'factor model design: ', args.x_design, '\n',
        'link function design: ', args.y_design, '\n',
        'number of factors: ', args.r, '\n',
        'number of subjects: ', args.m, '\n',
        'number of time points: ', args.n, '\n',
        'number of features: ', args.p, '\n',
        'number of true signals: ', args.s, '\n',
        'amplitude of the true signals: ', args.amplitude, '\n',
        'targeted FDR level: ', args.q, '\n',
        'number of iterations: ', args.it, '\n',
        'number of bottleneck dimension in the autoencoder: ', args.n_bottleneck, '\n',
        'normalization in autoencoder: ', args.aut_norm, '\n',
        'number of autoencoder training epochs: ', args.aut_epoch, '\n',
        'learning rate for the autoencoder: ', args.aut_lr, '\n',
        'number of prediction training epochs: ', args.mlp_epoch, '\n',
        'learning rate for the prediction network: ',args.mlp_lr, '\n'
    )

    result = np.repeat(np.repeat([[0]], 4, 0), args.it + 2, axis=1)
    # r_est = np.zeros(args.it, dtype=int)
    colnames = ['mean', 'sd'] + [str(obj) for obj in range(1, args.it + 1)]
    result = pd.DataFrame(result, index=['FDR', 'Power', 'FDR+', 'Power+'], columns=colnames)

    for i in range(args.it):
        print('Run_' + str(i + 1))
        sim = Simulation(m=args.m, n=args.n, p=args.p, r=args.r, factor_design=args.x_design, rho=args.rho, s=args.s, A=args.amplitude, link_design=args.y_design)
        X, y, true_beta = sim.simulate()
        print("input shape: ", X.shape)
        print("response shape:", y.shape)

        dlt = DeepLINK_T(X, y,
                 bottleneck_dim=args.n_bottleneck, ae_lr=args.aut_lr, ae_epoch=args.aut_epoch, ae_norm=args.aut_norm,
                 stats_lr=args.mlp_lr, stats_epoch=args.mlp_epoch, q=args.q)
        selected_features = dlt.infer()
        result.iloc[2, i + 2] = fdp(selected_features, true_beta)
        result.iloc[3, i + 2] = pow(selected_features, true_beta)
        # result.to_csv(args.output_path, index=True, header=True, sep=',')

    result.iloc[:, 0] = np.mean(result.iloc[:, 2:], axis=1)
    result.iloc[:, 1] = np.std(result.iloc[:, 2:], axis=1, ddof=1)
    result.to_csv(args.output_path, index=True, header=True, sep=',')


