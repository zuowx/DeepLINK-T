# Standard normal epsilon, no division
import numpy as np
import random

class Simulation():
# Simulation studies
# Input:
#   parameters for the factor model
#   X: design matrix (shape of [m, n, p]); missing time points are rows of NaN
#   m: number of subjects
#   n: number of time points
#   p: number of features
#   rho: parameters in AR(1) covariance structure to sample model errors
#   parameters for the regression model
#   s: number of true features (level of sparsity)
#   A: signal amplitude; the nonzero coefficients are +-A * sqrt(n_obs / 1000), with n_obs the mean
#      number of observed time points per subject
#   link_design: form of the link function, choose from:
#           1. 'linear': linear model
#           2. 'nonlinear': noninear sin*exp model
    def __init__(self, X, rho=0.9, s=10, A=10, norm='l2', link_design='linear'):
        self.X = X
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.p = X.shape[2]
        self.rho = rho
        self.s = s
        self.A = A
        self.norm = norm
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

    # Response vector generation for multiple subjects, at the observed time points only
    # Output:
    #   [coefficient vector, response vector (NaN at the missing time points)]
    def rvg(self, X_ms):
        m, n, p = X_ms.shape
        mask = np.isnan(X_ms).all(axis=2)
        magnitude = self.A * np.sqrt((~mask).sum(axis=1).mean() / 1000)
        btrue = np.random.choice([magnitude, -magnitude], size=self.s)
        bfalse = np.repeat(0, p - self.s)
        beta = np.concatenate((btrue, bfalse)).reshape((p, 1))
        np.random.shuffle(beta)

        y_ms = np.full((m, n, 1), np.nan)
        for i in range(m):
            obs = ~mask[i]
            epsilon = np.random.multivariate_normal(mean=[0] * n, cov=self.ar1_cov(self.rho, n, 1)).reshape((n, 1))
            if self.link_design == 'linear':
                nl_y = X_ms[i, obs] @ beta
            elif self.link_design == 'nonlinear':
                nl_y = np.sin(X_ms[i, obs] @ beta) * np.exp(X_ms[i, obs] @ beta)
            y_ms[i, obs] = nl_y + epsilon[obs]
        return [beta, y_ms]

    def simulate(self):
        X_ms = np.array(self.X, dtype=float)
        ### Normalize each subject's data over its observed time points
        for X in X_ms:
            obs = ~np.isnan(X).all(axis=1)
            if self.norm == 'std':
                X[obs] -= np.mean(X[obs], axis=0)
                X[obs] /= np.std(X[obs], axis=0, ddof=1)
            elif self.norm == 'l2':
                X[obs] /= np.sqrt(np.sum(X[obs] ** 2, axis=0))
        Gy = self.rvg(X_ms)
        X = X_ms
        y = Gy[1]
        true_beta = Gy[0]
        return X, y, true_beta

if __name__ == '__main__':
    import os
    import json
    import argparse
    import re
    
    import numpy as np
    import pandas as pd

    from collections import defaultdict
    from DeepLINK_T import DeepLINK_T
    from utility import pow, fdp

    #from PCp1_numFactors import PCp1 as PCp1
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_path", dest="input_path", help='path to the explanatory variables (tensor in .npy format (number of subjects, number of time points, number of feature))')
    argparser.add_argument("--y_design", dest="y_design", default='linear', help='link function design')
    argparser.add_argument("--s", dest="s", default=10, help='number of true signals', type=int)
    argparser.add_argument("--rho", dest="rho", default=0.9, help='parameter in the AR(1) covariance structure', type=float)
    argparser.add_argument("--amplitude", dest="amplitude", default=10, help='amplitude of the true signals', type=float)
    argparser.add_argument("--norm", dest="norm", default='l2', help='normalization for the original dataset')
    argparser.add_argument("--q", dest="q", default=0.2, help='targeted FDR level', type=float)
    argparser.add_argument("--it", dest="it", default=50, help='number of iterations for running DeepLINK-T', type=int)
    argparser.add_argument("--n_bottleneck", dest="n_bottleneck", default=15, help='number of bottleneck dimension in the autoencoder', type=int)
    argparser.add_argument("--aut_epoch", dest="aut_epoch", default=1000, help='number of autoencoder training epochs', type=int)
    argparser.add_argument("--aut_lr", dest="aut_lr", default=0.001, help='learning rate for the autoencoder', type=float)
    argparser.add_argument("--aut_norm", dest="aut_norm", default='ln', help='normalization for the autoencoder (either bn or ln)')
    #argparser.add_argument("--aut_stacked", dest="aut_stacked", default="False", help='choose whether to use staked LSTM autoencoder')
    argparser.add_argument("--mlp_epoch", dest="mlp_epoch", default=1000, help='number of prediction training epochs', type=int)
    argparser.add_argument("--mlp_lr", dest="mlp_lr", default=0.001, help='learning rate for the prediction network', type=float)
    argparser.add_argument("--output_path", dest="output_path")
    
    args = argparser.parse_args()
    X0 = np.load(args.input_path)
    
    print(
        'link function design: ', args.y_design, '\n',
        'number of true signals: ', args.s, '\n',
        'amplitude of the true signals: ', args.amplitude, '\n',
        'normalization for the original dataset:', args.norm, '\n',
        'targeted FDR level: ', args.q, '\n',
        'number of iterations: ', args.it, '\n',
        'number of bottleneck dimension in the autoencoder: ', args.n_bottleneck, '\n',
        'normalization in autoencoder: ', args.aut_norm, '\n',
        'number of autoencoder training epochs: ', args.aut_epoch, '\n',
        'learning rate for the autoencoder: ', args.aut_lr, '\n',
        #'choose whether to use staked LSTM autoencoder: ', args.aut_stacked, '\n',
        'number of prediction training epochs: ', args.mlp_epoch, '\n',
        'learning rate for the prediction network: ', args.mlp_lr, '\n',
        'real dataset used: ', args.output_path, '\n'
    )

    result_dlt = np.repeat(np.repeat([[0]], 4, 0), args.it + 2, axis=1)
    result_dlt_ae = np.repeat(np.repeat([[0]], 4, 0), args.it + 2, axis=1)
    # r_est = np.zeros(args.it, dtype=int)
    colnames = ['mean', 'sd'] + [str(obj) for obj in range(1, args.it + 1)]
    result_dlt = pd.DataFrame(result_dlt, index=['', 'DLT', 'FDR+', 'Power+'], columns=colnames)
    result_dlt_ae = pd.DataFrame(result_dlt_ae, index=['', 'DLT_ae', 'FDR+', 'Power+'], columns=colnames)

    for i in range(args.it):
        print('Run_' + str(i + 1))
        sim = Simulation(X=X0, rho=args.rho, s=args.s, A=args.amplitude, norm=args.norm, link_design=args.y_design)
        X, y, true_beta = sim.simulate()
        # To ensure y in the form of (m, n, 1)
        print("input shape: ", X.shape)
        print("response shape:", y.shape)
        
        # Results of dlt_stacked
        dlt = DeepLINK_T(X, y,
                 bottleneck_dim=args.n_bottleneck, ae_lr=args.aut_lr, ae_epoch=args.aut_epoch, ae_stacked=True, stats_lr=args.mlp_lr, stats_epoch=args.mlp_epoch, q=args.q)
        selected_features = dlt.infer()
        print("dlt select:", selected_features)
        result_dlt.iloc[2, i + 2] = fdp(selected_features, true_beta)
        result_dlt.iloc[3, i + 2] = pow(selected_features, true_beta)

    # Results of dlt_ae
        dlt_ae = DeepLINK_T(X, y,
                 bottleneck_dim=args.n_bottleneck, ae_lr=args.aut_lr, ae_epoch=args.aut_epoch, ae_norm=args.aut_norm, ae_stacked=False, stats_lr=args.mlp_lr, stats_epoch=args.mlp_epoch, q=args.q)
        selected_features_ae = dlt_ae.infer()
        print("dlt_ae select:", selected_features_ae)
        result_dlt_ae.iloc[2, i + 2] = fdp(selected_features_ae, true_beta)
        result_dlt_ae.iloc[3, i + 2] = pow(selected_features_ae, true_beta)


    #result.iloc[:, 0] = np.mean(result.iloc[:, 2:], axis=1)
    result_dlt.iloc[:, 0] = np.mean(result_dlt.iloc[:, 2:], axis=1)
    result_dlt.iloc[:, 1] = np.std(result_dlt.iloc[:, 2:], axis=1, ddof=1)

    result_dlt_ae.iloc[:, 0] = np.mean(result_dlt_ae.iloc[:, 2:], axis=1)
    result_dlt_ae.iloc[:, 1] = np.std(result_dlt_ae.iloc[:, 2:], axis=1, ddof=1)
    result = pd.concat((result_dlt, result_dlt_ae), axis=0)
    
    # output_path = "~/DeepLINK-T_outputs/comp/in-place-pad/#_!_amp@_&.csv"
    #output_path = re.sub("%", str(int(args.s)), output_path)
    # output_path = re.sub("#", args.output_path, output_path)
    # output_path = re.sub("!", args.y_design, output_path)
    # output_path = re.sub("@", str(int(args.amplitude)), output_path)
    # output_path = re.sub("&", args.norm, output_path)
    
    result.to_csv(args.output_path, index=True, header=True, sep=',')


