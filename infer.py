import os
import json
import argparse

import numpy as np
import pandas as pd

from DeepLINK_T import DeepLINK_T
from collections import defaultdict

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_path", dest="input_path", help='path to the explanatory variables (tensor in .npy format (number of subjects, number of time points, number of feature))')
argparser.add_argument("--response_path", dest="response_path", help='path to the response variables (matrix in .npy format)')
argparser.add_argument("--output_path", dest="output_path", default='output.json', help='output path for selection results (in .json format with key=feature, value=list of selected ranks in each run)')
argparser.add_argument("--n_iter", dest="n_iter", type=int, default=200, help='number of iterations for running DeepLINK-T')
argparser.add_argument("--q", dest="q", default=0.2, type=float, help='targeted FDR level')
argparser.add_argument("--n_bottleneck", dest="n_bottleneck", default=15, help='number of bottleneck dimension in the autoencoder', type=int)
argparser.add_argument("--aut_epoch", dest="aut_epoch", default=1000, help='number of autoencoder training epochs', type=int)
argparser.add_argument("--aut_lr", dest="aut_lr", default=0.001, help='learning rate for the autoencoder', type=float)
argparser.add_argument("--aut_norm", dest="aut_norm", default='', help='normalization for the autoencoder (either bn or ln)')
argparser.add_argument("--mlp_epoch", dest="mlp_epoch", default=1000, help='number of prediction training epochs', type=int)
argparser.add_argument("--mlp_lr", dest="mlp_lr", default=0.001, help='learning rate for the prediction network', type=float)
argparser.add_argument("--fit_type", dest="fit_type", default='regression', help='either regression or classification')
argparser.add_argument("--response_type", dest="response_type", default='sequence', help='either sequence or scaler')

if __name__ == '__main__':
    args = argparser.parse_args()

    X = np.load(args.input_path)
    y = np.load(args.response_path)
    feat_importance = defaultdict(list)
    for i in range(args.n_iter):
        print('Run_' + str(i + 1))
        dlt = DeepLINK_T(X, y,
                         bottleneck_dim=args.n_bottleneck, ae_lr=args.aut_lr, ae_epoch=args.aut_epoch, ae_norm=args.aut_norm,
                         stats_lr=args.mlp_lr, stats_epoch=args.mlp_epoch, q=args.q, fit_type=args.fit_type, response_type=args.response_type)
        selected_features = dlt.infer()
        for i, feat in enumerate(selected_features):
            feat_importance[int(feat)].append(i)
    with open(args.output_path, 'w') as f:
        json.dump(feat_importance, f)
