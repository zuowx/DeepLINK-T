# DeepLINK-T: deep learning inference for time series data using knockoffs and LSTM

DeepLINK-T is a variable selection framework that guarantees the false discovery rate (FDR) control in time series data. Three key ingredients for DeepLINK-T are 
1) a Long Short-Term Memory (LSTM) autoencoder for generating time series knockoff variables,
2) an LSTM prediction network using both original and knockoff variables,
3) the application of the knockoffs framework for variable selection with FDR control.

## Dependencies

DeepLINK-T requires Python 3 (>= 3.7.6) with the following packages:

- keras >= 2.4.3
- numpy >= 1.18.5
- pandas >= 1.1.0
- tensorflow >= 2.3.0


## Installation

Clone the github repository and enter DeepLINK-T directory with

```
  $ git clone https://github.com/zuowx/DeepLINK-T.git
  $ cd DeepLINK-T
```

Create a conda environment for DeepLINK-T and install all packages with

```
  $ conda create -n $ENV_NAME python=3.7.6
  $ conda activate $ENV_NAME
  $ pip install -r requirements.txt
```
    
## Usage

The inputs of DeepLINK-T are a data tensor (number of subjects $\times$ number of time points $\times$ number of features) and the corresponding response vector. Let the dimension of the data matrix be $m\times n\times p$. Then the response should be a array with dimensionality $m$ (scaler response) or a matrix with dimensionality $m\times n$ (sequence response). The input should be in '.npy' format and without row and column names. Another required argument is the output directory. The output of DeepLINK-T is a json file with key=feature, value=list of selected ranks in each run. Notice that feature index is 0-based.

Variable selection for regression task:

```
  $ python infer.py --input_path $INPUT_VARIABLES --response_path $RESPONSE --output_path $OUTPUT
```

Variable selection for classification task:

```
  $ python infer.py --input_path $INPUT_VARIABLES --response_path $RESPONSE --output_path $OUTPUT --fit_type classification
```

The output of DeepLINK-T is a json file with each key as a feature index and each value as a list of ranks in each iteration. The length of the value list could be less than the number of specified iterations.



### Options for infer.py

```
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        path to the explanatory variables (tensor in .npy
                        format (number of subjects, number of time points,
                        number of feature))
  --response_path RESPONSE_PATH
                        path to the response variables (matrix in .npy format)
  --output_path OUTPUT_PATH
                        output path for selection results (in .json format
                        with key=feature, value=list of selected ranks in each
                        run)
  --n_iter N_ITER       number of iterations for running DeepLINK-T
  --q Q                 targeted FDR level
  --n_bottleneck N_BOTTLENECK
                        number of bottleneck dimension in the autoencoder
  --aut_epoch AUT_EPOCH
                        number of autoencoder training epochs
  --aut_lr AUT_LR       learning rate for the autoencoder
  --aut_norm AUT_NORM   normalization for the autoencoder (either bn or ln)
  --mlp_epoch MLP_EPOCH
                        number of prediction training epochs
  --mlp_lr MLP_LR       learning rate for the prediction network
  --fit_type FIT_TYPE   either regression or classification
  --response_type RESPONSE_TYPE
                        either sequence or scaler
```

### An inference example

Run the following code and the output is in `test/test.json`:
```   
  $ python infer.py --input_path test/test_X.npy --response_path test/test_y.npy --output_path test/test.json
```

The complete example may take several hours to run on GPU. Users may use the following code for a quicker test:
```   
  $ python infer.py --input_path test/test_X.npy --response_path test/test_y.npy --output_path test/test.json --n_iter 1 --aut_epoch 50 --mlp_epoch 50
```

The inference can be finished in 1 minute if running on GPU. The output should be an empty `test.json` file in the `test` directory.

## Replicate studies in the paper

### Simulation studies

Codes used in the simulation studies are in `simulation.py`. Options for the script include

```
  -h, --help            show this help message and exit
  --x_design X_DESIGN   factor model design
  --y_design Y_DESIGN   link function design
  --r R                 number of factors
  --m M                 number of subjects
  --n N                 number of time points
  --p P                 number of features
  --s S                 number of true signals
  --rho RHO             parameter in the AR(1) covariance structure
  --amplitude AMPLITUDE
                        amplitude of the true signals
  --q Q                 targeted FDR level
  --it IT               number of iterations for running DeepLINK-T
  --n_bottleneck N_BOTTLENECK
                        number of bottleneck dimension in the autoencoder
  --aut_epoch AUT_EPOCH
                        number of autoencoder training epochs
  --aut_lr AUT_LR       learning rate for the autoencoder
  --aut_norm AUT_NORM   normalization for the autoencoder (either bn or ln)
  --mlp_epoch MLP_EPOCH
                        number of prediction training epochs
  --mlp_lr MLP_LR       learning rate for the prediction network
  --output_path OUTPUT_PATH
```

### Real data analyses

Data used in the real data analyses are in folder `real_data/`. Detailed information of each real-world dataset is in `real_data/README.md`


## Copyright and License Information

Copyright (C) 2024 University of Southern California

Authors: Wenxuan Zuo, Zifan Zhu, Yuxuan Du, Yi-Chun Yeh, Jed A. Fuhrman, Jinchi Lv, Yingying Fan, Fengzhu Sun

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

Commercial users should contact Dr. Fengzhu Sun (<fsun@usc.edu>) or Dr. Yingying Fan (<fanyingy@usc.edu>), copyright at University of Southern California.
