import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization, LayerNormalization, Masking
from tensorflow.keras.models import Sequential
from pairwise_connected_layer import PairwiseConnected
from itertools import combinations
from tensorflow.keras.callbacks import EarlyStopping
from missing import compact, to_full_grid, tail_mask, fill_between

# Knockoff matrix generation by autoencoder for multiple subjects
# Input:
#   X: design matrix (shape of [m, n, p])
#   r: bottleneck layer dimension (default: 15)
#   lr: learning rate of the LSTM autoencoder (default: 0.01)
#   met: activation method (default: 'tanh')
#   epoch: number of training epochs (default: 500)
#   loss: loss function used in training (default: 'mean_squared_error')
#   verb: verbose level (default: 2)
# Output:
#   Xnew: [X, X_knockoff]

class DeepLINK_T():
    def __init__(self, X, y,
                 bottleneck_dim=64, ae_lr=0.001, ae_epoch=1000, ae_norm=None, ae_stacked=True,
                 stats_lstm_units=128, stats_lr=0.001, stats_epoch=1000, 
                 q=0.2, ko_plus=True, fit_type='regression', response_type='sequence'
                ):
        self.X = X
        self.y = y
        
        # parameters for LSTM autoencoder
        self.bottleneck_dim = bottleneck_dim
        self.ae_lr = ae_lr
        self.ae_epoch = ae_epoch
        self.ae_norm = ae_norm
        self.stacked = ae_stacked

        # parameters for LSTM prediction network
        self.stats_lstm_units = stats_lstm_units
        self.stats_lr = stats_lr
        self.stats_epoch = stats_epoch

        self.q = q
        self.ko_plus = ko_plus
        self.fit_type = fit_type
        self.response_type = response_type

    # LSTM autoencoder
    # Input:
    #   X: design matrix with shape (m, n, p)
    #   verb: verbose level (default: 2)
    #   weights: optional (m, n) 0/1 time-step weights; steps with weight 0 are left out of
    #            the loss and of the error variance (default: None, all steps)
    # Output:
    #   Xnew: concatenation of original variables and knockoff variables
    def knockoff_construct(self, X, verb=2, weights=None):
        m, n, p = X.shape
        def add_normalization(model):
            if self.ae_norm == 'bn':
                model.add(BatchNormalization())
            elif self.ae_norm == 'ln':
                model.add(LayerNormalization(axis=1))
            return model

        if self.stacked == True:
            # stacked LSTM
            autoencoder = Sequential()
            autoencoder.add(BatchNormalization())
            autoencoder.add(LSTM(self.bottleneck_dim, activation='tanh', input_shape=(n, p), return_sequences=True))
            autoencoder.add(BatchNormalization())
            autoencoder.add(LSTM(p, activation='tanh', return_sequences=True))
            autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.ae_lr), loss='mean_squared_error')
            autoencoder.fit(X, X, epochs=self.ae_epoch, verbose=verb, sample_weight=weights)
            C = autoencoder.predict(X)
        else:
             # LSTM autoencoder
            autoencoder = Sequential()
            autoencoder = add_normalization(autoencoder)
            autoencoder.add(LSTM(self.bottleneck_dim, activation='tanh', return_sequences=False))
            autoencoder.add(RepeatVector(n))
            autoencoder = add_normalization(autoencoder)
            autoencoder.add(LSTM(self.bottleneck_dim, activation='tanh', return_sequences=True))
            autoencoder.add(TimeDistributed(Dense(p)))
            autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.ae_lr), loss='mean_squared_error')
            autoencoder.fit(X, X, epochs=self.ae_epoch, verbose=verb, sample_weight=weights)
            C = autoencoder.predict(X)
    
    
        # construct X_knockoff
        E = X - C
        # sigma = np.sqrt(np.sum(E ** 2) / (n * p))
        # X_ko = C + sigma * np.random.randn(n, p)
        # Xnew = np.hstack((X, X_ko))
        if weights is None:
            sigma = np.array([sig * np.ones([n, p]) for sig in np.sqrt(np.sum(E ** 2, axis=(1, 2)) / (n * p))])
        else:
            w = np.asarray(weights, dtype=float)
            sigma = np.array([sig * np.ones([n, p]) for sig in np.sqrt(np.sum(E ** 2 * w[:, :, None], axis=(1, 2)) / (np.sum(w, axis=1) * p))])
        X_ko = C + sigma * np.random.randn(m, n, p)
        # Xnew = np.hstack((X, X_ko))
        Xnew = np.concatenate((X, X_ko), axis=-1)
        return Xnew

    # LSTM prediction network
    # Input:
    #   X: design matrix ([X, X_knockoff] with shape (m, n, 2p))
    #   y: response vector
    #   verb: verbose level (default: 2)
    #   weights: optional (m, n) 0/1 time-step weights of the end padding; padded steps are left out
    #            of the loss (sequence response) or skipped by the LSTM (scaler response)
    # Output:
    #   W: knockoff statistics
    def knockoff_stats(self, X, y, verb=2, weights=None):
        m = X.shape[0]
        n = X.shape[1]
        p = X.shape[2] // 2
        n_lstm_units = self.stats_lstm_units
        if self.fit_type == 'classification':
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_activation = None
            loss = 'mean_squared_error'
        return_sequences = True if self.response_type == 'sequence' else False
        
        # LSTM prediction network
        dp = Sequential()
        dp.add(TimeDistributed(PairwiseConnected(input_shape=(2 * p,))))
        if weights is not None and not return_sequences:
            dp.add(Masking(mask_value=0.0))
        dp.add(Dense(p, activation='elu', kernel_regularizer=keras.regularizers.l1(l1=0.0005)))
        dp.add(BatchNormalization())
        dp.add(LSTM(n_lstm_units, activation='tanh', return_sequences=return_sequences))
        dp.add(Dense(1, activation=output_activation))
        dp.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.stats_lr))
        dp.fit(X, y, epochs=self.stats_epoch, verbose=verb, sample_weight=weights if return_sequences else None)
    
        # Calculate importance of input variables using weights from each layer.
        weights = dp.get_weights()
        pc = np.zeros((2 * p, p))
        for i in range(p):
            pc[i][i] = weights[0][i]
            pc[p + i][i]= weights[0][p + i]
    
        w1 = pc @ (weights[1] * np.tile(weights[3], (weights[1].shape[0], 1))) @ weights[7][:,:n_lstm_units] @ weights[10]
        w2 = pc @ (weights[1] * np.tile(weights[3], (weights[1].shape[0], 1))) @ weights[7][:, n_lstm_units:2*n_lstm_units] @ weights[10]
        w3 = pc @ (weights[1] * np.tile(weights[3], (weights[1].shape[0], 1))) @ weights[7][:, 2*n_lstm_units:3*n_lstm_units] @ weights[10]
        w4 = pc @ (weights[1] * np.tile(weights[3], (weights[1].shape[0], 1))) @ weights[7][:, 3*n_lstm_units:] @ weights[10]
        
        w = np.sum(np.hstack([w1, w2, w3, w4])**2, axis=1)
        W = w[:p] - w[p:]
    
        return W
    
    
    # Feature selection with knockoff/knockoff+ threshold
    # Input:
    #   W: knockoff statistics
    #   q: FDR level
    #   ko_plus: indicate whether to use knockoff+ (True) or
    #            knockoff (False) threshold [default: True]
    # Output:
    #   array of discovered variables
    def knockoff_select(self, W, q, ko_plus):
        # find the knockoff threshold T among the |W_j| > 0, so T > 0
        t = np.sort(abs(W))
        t = t[t > 0]
        if ko_plus:
            ratio = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t]
        else:
            ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t]
        ind = np.where(np.array(ratio) <= q)[0]
        if len(ind) == 0:
            T = float('inf')
        else:
            T = t[ind[0]]
    
        # set of discovered variables
        return np.where(W >= T)[0]

    # Missing time points are rows of X that are NaN in every feature. The knockoffs are then
    # generated from the observed time points, and X, the knockoffs and a sequence response are
    # Kalman-imputed at the missing time points up to each subject's last observed one; the time
    # points after it are end padding.
    def infer(self):
        mask = np.isnan(self.X).all(axis=2)
        if not mask.any():
            self.Xnew = self.knockoff_construct(self.X, verb=0)

            # compute knockoff statistics
            self.W = self.knockoff_stats(self.Xnew, self.y, verb=0)
        else:
            if (np.isnan(self.X).any(axis=2) & ~mask).any():
                raise ValueError('a time point must be observed or missing in all features')
            p = self.X.shape[2]
            Xc, wc = compact(self.X, mask)
            X_ko = to_full_grid(self.knockoff_construct(Xc, verb=0, weights=wc)[..., p:], mask)
            self.Xnew = np.concatenate((fill_between(self.X, mask), fill_between(X_ko, mask)), axis=-1)
            y = self.y
            if self.response_type == 'sequence':
                y = fill_between(np.reshape(y, mask.shape + (-1,)), mask)
            tail = tail_mask(mask)
            weights = (~tail).astype(float) if tail.any() else None

            # compute knockoff statistics
            self.W = self.knockoff_stats(self.Xnew, y, verb=0, weights=weights)

        # feature selection
        selected_features = self.knockoff_select(self.W, self.q, ko_plus=self.ko_plus)
        return selected_features
