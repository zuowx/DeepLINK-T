import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Sequential
from pairwise_connected_layer import PairwiseConnected
from itertools import combinations
from tensorflow.keras.callbacks import EarlyStopping

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
                 bottleneck_dim=64, ae_lr=0.001, ae_epoch=1000, ae_norm=None,
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
    # Output:
    #   Xnew: concatenation of original variables and knockoff variables
    def knockoff_construct(self, X, verb=2):
        m, n, p = X.shape
        def add_normalization(model):
            if self.ae_norm == 'bn':
                model.add(BatchNormalization())
            elif self.ae_norm == 'ln':
                model.add(LayerNormalization(axis=1))
            return model
            
        # LSTM autoencoder
        autoencoder = Sequential()
        autoencoder = add_normalization(autoencoder)
        autoencoder.add(LSTM(self.bottleneck_dim, activation='tanh', return_sequences=False))
        autoencoder.add(RepeatVector(n))
        autoencoder = add_normalization(autoencoder)
        autoencoder.add(LSTM(self.bottleneck_dim, activation='tanh', return_sequences=True))
        autoencoder.add(TimeDistributed(Dense(p)))
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.ae_lr), loss='mean_squared_error')
        autoencoder.fit(X, X, epochs=self.ae_epoch, verbose=verb)
        C = autoencoder.predict(X)
    
        # construct X_knockoff
        E = X - C
        # sigma = np.sqrt(np.sum(E ** 2) / (n * p))
        # X_ko = C + sigma * np.random.randn(n, p)
        # Xnew = np.hstack((X, X_ko))
        sigma = np.array([sig * np.ones([n, p]) for sig in np.sqrt(np.sum(E ** 2, axis=(1, 2)) / (n * p))])
        X_ko = C + sigma * np.random.randn(m, n, p)
        # Xnew = np.hstack((X, X_ko))
        Xnew = np.concatenate((X, X_ko), axis=-1)
        return Xnew

    # LSTM prediction network
    # Input:
    #   X: design matrix ([X, X_knockoff] with shape (m, n, 2p))
    #   y: response vector
    #   verb: verbose level (default: 2)
    # Output:
    #   W: knockoff statistics
    def knockoff_stats(self, X, y, verb=2):
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
        dp.add(Dense(p, activation='elu', kernel_regularizer=keras.regularizers.l1(l1=0.0005)))
        dp.add(BatchNormalization())
        dp.add(LSTM(n_lstm_units, activation='tanh', return_sequences=return_sequences))
        dp.add(Dense(1, activation=output_activation))
        dp.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.stats_lr))
        dp.fit(X, y, epochs=self.stats_epoch, verbose=verb)
    
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
        # find the knockoff threshold T
        p = len(W)
        t = np.sort(np.concatenate(([0], abs(W))))
        if self.ko_plus:
            ratio = [(1 + sum(W <= -tt)) / max(1, sum(W >= tt)) for tt in t[:p]]
        else:
            ratio = [sum(W <= -tt) / max(1, sum(W >= tt)) for tt in t[:p]]
        ind = np.where(np.array(ratio) <= q)[0]
        if len(ind) == 0:
            T = float('inf')
        else:
            T = t[ind[0]]
    
        # set of discovered variables
        return np.where(W >= T)[0]

    def infer(self):
        self.Xnew = self.knockoff_construct(self.X, verb=0)

        # compute knockoff statistics
        self.W = self.knockoff_stats(self.Xnew, self.y, verb=0)

        # feature selection
        selected_features = self.knockoff_select(self.W, self.q, ko_plus=self.ko_plus)
        return selected_features
