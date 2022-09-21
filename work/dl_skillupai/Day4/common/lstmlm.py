# coding: utf-8
# import sys
# sys.path.append('..')
from common.time_layers import *
import numpy as np


class LSTMlm:
    '''
     LSTMレイヤを利用した言語モデル
    '''
    def __init__(self, vocab_size=10000, wordvec_size=650,
                 hidden_size=650):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = rn(V, D) / 100 # 小さな値で初期化する
        lstm_Wx1 = rn(D, 4 * H) * np.sqrt(2/(D+H)) # Xavierの初期値
        lstm_Wh1 = rn(H, 4 * H) * np.sqrt(2/(H+H)) # Xavierの初期値
        lstm_b1 = np.zeros(4 * H)
        affine_W = rn(H, V) * np.sqrt(2/(H+V)) # Xavierの初期値
        affine_b = np.zeros(V)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeAffine(affine_W, affine_b) 
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
