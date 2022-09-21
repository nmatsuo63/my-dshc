# coding: utf-8
# import sys
# sys.path.append('..')
import numpy as np
from common.time_layers import *

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # 乱数生成関数を別名で定義する
        rn = np.random.randn

        # パラメータの初期化
        embed_W = rn(V, D) / 100 # 小さな値で初期化する
        rnn_Wx = rn(D, H) * np.sqrt(2/(D+H)) # Xavierの初期値
        rnn_Wh = rn(H, H) * np.sqrt(2/(H+H)) # Xavierの初期値
        rnn_b = np.zeros(H)
        affine_W = rn(H, V) * np.sqrt(2/(H+V)) # Xavierの初期値
        affine_b = np.zeros(V)

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W), # 単語埋め込みレイヤ
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True), # RNNレイヤ. 中間層の状態を引き継ぐため、statefulをTrueにしておく
            TimeAffine(affine_W, affine_b) # Affineレイヤ
        ]
        self.loss_layer = TimeSoftmaxWithLoss() # 損失レイヤ

        # すべてのパラメータと勾配をそれぞれ1つのリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params # リストを結合する
            self.grads += layer.grads # リストを結合する

    def predict(self, xs):
        """
        予測関数
        """
        # 損失レイヤ以外の全てのレイヤについて計算する
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        """
        順伝播計算
        xs : 入力の単語ID, 配列形状は(ミニバッチ数、時間数)
        """
        # 損失レイヤ以外の全てのレイヤについて計算する
        xs = self.predict(xs)
        
        # 損失レイヤを計算する
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        """
        逆伝播計算
        """
        # 損失レイヤの逆伝播計算
        dout = self.loss_layer.backward(dout)
        
        # 損失レイヤ以外の逆伝播計算
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            
        return dout