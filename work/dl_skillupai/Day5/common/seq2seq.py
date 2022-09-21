# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.time_layers import TimeEmbedding,TimeLSTM,TimeAffine,TimeSoftmaxWithLoss
from common.base_model import BaseModel

class Encoder:
    """
    エンコーダ
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 初期値の設定
        embed_W = rn(V, D) / 100 # 小さな値で初期化
        lstm_Wx = rn(D, 4 * H) * np.sqrt(2/(D+H)) # Xavierの初期値
        lstm_Wh = rn(H, 4 * H) * np.sqrt(2/(H+H)) # Xavierの初期値
        lstm_b = np.zeros(4 * H)

        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        # パラメータ、勾配をそれぞれまとめる
        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        """
        順伝播
        xs : 入力データ
        """
        # 単語埋め込みレイヤ
        xs = self.embed.forward(xs)
        
        # LSTMレイヤ
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :] # 最後の中間状態だけreturnする

    def backward(self, dh):
        """
        逆伝播
        dh : 勾配
        """
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class Decoder:
    """
    デコーダ
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 初期値の設定
        embed_W = rn(V, D) / 100 # 小さな値で初期化
        lstm_Wx = rn(D, 4 * H) * np.sqrt(2/(D+H)) # Xavierの初期値
        lstm_Wh = rn(H, 4 * H) * np.sqrt(2/(H+H)) # Xavierの初期値
        lstm_b = np.zeros(4 * H)
        affine_W = rn(H, V) * np.sqrt(2/(H+V)) # Xavierの初期値
        affine_b = np.zeros(V)

        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        # パラメータ、勾配をそれぞれまとめる
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        """
        順伝播
        xs : デコーダへの入力データ(教師強制用)
        h : エンコーダから出力された中間状態
        """
        # エンコーダ中間状態をセット
        self.lstm.set_state(h)

        # 単語埋め込みレイヤ
        out = self.embed.forward(xs)
        
        # LSTMレイヤ
        out = self.lstm.forward(out)
        
        # 全結合レイヤ
        score = self.affine.forward(out)
        
        return score

    def backward(self, dscore):
        """
        逆伝播
        """
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h, start_id, sample_size):
        """
        予測
        h : 中間層のデータ
        start_id : 頭の区切り文字のid
        sample_size : 出力させる単語列の長さ
        """
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            """
            sample_sizeだけ繰り返す
            sample_id : 直前に出力された単語のid, 初期値は頭区切り文字
            """
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseModel):
    """
    seq2seq
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # レイヤの定義
        self.encoder = Encoder(V, D, H) 
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        # パラメータ、勾配をそれぞれまとめる
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        """
        順伝播
        xs : 入力データ 
        ts : 正解データ
        """
        # デコーダ側の入出力データ
        # 教師強制で学習させるため、入力と出力は同じデータにする
        decoder_xs = ts[:, :-1]# 最後の単語を捨てる
        decoder_ts = ts[:, 1:] # 頭の区切り文字を捨てる

        # エンコーダ
        h = self.encoder.forward(xs)
        # デコータ
        score = self.decoder.forward(decoder_xs, h)
        # 損失
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        """
        逆伝播
        """
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        """
        予測
        xs : 入力単語列
        start_id : 頭の区切り文字のid
        sample_size : 出力させる単語列の長さ
        """
        # エンコーダ
        h = self.encoder.forward(xs)
        # デコーダ
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
