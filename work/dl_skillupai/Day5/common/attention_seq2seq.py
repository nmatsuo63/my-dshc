# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.seq2seq import Encoder, Seq2seq
from common.attention_layer import TimeAttention
from common.layers import Tanh


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs) # 全ての中間層の情報を返す
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs) #  Decoderから伝わってきた勾配を全て伝える
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    """
    アテンション付きデコーダ
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期値
        embed_W = rn(V, D) / 100 # 小さな値で初期化.   他の方法としては、一様分布でサンプリングしノード数で割るという方法もある
        lstm_Wx = rn(D, 4 * H) * np.sqrt(2/(D+H)) # Xavierの初期値
        lstm_Wh = rn(H, 4 * H) * np.sqrt(2/(H+H)) # Xavierの初期値
        lstm_b = np.zeros(4 * H)
        affine_W_c = rn(2*H, V) * np.sqrt(2/(2*H+V)) # Xavierの初期値
        affine_b_c = np.zeros(V)
        affine_W_s = rn(V, V) * np.sqrt(2/(V+V)) # Xavierの初期値
        affine_b_s = np.zeros(V)

        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine_c = TimeAffine(affine_W_c, affine_b_c)
        self.tanh = Tanh()
        self.affine_s = TimeAffine(affine_W_s, affine_b_s)
        layers = [self.embed, self.lstm, self.attention, self.affine_c, self.tanh, self.affine_s]

        # パラメータ、勾配をまとめる
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        """
        順伝播
        xs : 入力データ(教師強制用)
        enc_hs : エンコーダで計算された中間状態
        """
        # 中間状態をセット
        h = enc_hs[:,-1] # 最後だけ使う
        self.lstm.set_state(h)

        # 単語埋め込みレイヤ
        out = self.embed.forward(xs)
        
        # LSTMレイヤ
        dec_hs = self.lstm.forward(out)
        
        # アテンションレイヤ
        c = self.attention.forward(enc_hs, dec_hs) # エンコーダの中間状態とLSTMの中間状態を使って、重みcを求める
        
        # 結合
        out = np.concatenate((c, dec_hs), axis=2)
        
        # affine_c
        out = self.affine_c.forward(out)
        
        # tanh
        out = self.tanh.forward(out)        
        
        # affine_s
        out = self.affine_s.forward(out)        

        return out

    def backward(self, dscore):
        """
        逆伝播
        """
        dout = self.affine_s.backward(dscore)
        dout = self.tanh.backward(dout)
        dout = self.affine_c.backward(dout)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        """
        予測
        """
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            
            # 結合
            out = np.concatenate((c, dec_hs), axis=2)

            # affine_c
            out = self.affine_c.forward(out)

            # tanh
            out = self.tanh.forward(out)        

            # affine_s
            out = self.affine_s.forward(out) 

            sample_id = np.argmax(out.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
