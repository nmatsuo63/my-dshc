# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel
from common.time_layers import TimeBiLSTM
from common.seq2seq import Seq2seq
from common.seq2seq import Decoder

class BiEncoder:
    """
    エンコーダ
    LSTMは双方向
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 初期値の設定
        embed_W = rn(V, D) / 100 # 小さな値で初期化
        
        # 順方向LSTMの初期パラメータ
        lstm_Wx_f = rn(D, 4 * H) * np.sqrt(2/(D+H)) # Xavierの初期値
        lstm_Wh_f = rn(H, 4 * H) * np.sqrt(2/(H+H)) # Xavierの初期値
        lstm_b_f = np.zeros(4 * H)
        # 逆方向LSTMの初期パラメータ
        lstm_Wx_b = rn(D, 4 * H) * np.sqrt(2/(D+H)) # Xavierの初期値
        lstm_Wh_b = rn(H, 4 * H) * np.sqrt(2/(H+H)) # Xavierの初期値
        lstm_b_b = np.zeros(4 * H)  
        
        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        # 双方向LSTMを定義
        self.lstm = TimeBiLSTM(lstm_Wx_f, lstm_Wh_f, lstm_b_f, lstm_Wx_b, lstm_Wh_b, lstm_b_b, stateful=False)
    
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
    

class BiSeq2seq(Seq2seq):
    """
    エンコーダが双方向LSTMになったseq2seqモデル
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # エンコーダ(双方向LSTM)
        self.encoder = BiEncoder(V, D, H)
        # デコーダ
        self.decoder = Decoder(V, D, H*2)# 双方向の中間層を引数に取るため、Hを2倍しておく
        # ソフトマックス+損失
        self.softmax = TimeSoftmaxWithLoss()
        # パラメータ、勾配をそれぞれまとめる
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

