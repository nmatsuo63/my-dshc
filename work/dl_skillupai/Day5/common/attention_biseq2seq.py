# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel
from common.time_layers import TimeBiLSTM
from common.seq2seq import Seq2seq
from common.attention_seq2seq import AttentionDecoder

class AttentionBiEncoder:
    """
    アテンション付きエンコーダ
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
        hs = self.lstm.forward(xs) # 全ての中間層の情報を返す
        
        return hs

    def backward(self, dhs):
        """
        逆伝播
        dhs : 勾配
        """
        # LSTMレイヤ
        dout = self.lstm.backward(dhs) #  Decoderから伝わってきた勾配を全て伝える
        
        # 単語埋め込みレイヤ
        dout = self.embed.backward(dout)
        
        return dout
    

class AttentionBiSeq2seq(Seq2seq):
    """
    エンコーダが双方向LSTMになったアテンション付きseq2seqモデル
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # アテンション付きエンコーダ(双方向LSTM)
        self.encoder = AttentionBiEncoder(V, D, H)
        # アンテション付きデコーダ
        self.decoder = AttentionDecoder(V, D, H*2)# 双方向の中間層を引数に取るため、Hを2倍しておく
        # ソフトマックス+損失
        self.softmax = TimeSoftmaxWithLoss()
        # パラメータ、勾配をそれぞれまとめる
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
