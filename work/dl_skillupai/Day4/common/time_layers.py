# coding: utf-8
import numpy as np
from common.layers import * 
from common.functions import sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):
        """
        Wx : 入力xにかかる重み
        Wh : １時刻前のhにかかる重み
        b : バイアス
        """
        
        # パラメータのリスト
        self.params = [Wx, Wh, b]
        
        # 勾配のリスト
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        順伝播計算
        """
        Wx, Wh, b = self.params
        
        # 行列の積　+　行列の積 + バイアス
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        
        # 活性化関数に入れる
        h_next = np.tanh(t)

        # 値の一時保存
        self.cache = (x, h_prev, h_next)
        
        return h_next

    def backward(self, dh_next):
        """
        逆伝播計算
        """
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanhでの逆伝播
        # dh_next * (1 - y^2)
        A3 = dh_next * (1 - h_next ** 2)
        
        # バイアスbの勾配
        # Nの方向に合計する
        db = np.sum(A3, axis=0)
        
        # 重みWhの勾配
        dWh = np.dot(h_prev.T, A3)
        
        # 1時刻前に渡す勾配
        dh_prev = np.dot(A3, Wh.T)
        
        # 重みWxの勾配
        dWx = np.dot(x.T, A3)
        
        # 入力xに渡す勾配
        dx = np.dot(A3, Wx.T)

        # 勾配をまとめる
        self.grads[0][:] = dWx # 同じメモリ位置に代入
        self.grads[1][:] = dWh # 同じメモリ位置に代入
        self.grads[2][:] = db # 同じメモリ位置に代入

        return dx, dh_prev
    
    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None    
        
    
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        """
        Wx : 入力xにかかる重み
        Wh : １時刻前のhにかかる重み
        b : バイアス
        stateful : 中間層の出力を次のミニバッチ に渡す場合はTrueにする
        """
        # パラメータのリスト
        self.params = [Wx, Wh, b]
        
        # 勾配のリスト
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        """
        順伝播計算
        xs : 配列形状は、(バッチサイズ、時間数、前層のノード数)
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape # バッチサイズ、時間数、前層のノード数
        D, H = Wx.shape # 入力層のノード数、中間層のノード数

        self.layers = []
        
        # hsは、中間層の出力hを時間方向につなげたもの
        hs = np.empty((N, T, H))

        # 中間層の出力hを初期化する
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H))

        # 時間方向に計算を進める
        for t in range(T):
            
            # RNNレイヤを定義する
            layer = RNN(*self.params) # *を変数前につけると、各引数に展開される
            
            # 時刻tのデータをRNNレイヤに入力する
            self.h = layer.forward(xs[:, t, :], self.h)
            
            # 中間層の出力hをhsに代入する
            hs[:, t, :] = self.h
            
            # レイヤを追加する
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        """
        逆伝播計算
        dhs : 各時刻における出力層からの勾配を格納した変数. 配列形状は(バッチ数、時間数、中間層のノード数)
        """
        
        Wx, Wh, b = self.params
        N, T, H = dhs.shape # バッチサイズ、時間数、中間層のノード数
        D, H = Wx.shape # 前層のノード数、　中間層のノード数

        # dxsを初期化する. dxsは、各時刻におけるdxを格納する変数
        dxs = np.empty((N, T, D)) # バッチ数、時間数、前層のノード数
        
        # dhの初期値
        dh = 0
        
        # 勾配の初期値
        grads = [0, 0, 0] #Wxの勾配、 Whの勾配、 bの勾配
        
        # 時間方向と逆向きに計算を進める
        for t in reversed(range(T)):
            
            # RNNレイヤの呼び出し
            layer = self.layers[t]
            
            # RNNレイヤの逆伝播計算
            # RNNレイヤに入力される勾配は、2方向から来るので、2つの値を足す
            dx, dh = layer.backward(dhs[:, t, :] + dh) 

            # dxをdxsに格納する
            dxs[:, t, :] = dx

            # Wxの勾配、 Whの勾配、 bの勾配、をそれぞれ足し合わせる
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        # Wxの勾配、 Whの勾配、 bの勾配、を保持しておく
        for i, grad in enumerate(grads):
            self.grads[i][:] = grad # 同じメモリ位置に代入
            
        # 最後の中間層のdhを保持しておく
        self.dh = dh

        return dxs

        
class Embedding:
    def __init__(self, W):
        """
        W : 重み行列, word2vecの埋め込み行列に相当する。配列形状は、(語彙数、埋め込みベクトルの要素数)
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        """
        順伝播計算
        """
        W, = self.params # dWの後の,はリストから1つだけを抜き出すためにつけている
        self.idx = idx
        
        # 埋め込み行列から埋め込みベクトルを取り出す
        out = W[idx]
        
        return out

    def backward(self, dout):
        """
        逆伝播計算
        """
        # gradsというリストの1要素目を参照する
        dW = self.grads[0]
        
        # 配列の全ての要素に0を代入する
        dW.fill(0)
        
        # dWのidxの場所にdoutを代入する
        np.add.at(dW, self.idx, dout)
        return None
    
    
class TimeEmbedding:
    def __init__(self, W):
        """
        W : 重み行列, word2vecの埋め込み行列に相当する。配列形状は、(語彙数、埋め込みベクトルの要素数)
        """        
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        """
        順伝播計算
        xs : 入力の単語ID, 配列形状は(バッチサイズ、時間数)
        """
        N, T = xs.shape # バッチサイズ、時間数
        V, D = self.W.shape # 語彙数、埋め込みベクトルの要素数

        # 初期化
        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        # 時間方向に計算を進める
        for t in range(T):
            
            # Embeddigレイヤを生成し、順伝播計算を行う
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            
            #  Embeddigレイヤを保持しておく
            self.layers.append(layer)

        return out

    def backward(self, dout):
        """
        逆伝播計算
        """
        N, T, D = dout.shape # バッチサイズ、時間数、埋め込みベクトルの要素数

        grad = 0
        
        # 時間方向に計算を進める(時間方向には独立しているので逆方向に進めなくてよい)
        for t in range(T):
            layer = self.layers[t]
            
            # 逆伝播計算
            layer.backward(dout[:, t, :])
            
            # 勾配を足し合わせる
            grad += layer.grads[0]

        self.grads[0][:] = grad # 同じメモリ位置に代入

        return None


class TimeAffine:
    def __init__(self, W, b):
        
        # パラメータのリスト
        self.params = [W, b]
        
        # 勾配のリスト
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
 
        self.x = None

    def forward(self, x):
        """
        順伝播計算
        x : 入力データ
        """
        N, T, D = x.shape # バッチサイズ、時間数、前層のノード数
        W, b = self.params

        # 全ての時刻について、一度でAffineの順伝播計算を行う
        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b # 行列の積 + バイアス
        
        # xを保持
        self.x = x
        
        return out.reshape(N, T, -1)

    def backward(self, dout):
        """
        逆伝播計算
        """
        x = self.x
        N, T, D = x.shape # バッチサイズ、時間数、前層のノード数
        W, b = self.params

        # 全ての時刻について、一度でAffineの逆伝播計算を行う
        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)
        db = np.sum(dout, axis=0) # バイアスの勾配
        dW = np.dot(rx.T, dout) # 重みWの勾配
        dx = np.dot(dout, W.T) # 前層へ伝える勾配
        dx = dx.reshape(*x.shape)

        self.grads[0][:] = dW # 同じメモリ位置に代入
        self.grads[1][:] = db # 同じメモリ位置に代入
        
        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

    
class LSTM:
    def __init__(self, Wx, Wh, b):
        '''
        Parameters
        ----------
        Wx: 入力x用の重みパラーメタ（4つ分の重みをまとめたもの)
        Wh: 隠れ状態h用の重みパラメータ（4つ分の重みをまとめたもの）
        b: バイアス（4つ分のバイアスをまとめたもの）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        """
        順伝播計算
        """        
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
    
#         print(f.shape, c_prev.shape, g.shape, i.shape)
        c_next = f * c_prev + g * i
        tanh_c_next = np.tanh(c_next)
        h_next = o * tanh_c_next

        self.cache = (x, h_prev, c_prev, i, f, g, o, tanh_c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        """
        逆伝播計算
        """        
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, tanh_c_next = self.cache

        A2 = (dh_next * o) * (1 - tanh_c_next ** 2)
        ds = dc_next + A2

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][:] = dWx # 同じメモリ位置に代入
        self.grads[1][:] = dWh # 同じメモリ位置に代入
        self.grads[2][:] = db # 同じメモリ位置に代入

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev




class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H))

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H))
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H))

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D))
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][:] = grad

        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

        
        
class GRU:
    def __init__(self, Wx, Wh, b):
        '''
        Wx: 入力x用の重みパラーメタ（3つ分の重みをまとめたもの）
        Wh: 隠れ状態h用の重みパラメータ（3つ分の重みをまとめたもの）
        b: バイアス（3つ分のバイアスをまとめたもの）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        順伝播計算
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bhz,   bhr,  bhh =  b[:H], b[H:2 * H], b[2 * H:]
        
        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bhz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + bhr)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r*h_prev, Whh) + bhh)
        h_next = z * h_prev + (1-z) * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        """
        逆伝播計算
        """        
        Wx, Wh, b = self.params
    
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat = dh_next * (1 - z)
        dh_prev = dh_next * z

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dbt = dt
        dWhh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Whh.T)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dh_prev += r * dhr

        # update gate(z)
        dz =  dh_next * h_prev - dh_next * h_hat
        dt = dz * z * (1-z)
        dbz = dt
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # reset gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dbr = dt
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        dA = np.hstack((dbz, dbr, dbt ))
        
        dWx = np.hstack((dWxz, dWxr, dWxh))
        dWh = np.hstack((dWhz, dWhr, dWhh))
        db = dA.sum(axis=0)
        
        self.grads[0][:] = dWx # 同じメモリ位置に代入
        self.grads[1][:] = dWh # 同じメモリ位置に代入
        self.grads[2][:] = db # 同じメモリ位置に代入
        
        return dx, dh_prev
    

class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]        
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, H3 = Wh.shape

        self.layers = []
        hs = np.empty((N, T, H))

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H))

        for t in range(T):
            layer = GRU(Wx, Wh, b)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D))
        dh= 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][:] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None    