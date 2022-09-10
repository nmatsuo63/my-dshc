import numpy as np

def numerical_gradient(f, x, W, t):
    """
    f : 目的関数
    x : 入力データ
    t : 正解データ   
    """
    h = 1e-4 # 0.0001
    grad = np.zeros_like(W)
    
    it = np.nditer(W, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index # indexを取り出す
        tmp_val = W[idx]
        
        W[idx] = tmp_val + h
        fxh1 = f(x, t)
        
        W[idx] = tmp_val - h 
        fxh2 = f(x, t)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        W[idx] = tmp_val # 値を元に戻す
        
        it.iternext()    # 次のindexへ進める
        
    return grad