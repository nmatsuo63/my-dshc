# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        print("Trainerクラスのインスタンスが無事生成されました")
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        print(f'optimizer:{self.optimizer}')
        
        # 学習に利用するデータ数
        self.train_size = x_train.shape[0]
        # 1エポックにおけるミニバッチの数 （= 1000 / 100 = 10）
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        # 最後のエポックにおける終了時のiteration数　（= 20 * 1000 / 100 = 200）
        self.max_iter = int(epochs * self.iter_per_epoch)
        # 現在のiteration数
        self.current_iter = 0
        # 現在のエポック数
        self.current_epoch = 0
        
        self.train_loss_list = []; self.test_loss_list = []
        self.train_acc_list = []; self.test_acc_list = []
        print(f'総iter数：{self.max_iter} = エポック数：{self.epochs}, ミニバッチ数：{self.iter_per_epoch}')

    def train_step(self):
        # ミニバッチ作成（randomを利用）
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        # print("trainer.pyのデバッグ")
        # print("x_batch.shape", x_batch.shape)
        # print("t_batch.shape", t_batch.shape)

        # 勾配計算
        grads = self.network.gradient(x_batch, t_batch)
        # print("勾配計算完了")
        # 更新
        self.optimizer.update(self.network.params, grads)
        # print("更新計算完了")
        # 損失関数を計算
        loss = self.network.loss(x_batch, t_batch)
        # print("損失関数計算完了")
        # # 損失量をリストに追加
        # self.train_loss_list.append(loss)

        # verbose（詳細）を出力するか否か
        # self.verbose=False
        if self.verbose: 
            print(f'進捗：{self.current_iter / self.max_iter:.1%}, 訓練データの誤差：{loss:.3}')
        
        # エポックに分割
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            # 学習データと検証データを設定
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            
            # エポックごとのサンプル数を評価する
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            # 訓練データの精度を計算する
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            # print("train_acc　計算完了")
            train_loss = self.network.loss(x_train_sample, t_train_sample)
            # print("train_loss　計算完了")
            
            # 検証データの精度を計算する
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            # print("test_acc　計算完了")
            test_loss = self.network.loss(x_test_sample, t_test_sample)
            # print("test_loss　計算完了")
            
            # 精度と誤差をリストに記録する
            self.train_acc_list.append(train_acc); self.test_acc_list.append(test_acc)
            self.train_loss_list.append(train_loss); self.test_loss_list.append(test_loss)
            
            # verbose（詳細）を出力するか否か
            if self.verbose: 
                print(f'==epoch:{self.current_epoch}, train_acc:{train_acc:.3}, test_acc:{test_acc:.3}==')
        self.current_iter += 1

    def train(self):
        # 最後のエポックまでの総iter回tra_step()を実行する
        for i in range(self.max_iter):
            # print(f'train{i}回目（全{self.max_iter}回）')
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

