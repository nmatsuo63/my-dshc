import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict


class State():
    """
    状態のクラス
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def clone(self):
        """
        自分自身を複製するためのメソッド
        """
        return State(self.row, self.col)
    
    def __hash__(self):
        """
        ハッシュメソッド
        dict のようなハッシュを使ったコレクション型の要素に対する操作から呼び出される
        __eq__メソッドとセットで用いる
        """
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        """
        ハッシュとセットで用いる
        """
        return self.row == other.row and self.col == other.col
    
    def __repr__(self):
        """
        オブジェクトを表す文字列を返す
        """
        return "<State: [{}, {}]>".format(self.row, self.col)

    
class Action():
    """
    行動のクラス
    """
    dic_actions = OrderedDict()
    dic_actions[0] = "UP"
    dic_actions[1] = "DOWN"
    dic_actions[2] = "LEFT"
    dic_actions[3] = "RIGHT"
    
    
class Environment():
    """
    環境のクラス
    """
    def __init__(self, grid, move_prob=0.8, agent_init_state=(0,0)):
        """
        grid : 2d-array, 迷路の各セルの条件
            0: 通常のセル
            -1: 落とし穴 (ゲーム終了)
            1: ゴール (ゲーム終了)
            9: 壁 (移動不可)
        move_prob : float, 移動確率
        """
        # 迷路条件を設定する
        self.grid = grid
        self.row_length = grid.shape[0]
        self.col_length = grid.shape[1]
        
        # 行動を定義する
        self.Action = Action()
        self.actions = list(self.Action.dic_actions.keys())
        
        # 状態を定義する
        self.states = self.get_all_states()
        
        # エージェントの位置を初期化する
        self.reset(agent_init_state)
            
        # 報酬の初期値(通常セルに割り当てられる)
        # 通常セルの報酬をマイナス値に設定しておくと、エージェントが早くゴールに到達するようになる
        self.default_reward = -0.04

        # エージェントが選択した方向に移動する確率
        # エージェントは、(1- move_prob)の確率で、選択した方向とは異なる方向に移動する
        # これは、実際の環境において制御通りに移動できないことを想定している
        self.move_prob = move_prob

    def get_all_states(self):
        """
        全ての状態を定義するためのメソッド
        """
        states = []
        for row in range(self.row_length):
            for col in range(self.col_length):
                # ブロックを配置した場所は状態を定義する必要がないのでスキップする
                if self.grid[row][col] != 9:
                    states.append(State(row, col))
        return states

    def reset(self, agent_init_state=(0,0)):
        """
        エージェントの位置を初期化するメソッド
        """
        # エージェントの初期位置を設定する
        self.agent_state = State(*agent_init_state)
        return self.agent_state    
    
    def check_next_state(self, state, action):
        """
        次の状態を確認するためのメソッド
        """

        # 現状の状態オブジェクトを複製する
        next_state = state.clone()

        # エージェントを移動させる
        if action == 0: #"UP"
            next_state.row -= 1
        elif action == 1: #"DOWN"
            next_state.row += 1
        elif action == 2: #"LEFT"
            next_state.col -= 1
        elif action == 3: #"RIGHT"
            next_state.col += 1

        # 更新後の状態が迷路からはみ出してしまう場合は、更新しないことにする        
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.col < self.col_length):
            next_state = state

        # 更新後の状態がブロックセルに入ってしまう場合は、更新しないことにする
        if self.grid[next_state.row][next_state.col] == 9:
            next_state = state

        return next_state

    def calc_transit_prob(self, state, action):
        """
        遷移確率を計算するメソッド
        """
        dic_transition_probs = {}
        if not self.can_action_at(state):
            """
            ゴールもしくは、穴にいる場合
            """
            return dic_transition_probs

        for a in self.actions:
            if a == action:
                """
                選ばれた方向の場合
                """
                prob = self.move_prob
            else:
                """
                選ばれた方向と異なる方向
                """
                # 選ばれた方向以外に進む確率を3等分する
                prob = (1 - self.move_prob) / 3
            
            # エージェントの稼働範囲に応じて移動確率を調整する
            next_state = self.check_next_state(state, a) # あるaが移動できない行動である場合は、現状のstateがそのまま返ってくる
            if next_state not in dic_transition_probs:
                dic_transition_probs[next_state] = prob
            else:
                dic_transition_probs[next_state] += prob

        return dic_transition_probs

    def can_action_at(self, state):
        """
        あるセル(state)が行動可能なセルかどうかを確認するメソッド
        """
        if self.grid[state.row][state.col] == 0:
            return True
        else:
            return False


    def reward_func(self, state):
        """
        報酬関数
        """
        reward = self.default_reward # 報酬の初期値
        done = False

        # 今の状態の設定を確認する
        attribute = self.grid[state.row][state.col]
        
        # 今の状態の設定に応じて、報酬を与える
        if attribute == 1:
            """
            ゴールに到達した場合
            """
            reward = 1 # 報酬を与える
            done = True # 終了フラグ
        elif attribute == -1:
            """
            穴に落ちた場合
            """
            reward = -1 # 負の報酬を与える
            done = True # 終了フラグ

        return reward, done

    def transit(self, state, action):
        """
        遷移関数
        次の状態(移動先)を決定し、報酬を算出する
        """
        
        # 遷移確率を計算する
        dic_transition_probs = self.calc_transit_prob(state, action)
        if len(dic_transition_probs) == 0:
            """
            ゴールもしくは穴にいる場合
            """
            return None, None, True

        # dic_transition_probsをリストに変換する
        next_states = []
        probs = []
        for s, prob in dic_transition_probs.items():
            next_states.append(s)
            probs.append(prob)

        # 次の状態(移動先)を確率に基づいて決定する
        next_state = np.random.choice(next_states, p=probs)
        
        # 報酬を算出する
        reward, done = self.reward_func(next_state)
        
        return next_state, reward, done
    
    def step(self, action):
        """
        ある行動を実際にとってみたときの、次の状態と獲得報酬額を決めるための関数
        """
        # 今の状態からある行動をとってみると、次の状態と獲得報酬額がわかる
        next_state, reward, done = self.transit(self.agent_state, action)
        
        # 状態を更新する
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done
