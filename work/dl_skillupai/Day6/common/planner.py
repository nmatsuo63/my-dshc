import numpy as np

class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        """
        初期化メソッド
        """
        # 環境の初期化
        self.env.reset()

    def plan(self, gamma=0.9, threshold=0.0001):
        """
        planメソッドの実態は、このクラスを継承するクラスに記述する
        """
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        """
        行動確率, 次の状態, 報酬を順番に返すメソッド
        """
        dic_transition_probs = self.env.calc_transit_prob(state, action)
        for next_state, prob in dic_transition_probs.items():
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward # 順番にreturnする(イテレータ)

    def dict_to_array(self, state_reward_dict):
        """
        dict形式を2次元配列形式に変換するメソッド
        """
        grid = np.zeros((self.env.row_length, self.env.col_length))
        for s in state_reward_dict:
            grid[s.row, s.col] = state_reward_dict[s]

        return grid