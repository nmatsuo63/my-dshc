import numpy as np
import matplotlib.pyplot as plt

class CommonAgent():
    """
    共通のエージェントクラス
    """

    def __init__(self, epsilon):
        self.Q = {}
        self.epsilon = epsilon
        self.rewards_log = []

    def init_log(self):
        """
        ログの初期化
        """
        self.rewards_log = []

    def log(self, reward):
        """
        ログの記録
        """
        self.rewards_log.append(reward)

    def show_rewards_log(self, interval=50, episode=-1):
        """
        エピソード終了直前の報酬の可視化
        """
        if episode > 0:
            rewards = self.rewards_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))
        else:
            indices = list(range(0, len(self.rewards_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.rewards_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure(figsize=(6,3))
            plt.title("Averaged final reward")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",label="Averaged final reward for each {} episode".format(interval))
            plt.legend(bbox_to_anchor=(0.75,-0.2))
            plt.ylim([-1.5,1.5])
            plt.xlabel("episode")
            plt.show()