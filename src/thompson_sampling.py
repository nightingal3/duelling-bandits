import numpy as np
from random import uniform
from typing import List

from beta_bernouilli_bandit import BetaBernouilliBandit

class ThompsonSamplingPolicy:
    def __init__(self, means_list: List):
        self.bandits = []
        for _ in range(len(means_list)):
            self.bandits.append(BetaBernouilliBandit())
        self.means_list = means_list
        self.rewards_list = []

    def choose_action(self):
        samples = [bandit.draw() for bandit in self.bandits]
        chosen = np.argmax(samples)
        reward = 1 if uniform(0, 1) < self.means_list[chosen] else 0
        self.update_bandit(chosen, reward)
        self.rewards_list.append(reward)
        print(chosen)

    def update_bandit(self, bandit_ind, reward):
        self.bandits[bandit_ind].update_success_or_failure(reward)
    
    def reset_state(self):
        self.bandist = []
        for _ in range(len(self.means_list)):
            self.bandits.append(BetaBernouilliBandit())
        self.rewards_list = []


if __name__== "__main__":
    sampler = ThompsonSamplingPolicy(means_list=[0.25, 0.3, 0.5, 0.35])
    for _ in range(0, 1000):
        sampler.choose_action()
