from math import sqrt, log
import numpy as np
from random import uniform

from beta_bernouilli_bandit import BetaBernouilliBandit
from thompson_sampling import ThompsonSamplingPolicy
from gen_preference_matrix import PreferenceMatrix

import pdb

class DoubleThompsonSamplingPolicy:
    def __init__(self, preference_matrix: PreferenceMatrix, alpha: float = 0.001):
        self.preference_matrix = preference_matrix
        self.rewards_matrix = np.zeros(preference_matrix.shape)
        self.alpha = alpha
        self.timestep = 1
        self.bandits = {}
        self.num_actions = preference_matrix.shape[0]
        for i in range(preference_matrix.shape[0]):
            self.bandits[i] = {
                j: BetaBernouilliBandit()
                for j in range(i, preference_matrix.shape[0])
                if j != i
            }
        print(self.bandits)

    def choose_actions(self):
        first_action = self.choose_first_action()
        second_action = self.choose_second_action(first_action)
        reward = (
            1 if uniform(0, 1) < self.preference_matrix[first_action][second_action]
            else 0
        )
        self.bandits[first_action][second_action].update_success_or_failure(reward)
        self.bandits[second_action][first_action].update_success_or_failure(1 - reward)
        self.timestep += 1

    def choose_first_action(self):
        upper_conf_bound = np.zeros((self.num_actions, self.num_actions))
        lower_conf_bound = np.zeros((self.num_actions, self.num_actions))
        for i in range(self.num_actions):
            for j in range(self.num_actions):
                if j == i:
                    continue
                if j < i:
                    wins = self.bandits[j][i].losses
                    losses = self.bandits[j][i].wins
                else:
                    wins = self.bandits[i][j].wins
                    losses = self.bandits[i][j].losses
    
                if wins + losses == 0:
                    history = 1
                    cb = 1
                else:
                    history[i][j] = wins/(wins + losses)
                    cb = sqrt((self.alpha * log(self.timestep))/(wins + losses))

                upper_conf_bound[i][j] = history + cb
                lower_conf_bound[i][j] = history - cb

        print(upper_conf_bound)
        print(lower_conf_bound)                

        copeland_ub = (1/(self.preference_matrix.shape[0] - 1)) * np.sum(upper_conf_bound, axis=1)
        candidates = np.argwhere(copeland_ub==np.amax(copeland_ub))
        print(candidates)

        estimated_samples = np.zeros(self.rewards_matrix.shape)
        for i in range(self.rewards_matrix.shape[0]):
            for j in range(i + 1, self.num_actions):
                estimated_samples[i][j] = self.bandits[i][j].draw()
                estimated_samples[j][i] = 1 - estimated_samples[i][j]

        print(estimated_samples)

        likely_wins = np.zeros(self.rewards_matrix.shape)
        for c in candidates:
            i = c[0]
            for j in range(self.num_actions):
                if i == j:
                    continue
                if estimated_samples[i][j] > 1/2:
                    likely_wins[i][j] = 1

        action = np.random.choice(np.argwhere(likely_wins==np.amax(likely_wins))[0]) # break ties randomly
        print(action)
        return action

    def choose_second_action(self, first_action: int):
        pass


if __name__ == "__main__":
    pm = PreferenceMatrix(num_actions=4)
    pm.set_matrix_explicit(np.array([[0.5, 0.8, 0.6, 0.4],
    [0.2, 0.5, 0.9, 0.3],
    [0.4, 0.1, 0.5, 0.5],
    [0.6, 0.7, 0.5, 0.5]]))
    sampler = DoubleThompsonSamplingPolicy(preference_matrix=pm)
    sampler.choose_first_action()
   







