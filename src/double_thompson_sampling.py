from itertools import combinations
from math import sqrt, log
import numpy as np
from random import uniform
from statsmodels.stats.power import GofChisquarePower
from typing import List

from beta_bernouilli_bandit import BetaBernouilliBandit
from thompson_sampling import ThompsonSamplingPolicy
from gen_preference_matrix import PreferenceMatrix

class DoubleThompsonSamplingPolicy:
    def __init__(self, preference_matrix: PreferenceMatrix, alpha: float = 0.001):
        self.preference_matrix = preference_matrix
        self.rewards_matrix = np.zeros(preference_matrix.shape)
        self.alpha = alpha
        self.timestep = 1
        self.rewards_over_time = []
        self.bandits = {}
        self.num_actions = preference_matrix.shape[0]
        for i in range(preference_matrix.shape[0]):
            self.bandits[i] = {
                j: BetaBernouilliBandit()
                for j in range(i, preference_matrix.shape[0])
                if j != i
            }
        self.upper_conf_bound = np.zeros((self.num_actions, self.num_actions))
        self.lower_conf_bound = np.zeros((self.num_actions, self.num_actions))
        print(self.bandits)

    def choose_actions(self):
        first_action = self.choose_first_action()
        second_action = self.choose_second_action(first_action)
        self.update_borda_reward(first_action, second_action)
        reward = (
            1 if uniform(0, 1) < self.preference_matrix[first_action][second_action]
            else 0
        )
        if first_action < second_action:
            self.bandits[first_action][second_action].update_success_or_failure(reward)
        else:
            self.bandits[second_action][first_action].update_success_or_failure(1 - reward)

        self.timestep += 1
        return first_action, second_action

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
                    history = wins/(wins + losses)
                    cb = sqrt((self.alpha * log(self.timestep))/(wins + losses))

                upper_conf_bound[i][j] = history + cb
                lower_conf_bound[i][j] = history - cb

        #print(upper_conf_bound)
        #print(lower_conf_bound)                

        copeland_ub = (1/(self.preference_matrix.shape[0] - 1)) * np.sum(upper_conf_bound, axis=1)
        candidates = np.argwhere(copeland_ub==np.amax(copeland_ub))
        #print(candidates)

        estimated_samples = np.zeros(self.rewards_matrix.shape)
        for i in range(self.rewards_matrix.shape[0]):
            for j in range(i + 1, self.num_actions):
                estimated_samples[i][j] = self.bandits[i][j].draw()
                estimated_samples[j][i] = 1 - estimated_samples[i][j]

        #print(estimated_samples)

        likely_wins = np.zeros(self.rewards_matrix.shape)
        for c in candidates:
            i = c[0]
            for j in range(self.num_actions):
                if i == j:
                    continue
                if estimated_samples[i][j] > 1/2:
                    likely_wins[i][j] = 1

        action = np.random.choice(np.argwhere(likely_wins==np.amax(likely_wins))[0]) # break ties randomly
        return action

    def choose_second_action(self, first_action: int):
        expected_samples = np.zeros(self.rewards_matrix.shape)
        for i in range(self.num_actions):
            if i == first_action:
                continue
            if i < first_action:
                expected_samples[i][first_action] = self.bandits[i][first_action].draw()
            else:
                expected_samples[i][first_action] = 1 - self.bandits[first_action][i].draw()


        uncertain_pairs = np.zeros((self.num_actions, 1))
        for i in range(self.num_actions):
            if i == first_action:
                continue
            if self.lower_conf_bound[i][first_action] <= 1/2:
                uncertain_pairs[i] = expected_samples[i][first_action]
        
        action = np.argmax(uncertain_pairs)
        if action == first_action:
            pdb.set_trace()
        return action 
    
    def update_borda_reward(self, first_action: int, second_action: int) -> None:
        total_reward = 0
        total_reward += self.preference_matrix.borda_score(first_action)
        total_reward += self.preference_matrix.borda_score(second_action)
        self.rewards_over_time.append(total_reward)

    def return_preferences_from_duel_history(self):
        predicted_pref_matrix = np.zeros(self.rewards_matrix.shape)
        for i in self.bandits:
            for j in self.bandits[i]:
                predicted_pref_matrix[i][j] = self.bandits[i][j].wins / (self.bandits[i][j].wins + self.bandits[i][j].losses)
                predicted_pref_matrix[j][i] = self.bandits[i][j].losses / (self.bandits[i][j].wins + self.bandits[i][j].losses)
        return predicted_pref_matrix

    def get_power(self, effect_size: float, action1: int, action2: int) -> float:
        if action1 < action2:
            power = GofChisquarePower().solve_power(effect_size=effect_size, nobs=self.bandits[action1][action2].wins + self.bandits[action1][action2].losses, alpha=0.05, n_bins=2)
        else:
            power = GofChisquarePower().solve_power(effect_size=effect_size, nobs=self.bandits[action2][action1].wins + self.bandits[action2][action1].losses, alpha=0.05, n_bins=2)

        return power
    
    def get_all_power(self, effect_size: float) -> List:
        powers = []
        for action1 in range(self.num_actions):
            for action2 in range(action1 + 1, self.num_actions):
                powers.append((self.get_power(effect_size, action1, action2), action1, action2))
        
        return powers
                



if __name__ == "__main__":
    pm = PreferenceMatrix(num_actions=4)
    pm.set_matrix_explicit(np.array([[0.5, 0.8, 0.6, 0.4],
    [0.2, 0.5, 0.9, 0.3],
    [0.4, 0.1, 0.5, 0.5],
    [0.6, 0.7, 0.5, 0.5]]))
    sampler = DoubleThompsonSamplingPolicy(preference_matrix=pm, alpha=1e-32)
    actions_arr = {}

    for _ in range(100000):
        actions = sampler.choose_actions()
        #print(actions)
        if actions not in actions_arr:
            actions_arr[actions] = 1
        else:
            actions_arr[actions] += 1
    print(actions_arr)
    print(sampler.return_preferences_from_duel_history())
   
   







