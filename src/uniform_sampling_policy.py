from itertools import combinations
import numpy as np
from random import uniform, choice
from statsmodels.stats.power import GofChisquarePower
from typing import List

from gen_preference_matrix import PreferenceMatrix


class UniformSamplingPolicy:
    def __init__(self, preference_matrix: PreferenceMatrix):
        self.preference_matrix = preference_matrix
        self.timestep = 1
        self.rewards_over_time = []
        self.num_actions = preference_matrix.shape[0]
        self.wins = np.zeros((self.num_actions, self.num_actions))
        self.strong_regret = 0
        self.weak_regret = 0


    def choose_actions(self):
        first_action, second_action = choice(
            list(combinations(range(self.num_actions), 2))
        )
        reward = (
            1
            if uniform(0, 1) < self.preference_matrix[first_action][second_action]
            else 0
        )
        self.update_borda_reward(first_action, second_action)
        self.update_with_reward(first_action, second_action, reward)
        if not ((first_action == 0 and reward == 1) or (second_action == 0 and reward == 0)):
            # Only update the regret if the arm chosen is not the condorcet winner.
            self.strong_regret += self.get_epsilon(first_action)
            self.strong_regret += self.get_epsilon(second_action)
            self.weak_regret += min(self.get_epsilon(first_action), self.get_epsilon(second_action))

        self.timestep += 1
        return first_action, second_action

    def update_with_reward(self, first_action, second_action, reward) -> None:
        if reward:
            self.wins[first_action][second_action] += 1
        else:
            self.wins[second_action][first_action] += 1

    def update_borda_reward(self, first_action: int, second_action: int) -> None:
        total_reward = 0
        total_reward += self.preference_matrix.borda_score(first_action)
        total_reward += self.preference_matrix.borda_score(second_action)
        self.rewards_over_time.append(total_reward)

    def get_power(self, effect_size, action1, action2) -> float:
        power = GofChisquarePower().solve_power(
            effect_size=effect_size,
            nobs=self.wins[action1][action2] + self.wins[action2][action1],
            alpha=0.05,
            n_bins=2,
        )
        return power

    def get_all_power(self, effect_size: float) -> List:
        powers = []
        for action1 in range(self.num_actions):
            for action2 in range(action1 + 1, self.num_actions):
                powers.append(
                    (self.get_power(effect_size, action1, action2), action1, action2)
                )

        return powers

    def get_epsilon(self, action):
        best_arm = self.preference_matrix.condorcet_winner()
        return self.preference_matrix[best_arm][action] - 0.5


if __name__ == "__main__":
    pm = PreferenceMatrix(num_actions=4)
    pm.set_matrix_explicit(
        np.array(
            [
                [0.5, 0.8, 0.6, 0.4],
                [0.2, 0.5, 0.9, 0.3],
                [0.4, 0.1, 0.5, 0.5],
                [0.6, 0.7, 0.5, 0.5],
            ]
        )
    )
    print(pm.borda_winner())
    assert False
    sampler = UniformSamplingPolicy(pm)
    actions_arr = {item: 0 for item in combinations(range(4), 2)}
    for _ in range(10000):
        actions = sampler.choose_actions()
        print(actions)
        actions_arr[actions] += 1
    print(actions_arr)
