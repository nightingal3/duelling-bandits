from itertools import combinations
import numpy as np
from random import uniform, choice

from gen_preference_matrix import PreferenceMatrix

class UniformSamplingPolicy:
    def __init__(self, preference_matrix: PreferenceMatrix):
        self.preference_matrix = preference_matrix
        self.timestep = 1
        self.num_actions = preference_matrix.shape[0]

    def choose_actions(self):
        first_action, second_action = choice(list(combinations(range(self.num_actions), 2)))
        reward = (
            1 if uniform(0, 1) < self.preference_matrix[first_action][second_action]
            else 0
        )
        return first_action, second_action

if __name__ == "__main__":
    pm = PreferenceMatrix(num_actions=4)
    pm.set_matrix_explicit(np.array([[0.5, 0.8, 0.6, 0.4],
    [0.2, 0.5, 0.9, 0.3],
    [0.4, 0.1, 0.5, 0.5],
    [0.6, 0.7, 0.5, 0.5]]))
    print(pm.borda_winner())
    assert False
    sampler = UniformSamplingPolicy(pm)
    actions_arr = {item: 0 for item in combinations(range(4), 2)}
    for _ in range(10000):
        actions = sampler.choose_actions()
        print(actions)
        actions_arr[actions] += 1
    print(actions_arr)


    