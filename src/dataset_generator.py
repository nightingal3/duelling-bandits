from itertools import combinations
import numpy as np
import random
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from typing import List

from uniform_sampling_policy import UniformSamplingPolicy
from double_thompson_sampling import DoubleThompsonSamplingPolicy
from gen_preference_matrix import PreferenceMatrix

import pdb

def estimate_sample_size(num_actions: int, effect_size: float = 0.1, effect_size_threshold: float = 0.8, significance_threshold: float = 0.05) -> float:
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(effect_size=effect_size, power=effect_size_threshold, alpha=significance_threshold)
    total_sample_size = sample_size * len(list(combinations(range(num_actions), 2))) # assume same effect size between arms for now
    return total_sample_size

def generate_rewards_for_arm(effect_size: float, sample_size: int) -> List:
    if effect_size == 0.1:
        threshold = 0.55
    elif effect_size == 0.3:
        threshold = 0.65
    elif effect_size == 0.5:
        threshold = 0.75

    rewards = [1 if random.uniform(0, 1) < threshold else 0 for _ in range(round(sample_size * 100))]
    rewards_gen = (x for x in rewards)
    return rewards_gen

def pref_matrix_from_effect_size(effect_size: float, num_actions: int) -> PreferenceMatrix:
    data = np.zeros((num_actions, num_actions))
    for i in range(num_actions):
        for j in range(i):
            if effect_size == 0.1:
                threshold = 0.55
            elif effect_size == 0.3:
                threshold = 0.65
            elif effect_size == 0.5:
                threshold = 0.75
            data[i][j] = threshold
            data[j][i] = 1 - threshold 
    
    pm = PreferenceMatrix(num_actions=num_actions, data=data)
    return pm

def run_simulation_uniform(num_actions: int, effect_size: int = 0.3) -> None:
    pm = pref_matrix_from_effect_size(effect_size=effect_size, num_actions=3)
    policy = UniformSamplingPolicy(pm)
    policy_ts = DoubleThompsonSamplingPolicy(pm)
    sample_size = estimate_sample_size(num_actions=num_actions, effect_size=effect_size)
    combs = list(combinations(range(num_actions), 2))
    print(pm)

    # convention: 1 means the victory of the arm with the lowest index
    rewards = {combination: generate_rewards_for_arm(effect_size, sample_size) for combination in combs}
    num_duels = {combination: 0 for combination in combs}
    num_duels_ts = {combination: 0 for combination in combs}

    for i in range(int(sample_size * 100)):
        action1, action2 = policy.choose_actions()
        ts_action1, ts_action2 = policy_ts.choose_actions()
        duel = tuple(sorted([action1, action2]))
        ts_duel = tuple(sorted([ts_action1, ts_action2]))
        num_duels[duel] += 1
        num_duels_ts[ts_duel] += 1
        winner = next(rewards[duel])
        
    print(num_duels)
    print(num_duels_ts)
    print(policy_ts.bandits)
    print(tt_ind_solve_power(effect_size=0.3, nobs1=policy_ts.bandits[0][1].wins, alpha=0.05, ratio=policy_ts.bandits[0][1].losses/policy_ts.bandits[0][1].wins))
        #print(winner)





if __name__ == "__main__":
    #sample_size = estimate_sample_size(3, effect_size=0.3)
    #print(generate_rewards_for_arm(0.3, sample_size))
    run_simulation_uniform(3, effect_size=0.5)