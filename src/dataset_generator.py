import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from typing import List
import seaborn as sns
import pandas as pd

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

def run_simulation(num_actions: int, effect_size: int = 0.3) -> None:
    pm = pref_matrix_from_effect_size(effect_size=effect_size, num_actions=num_actions)
    policy = UniformSamplingPolicy(pm)
    policy_ts = DoubleThompsonSamplingPolicy(pm)
    sample_size = estimate_sample_size(num_actions=num_actions, effect_size=effect_size)
    combs = list(combinations(range(num_actions), 2))
    print(pm)

    # convention: 1 means the victory of the arm with the lowest index
    rewards = {combination: generate_rewards_for_arm(effect_size, sample_size) for combination in combs}
    num_duels = {combination: 0 for combination in combs}
    num_duels_ts = {combination: 0 for combination in combs}
    effect_size_over_time = []
    effect_size_over_time_uni = []
    effect_size_df = [] #columns=["timestep", "sampling_type", "action1", "action2", "power"])

    for i in range(int(sample_size * 10)):
        action1, action2 = policy.choose_actions()
        ts_action1, ts_action2 = policy_ts.choose_actions()
        duel = tuple(sorted([action1, action2]))
        ts_duel = tuple(sorted([ts_action1, ts_action2]))
       
        #winner = next(rewards[duel])
        #winner_ts = next(rewards[ts_duel])

        #if policy_ts.bandits[ts_duel[0]][ts_duel[1]].wins > 0:
            #power = policy_ts.get_power(effect_size, ts_action1, ts_action2)
            #effect_size_over_time.append(power)
        all_powers = policy_ts.get_all_power(effect_size)
        for power in all_powers:
            effect_size_df.append([i, "thompson_sampling", ts_action1, ts_action2, power])
        
        all_powers_uni = policy.get_all_power(effect_size)
        for power in all_powers_uni:
            effect_size_df.append([i, "uniform", action1, action2, power])
        
    effect_size_df = pd.DataFrame(effect_size_df, columns=["timestep", "sampling_type", "action1", "action2", "power"])
    return effect_size_df

def plot_effect_size_over_time(effect_size_df) -> None:
    sns.lineplot(data=effect_size_df, x="timestep", y="power", hue="sampling_type")
    plt.savefig("testing.png")

if __name__ == "__main__":
    #sample_size = estimate_sample_size(3, effect_size=0.3)
    #print(generate_rewards_for_arm(0.3, sample_size))
    effect_size_df = run_simulation(5, effect_size=0.1)
    print(effect_size_df)
    plot_effect_size_over_time(effect_size_df)