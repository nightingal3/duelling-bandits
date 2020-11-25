import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from statsmodels.stats.power import GofChisquarePower
from typing import List
import seaborn as sns
import pandas as pd

from uniform_sampling_policy import UniformSamplingPolicy
from double_thompson_sampling import DoubleThompsonSamplingPolicy
from gen_preference_matrix import PreferenceMatrix

import pdb

def estimate_sample_size(num_actions: int, effect_size: float = 0.1, effect_size_threshold: float = 0.8, significance_threshold: float = 0.05) -> float:
    power_analysis = GofChisquarePower() #is this the right power analysis?
    sample_size = power_analysis.solve_power(effect_size=effect_size, power=effect_size_threshold, alpha=significance_threshold, n_bins=2)
    total_sample_size = sample_size * len(list(combinations(range(num_actions), 2))) # assume same effect size between arms for now
    return total_sample_size

def generate_rewards_for_arm(effect_size: float, sample_size: int, sample_size_multiple: float = 1) -> List:
    if effect_size == 0.1:
        threshold = 0.55
    elif effect_size == 0.3:
        threshold = 0.65
    elif effect_size == 0.5:
        threshold = 0.75

    rewards = [1 if random.uniform(0, 1) < threshold else 0 for _ in range(round(sample_size * sample_size_multiple))]
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

def condorcet_winner_gen(effect_size: float) -> None:
    pass

def run_simulation(num_actions: int, effect_size: int = 0.3, sample_size_multiple: float = 1) -> None:
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

    for i in range(int(sample_size * sample_size_multiple)):
        action1, action2 = policy.choose_actions()
        ts_action1, ts_action2 = policy_ts.choose_actions()
        duel = tuple(sorted([action1, action2]))
        ts_duel = tuple(sorted([ts_action1, ts_action2]))

        all_powers = policy_ts.get_all_power(effect_size)
        for power, p_action1, p_action2 in all_powers:
            effect_size_df.append([i + 1, "thompson_sampling", p_action1, p_action2, power, policy_ts.rewards_over_time[-1]])
        
        all_powers_uni = policy.get_all_power(effect_size)
        for power, p_action1_uni, p_action2_uni in all_powers_uni:
            effect_size_df.append([i + 1, "uniform", p_action1_uni, p_action2_uni, power, policy.rewards_over_time[-1]])
        
    effect_size_df = pd.DataFrame(effect_size_df, columns=["timestep",  "sampling_type", "action1", "action2", "power", "borda_reward"])

    return effect_size_df

def run_multi_simulations(num_experiments: int, sim_output_file: str, num_actions: int, effect_size: int = 0.3, sample_size_multiple: float = 1) -> None:
    for i in num_experiments:
        with open(sim_output_file, "a") as out_f:
            out_f.write(f"SIMULATION NUMBER {i}: ")
            for comb in range(combinations):
                print(f"Combination {comb}: Found effect? {combinations[comb]["found_effect"]}\tTest stat: {combinations[comb]["t_stat"]}\tp-val: {combinations[comb]["p-val"]}")

def plot_effect_size_over_time(effect_size_df, filename: str = "testing.png") -> None:
    sns.lineplot(data=effect_size_df, x="timestep", y="power", hue="sampling_type")
    plt.savefig(filename)

def plot_effect_size_bars(filename: str = "testing.png", **effect_size_dfs) -> None:
    final_effect_size = pd.DataFrame()
    for num_timesteps, df in effect_size_dfs.items():
        final_effect_size_ts = df.loc[(df["timestep"] == int(num_timesteps) - 1) & (df["sampling_type"] == "thompson_sampling")]
        final_effect_size_uni = df.loc[(df["timestep"] == int(num_timesteps) - 1) & (df["sampling_type"] == "uniform")]
        
        final_effect_size = final_effect_size.append(final_effect_size_ts)
        final_effect_size = final_effect_size.append(final_effect_size_uni)
    print(final_effect_size)
    sns.barplot(x="timestep", y="power", hue="sampling_type", data=final_effect_size)
    plt.savefig(filename)

def plot_borda_reward_bars( num_duels: int, filename: str = "testing.png", **dfs) -> None:
    sum_borda_score = []
    for num_timesteps, df in dfs.items():
        borda_score_ts = (df.loc[df["sampling_type"] == "thompson_sampling"]["borda_reward"].sum())/(int(num_timesteps) * num_duels)
        borda_score_uni = df.loc[df["sampling_type"] == "uniform"]["borda_reward"].sum()/(int(num_timesteps) * num_duels)

        sum_borda_score.append([int(num_timesteps), "thompson_sampling", borda_score_ts])
        sum_borda_score.append([int(num_timesteps), "uniform", borda_score_uni])

    sum_borda_score = pd.DataFrame(sum_borda_score, columns=["timestep", "sampling_type", "average reward"])
    sns.barplot(x="timestep", y="average reward", hue="sampling_type", data=sum_borda_score)
    plt.savefig(filename)

def plot_borda_reward_over_time(reward_df: pd.DataFrame, filename: str = "testing.png") -> None:
    sns.lineplot(data=reward_df, x="timestep", y="borda_reward", hue="sampling_type")
    plt.savefig(filename)

if __name__ == "__main__":
    EFFECT_SIZE=0.1
    NUM_ARMS=3
    num_duels = len(list(combinations(range(NUM_ARMS), 2)))

    sample_size = estimate_sample_size(NUM_ARMS, effect_size=EFFECT_SIZE)
    #print(generate_rewards_for_arm(0.3, sample_size))
    effect_size_df_1 = run_simulation(NUM_ARMS, effect_size=EFFECT_SIZE, sample_size_multiple=1)
    effect_size_df_05 = run_simulation(NUM_ARMS, effect_size=EFFECT_SIZE, sample_size_multiple=0.5)
    effect_size_df_2 = run_simulation(NUM_ARMS, effect_size=EFFECT_SIZE, sample_size_multiple=2)
    effect_size_df_4 = run_simulation(NUM_ARMS, effect_size=EFFECT_SIZE, sample_size_multiple=4)

    #plot_effect_size_over_time(effect_size_df)
    #plot_effect_size_bars(f"effect_size_{EFFECT_SIZE}_{NUM_ARMS}_arms.png", **{str(int(0.5 * sample_size)): effect_size_df_05, str(int(sample_size)): effect_size_df_1, str(int(2 * sample_size)): effect_size_df_2, str(int(4 * sample_size)): effect_size_df_4})
    plot_borda_reward_bars(num_duels, f"borda_effect_size_{EFFECT_SIZE}_{NUM_ARMS}_arms.png", **{str(int(0.5 * sample_size)): effect_size_df_05, str(int(sample_size)): effect_size_df_1, str(int(2 * sample_size)): effect_size_df_2, str(int(4 * sample_size)): effect_size_df_4})
    plot_borda_reward_over_time(effect_size_df_4, f"borda_over_time_{EFFECT_SIZE}_{NUM_ARMS}_arms.png")