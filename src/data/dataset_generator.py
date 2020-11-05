from itertools import combinations
import numpy as np
import random
from statsmodels.stats.power import TTestIndPower
from typing import List

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

    rewards = [1 if random.uniform(0, 1) < threshold else 0 for _ in range(round(sample_size))]
    return rewards

if __name__ == "__main__":
    sample_size = estimate_sample_size(3, effect_size=0.3)
    print(generate_rewards_for_arm(0.3, sample_size))