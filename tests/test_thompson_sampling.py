import pytest

from src.thompson_sampling import ThompsonSamplingPolicy

class TestThompsonSamplingPolicy:
    @classmethod
    def setup_class(self):
        self.uniform_sampler = ThompsonSamplingPolicy(means_list=[0.5, 0.5, 0.5])
        self.skewed_sampler = ThompsonSamplingPolicy(means_list=[0.2, 0.3, 0.5])
        self.best_option_sampler = ThompsonSamplingPolicy(means_list=[0.5, 0.3, 1])

    def test_update_bandit(self):
        self.uniform_sampler.update_wins_or_losses()