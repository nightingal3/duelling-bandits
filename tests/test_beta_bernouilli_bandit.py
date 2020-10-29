import pytest

from src.beta_bernouilli_bandit import BetaBernouilliBandit

class TestBetaBernoulliBandit:
    @classmethod
    def setup_class(self):
        self.bandit = BetaBernouilliBandit()

    def test_update_success_or_failure_valid(self):
        self.bandit.update_success_or_failure(0)
        assert self.bandit.wins == 0
        assert self.bandit.losses == 1
        self.bandit.update_success_or_failure(1)
        assert self.bandit.wins == 1
        assert self.bandit.losses == 1
    
    def test_update_success_or_failure_invalid(self):
        with pytest.raises(Exception):
            self.bandit.update_success_or_failure(3)
    
    def test_reset_state(self):
        self.bandit.reset_state()
        assert self.bandit.wins == 0
        assert self.bandit.losses == 0

    def test_expected_val(self):
        self.bandit.reset_state()
        self.bandit.change_wins(5)
        self.bandit.change_losses(10)
        assert self.bandit.expected_value() == 1/3
