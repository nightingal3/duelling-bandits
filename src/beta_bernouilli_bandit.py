import numpy as np
import random

class BetaBernouilliBandit:
    def __init__(self, wins: int = 0, losses: int = 0):
        self.wins = wins
        self.losses = losses

    def __str__(self):
        return f"Beta Bernouilli Bandit:\nnum wins: {self.wins}\nnum losses: {self.losses}"
    
    def __repr__(self):
        return f"<BetaBernouilliBandit wins:{self.wins} losses:{self.losses}>"

    def update_success_or_failure(self, reward: int) -> None:
        if reward not in [0, 1]:
            raise ValueError("Only binary (0/1) success values accepted. To overwrite success/failure count, use change_wins/change_losses.")
        if reward:
            self.wins += 1
        else:
            self.losses += 1

    def change_wins(self, new_wins: int) -> None:
        self.wins = new_wins
    
    def change_losses(self, new_losses: int) -> None:
        self.losses = new_losses

    def reset_state(self) -> None:
        self.wins = 0
        self.losses = 0

    def expected_value(self) -> float:
        return self.wins + 1 / (self.wins + self.losses + 2)

    def draw(self) -> float:
        return np.random.beta(self.wins + 1, self.losses + 1)