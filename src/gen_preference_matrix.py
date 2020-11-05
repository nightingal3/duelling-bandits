from dataclasses import dataclass
import numpy as np
from typing import List

def init_preference_matrix(num_actions: int, num_criteria: int) -> np.ndarray:
    return np.zeros((num_actions, num_actions))

@dataclass
class PreferenceMatrix:
    num_actions: int
    num_criteria: int = 1

    def __post_init__(self):
        self.data = init_preference_matrix(self.num_actions, self.num_criteria)
        self.curr_condorcet_winner = None
        self.num_observations = np.array([0] * self.num_actions)
        self.shape = (self.num_actions, self.num_actions)
    
    def __getitem__(self, ind):
        return self.data[ind]

    def reset_state(self) -> None:
        self.data = np.zeros((self.num_actions, self.num_actions))
        self.curr_condorcet_winner = None
        self.num_observations = np.array([0] * self.num_actions)

    def set_matrix_explicit(self, matrix) -> None:
        if not isinstance(matrix, np.ndarray) or matrix.shape != (self.num_actions, self.num_actions):
            raise Exception("Matrix must be an ndarray with shape (num_actions, num_actions)")
        self.data = matrix
        self.curr_condorcet_winner = None
        self.num_observations = np.array([np.sum(self.data[i, :]) + np.sum(self.data[i, :]) - self.data[i, i] for i in range(self.num_actions)])

    def record_win(self, winner: int, loser: int) -> None:
        if winner != self.curr_condorcet_winner: 
            self.curr_condorcet_winner = None
        self.data[winner, loser] += 1
        self.num_observations[winner] += 1
        self.num_observations[loser] += 1

    def condorcet_winner(self) -> int:
        if self.curr_condorcet_winner == None:
            for i in range(self.num_actions):
                for j in range(self.num_actions):
                    if i == j:
                        continue
                    if self.data[j, i] >= self.data[i, j]:
                        break
                    if j == self.num_actions - 1:
                        self.curr_condorcet_winner = i
                        return i
            return -1

        else:
            return self.curr_condorcet_winner
    
    def borda_winner(self) -> np.ndarray:
        return np.argwhere(np.sum(self.data, axis=1)==np.amax(np.sum(self.data, axis=1)))

if __name__ == "__main__":
    matrix = PreferenceMatrix(num_actions=2, num_criteria=1)
    print(matrix.num_observations)

