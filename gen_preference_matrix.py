from dataclasses import dataclass, field
#from itertools import product, combinations
import numpy as np

def init_preference_matrix(num_actions: int, num_criteria: int) -> np.ndarray:
    return np.zeros((num_actions, num_actions))

@dataclass
class PreferenceMatrix:
    num_actions: int
    num_criteria: int = 1

    def __post_init__(self):
        self.data = init_preference_matrix(self.num_actions, self.num_criteria)
        self.curr_condorcet_winner = None

    def zero_matrix(self) -> None:
        self.data - np.zeros((self.num_actions, self.num_actions))

    def condorcet_winner(self) -> int:
        if self.curr_condorcet_winner == None:
            highest = 0
            winner = None
            for i in range(self.num_actions):
                if sum(self.data[i, :]) > 0.5 * self.num_actions and sum(self.data[i, :]) > highest:
                    highest = sum(self.data[i, :])
                    winner = i

            if winner:
                self.curr_condorcet_winner = winner
                return winner
            return -1
                    
        else:
            return self.curr_condorcet_winner



if __name__ == "__main__":
    matrix = PreferenceMatrix(num_actions=2, num_criteria=1)
    print(matrix.condorcet_winner())

