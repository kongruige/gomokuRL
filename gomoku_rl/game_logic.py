# gomoku-rl/gomoku_rl/game_logic.py

import numpy as np
import copy

class GomokuGame:
    """
    This class handles the logic for the game of Gomoku.
    """
    def __init__(self, board_size=13, win_condition=5):
        self.board_size = board_size
        self.win_condition = win_condition
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = None

    # --- THIS IS THE CORRECT PLACEMENT FOR CLONE ---
    def clone(self):
        """Creates a deep copy of the game state."""
        return copy.deepcopy(self)

    def get_board_state(self):
        """
        Returns the board state as a multi-channel numpy array for the neural network.
        Channel 1: Current player's stones
        Channel 2: Opponent's stones
        Channel 3: Indicator of whose turn it is
        """
        player = self.current_player
        opponent = 2 if player == 1 else 1

        player_stones = np.where(self.board == player, 1, 0)
        opponent_stones = np.where(self.board == opponent, 1, 0)
        turn_indicator = np.full((self.board_size, self.board_size), player - 1) # 0 for P1, 1 for P2

        return np.stack([player_stones, opponent_stones, turn_indicator])

    def make_move(self, move):
        """Places a stone on the board and checks for a win."""
        row, col = move
        if self.board[row, col] != 0 or self.game_over:
            return False  # Invalid move

        self.board[row, col] = self.current_player
        
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif not self.get_valid_moves():
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = 2 if self.current_player == 1 else 1
            
        return True

    def get_valid_moves(self):
        """Returns a list of (row, col) tuples for all empty squares."""
        return list(zip(*np.where(self.board == 0)))

    def _check_win(self, row, col):
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, self.win_condition):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, self.win_condition):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= self.win_condition:
                return True
        return False

    def display(self):
        print("   " + " ".join([f"{i:2}" for i in range(self.board_size)]))
        print("  " + "-" * (self.board_size * 3))
        for i, row in enumerate(self.board):
            row_str = " ".join(['.', 'X', 'O'][val] for val in row)
            print(f"{i:2}| {row_str}")
        print()


# --- Main loop to play the game ---
if __name__ == "__main__":
    game = GomokuGame(board_size=13, win_condition=5)

    while not game.game_over:
        game.display()
        print(f"Player {game.current_player}'s turn ('X' is 1, 'O' is 2)")

        try:
            move_str = input("Enter your move (row,col): ")
            row, col = map(int, move_str.split(','))

            if (row, col) not in game.get_valid_moves():
                print("Invalid move: Spot is either taken or out of bounds. Try again.")
                continue

            game.make_move(row, col)

        except ValueError:
            print("Invalid format. Please use 'row,col'.")
        except IndexError:
            print("Invalid move: Coordinates out of bounds. Try again.")

    # Print final result
    game.display()
    if game.winner == 0:
        print("The game is a draw!")
    else:
        print(f"Congratulations, Player {game.winner} ('{'X' if game.winner == 1 else 'O'}') wins!")