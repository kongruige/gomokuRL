# gomoku-rl/gomoku_rl/agent.py

import torch
import numpy as np

from .mcts import MCTS
from . import config

class Agent:
    """
    The AI Agent that uses MCTS to decide on the best move.
    """
    def __init__(self, neural_net, device):
        self.net = neural_net
        self.device = device

    def choose_move(self, game):
        """
        Runs MCTS simulations to determine the best move.
        In a real game, we choose the move with the highest visit count (most explored).
        """
        mcts = MCTS(game.clone(), self.net, self.device)
        move_probs = mcts.search(num_simulations=config.MCTS_SIMULATIONS)
        
        # In a competitive setting, choose the best move greedily
        best_move_index = np.argmax(move_probs)
        best_move = (best_move_index // config.BOARD_SIZE, best_move_index % config.BOARD_SIZE)
        
        return best_move