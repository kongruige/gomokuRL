# gomoku-rl/gomoku_rl/mcts.py

import numpy as np
import math
import torch

from . import config

class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    """
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # A map from action to Node
        self.visit_count = 0
        self.total_action_value = 0.0 # Q-value
        self.prior_probability = prior_p # P-value

    def select_child(self):
        """
        Selects the child with the highest Upper Confidence Bound (UCB) score.
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def expand(self, action_probs):
        """
        Expands the node by creating new children for all possible actions.
        action_probs: A list of (action, probability) tuples from the neural network.
        """
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_p=prob)

    def update(self, value):
        """
        Updates the node's statistics from a simulation result.
        """
        self.visit_count += 1
        self.total_action_value += value

    def get_ucb_score(self):
        """
        Calculates the UCB score for this node.
        """
        # Q-value: Average action value
        q_value = self.total_action_value / self.visit_count if self.visit_count > 0 else 0
        
        # U-value: Exploration term
        exploration_term = (config.CPUCT * self.prior_probability *
                            math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        
        return q_value + exploration_term
    
    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, game, neural_net, device):
        self.game = game
        self.net = neural_net
        self.device = device
        self.root = Node()

    def search(self, num_simulations):
        """
        Runs the MCTS algorithm for a specified number of simulations.
        """
        for _ in range(num_simulations):
            # 1. Selection
            node = self.root
            temp_game = self.game.clone() # Create a copy for simulation

            while not node.is_leaf():
                action, node = node.select_child()
                temp_game.make_move(action)

            # 2. Expansion & 3. Simulation (Evaluation)
            # The game state at the leaf node
            board_state = temp_game.get_board_state()
            
            # Check if the game is over at this leaf
            if temp_game.game_over:
                value = -1.0 if temp_game.winner != self.game.current_player and temp_game.winner != 0 else 0.0
            else:
                # Prepare board state for the network
                board_tensor = torch.from_numpy(board_state).float().to(self.device).unsqueeze(0)
                
                with torch.no_grad():
                    policy, value_tensor = self.net(board_tensor)
                
                value = value_tensor.item()
                
                # Get valid moves and their corresponding probabilities
                valid_moves = temp_game.get_valid_moves()
                action_probs = torch.exp(policy).squeeze(0).cpu().numpy()
                
                # Filter probabilities for valid moves
                valid_action_probs = []
                for move in valid_moves:
                    move_index = move[0] * config.BOARD_SIZE + move[1]
                    valid_action_probs.append(((move[0], move[1]), action_probs[move_index]))
                
                # Expand the leaf node
                node.expand(valid_action_probs)
            
            # 4. Backpropagation
            while node is not None:
                node.update(-value) # We negate the value because it's from the perspective of the next player
                value = -value
                node = node.parent
                
        # After all simulations, determine the move probabilities
        move_probs = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
        total_visits = sum(child.visit_count for child in self.root.children.values())
        
        if total_visits > 0:
            for action, child in self.root.children.items():
                move_index = action[0] * config.BOARD_SIZE + action[1]
                move_probs[move_index] = child.visit_count / total_visits
                
        return move_probs