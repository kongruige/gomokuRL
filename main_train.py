# gomoku-rl/main_train.py

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import random

from gomoku_rl.game_logic import GomokuGame
from gomoku_rl.neural_net import GomokuNet
from gomoku_rl.mcts import MCTS
from gomoku_rl import config

def execute_episode(net, device):
    """
    Executes one full game of self-play, collecting training data.
    Returns a list of (board_state, move_probs, winner) tuples.
    """
    game = GomokuGame(board_size=config.BOARD_SIZE, win_condition=config.WIN_CONDITION)
    train_examples = []
    
    while not game.game_over:
        # Get move probabilities from MCTS
        mcts = MCTS(game.clone(), net, device)
        move_probs = mcts.search(num_simulations=config.MCTS_SIMULATIONS)
        
        # Store the state, probabilities, and current player
        board_state = game.get_board_state()
        train_examples.append([board_state, move_probs, game.current_player])
        
        # Choose a move based on the MCTS probabilities and make it
        move_index = np.random.choice(len(move_probs), p=move_probs)
        move = (move_index // config.BOARD_SIZE, move_index % config.BOARD_SIZE)
        
        game.make_move(move)

    # After the game is over, update the training examples with the winner
    winner = game.winner
    for i, example in enumerate(train_examples):
        player = example[2]
        if winner == 0: # Draw
            example[2] = 0
        else:
            # If the player for this turn won, the value is 1, otherwise -1
            example[2] = 1 if player == winner else -1
            
    return train_examples

def train():
    """
    The main training loop.
    """
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    net = GomokuNet(board_size=config.BOARD_SIZE, 
                    num_residual_blocks=config.NUM_RESIDUAL_BLOCKS, 
                    num_channels=config.NUM_CHANNELS).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # Training data buffer
    training_data = []

    # Main training loop
    num_iterations = 100
    for i in range(num_iterations):
        print(f"--- Iteration {i+1}/{num_iterations} ---")

        # 1. Self-Play Phase
        print("Starting self-play phase...")
        num_episodes = 25
        iteration_train_examples = []
        for _ in tqdm(range(num_episodes), desc="Self-Play"):
            iteration_train_examples.extend(execute_episode(net, device))
        
        training_data.extend(iteration_train_examples)
        
        # Keep buffer size manageable
        if len(training_data) > 20000:
            training_data = training_data[-20000:]

        # 2. Training Phase
        print("Starting training phase...")
        num_epochs = 10
        batch_size = 64
        
        net.train() # Set model to training mode
        for epoch in tqdm(range(num_epochs), desc="Training"):
            random.shuffle(training_data)
            
            for batch_idx in range(0, len(training_data), batch_size):
                sample = training_data[batch_idx:batch_idx+batch_size]
                states, target_policies, target_values = zip(*sample)

                states = torch.from_numpy(np.array(states)).float().to(device)
                target_policies = torch.from_numpy(np.array(target_policies)).float().to(device)
                target_values = torch.from_numpy(np.array(target_values)).float().to(device).unsqueeze(1)

                # Forward pass
                pred_policies, pred_values = net(states)

                # Calculate loss
                policy_loss = -torch.sum(target_policies * pred_policies) / target_policies.size(0)
                value_loss = torch.nn.functional.mse_loss(pred_values, target_values)
                loss = policy_loss + value_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Iteration {i+1} complete. Final loss: {loss.item():.4f}")

        # 3. Save a checkpoint
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save(net.state_dict(), f"checkpoints/model_{i+1}.pth")
        print(f"Saved checkpoint model_{i+1}.pth")
        print("-" * 50)

if __name__ == "__main__":
    train()