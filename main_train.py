# gomoku-rl/main_train.py

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import random
import re # --- NEW: For parsing filenames ---

from gomoku_rl.game_logic import GomokuGame
from gomoku_rl.neural_net import GomokuNet
from gomoku_rl.mcts import MCTS
from gomoku_rl import config

# --- NEW: Helper function to find the latest checkpoint ---
def find_latest_checkpoint():
    if not os.path.exists("checkpoints"):
        return None, 0
    
    checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("model_") and f.endswith(".pth")]
    if not checkpoints:
        return None, 0

    # Find the checkpoint with the highest iteration number
    latest_iteration = -1
    latest_checkpoint_file = None
    for cp in checkpoints:
        match = re.search(r'model_(\d+).pth', cp)
        if match:
            iteration = int(match.group(1))
            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_checkpoint_file = cp
    
    if latest_checkpoint_file:
        return os.path.join("checkpoints", latest_checkpoint_file), latest_iteration
    return None, 0

def execute_episode(net, device):
    """
    Executes one full game of self-play, collecting training data.
    Returns a list of (board_state, move_probs, winner) tuples.
    """
    # (This function remains unchanged)
    game = GomokuGame(board_size=config.BOARD_SIZE, win_condition=config.WIN_CONDITION)
    train_examples = []
    
    while not game.game_over:
        mcts = MCTS(game.clone(), net, device)
        move_probs = mcts.search(num_simulations=config.MCTS_SIMULATIONS)
        
        board_state = game.get_board_state()
        train_examples.append([board_state, move_probs, game.current_player])
        
        move_index = np.random.choice(len(move_probs), p=move_probs)
        move = (move_index // config.BOARD_SIZE, move_index % config.BOARD_SIZE)
        
        game.make_move(move)

    winner = game.winner
    for i, example in enumerate(train_examples):
        player = example[2]
        if winner == 0:
            example[2] = 0
        else:
            example[2] = 1 if player == winner else -1
            
    return train_examples

def train():
    """
    The main training loop with resume capability.
    """
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    net = GomokuNet(board_size=config.BOARD_SIZE, 
                    num_residual_blocks=config.NUM_RESIDUAL_BLOCKS, 
                    num_channels=config.NUM_CHANNELS).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # --- NEW: Load from checkpoint if available ---
    start_iteration = 1
    latest_checkpoint, latest_iteration = find_latest_checkpoint()
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"Starting from iteration {start_iteration}")
    else:
        print("No checkpoint found, starting from scratch.")
    # --- END NEW ---

    training_data = []
    num_iterations = 100
    
    # --- MODIFIED: Main training loop starts from the correct iteration ---
    for i in range(start_iteration, num_iterations + 1):
        print(f"--- Iteration {i}/{num_iterations} ---")

        # 1. Self-Play Phase
        # (This section remains unchanged)
        print("Starting self-play phase...")
        num_episodes = 25
        iteration_train_examples = []
        for _ in tqdm(range(num_episodes), desc="Self-Play"):
            iteration_train_examples.extend(execute_episode(net, device))
        
        training_data.extend(iteration_train_examples)
        
        if len(training_data) > 20000:
            training_data = training_data[-20000:]

        # 2. Training Phase
        # (This section remains unchanged)
        print("Starting training phase...")
        num_epochs = 10
        batch_size = 64
        
        net.train()
        for epoch in tqdm(range(num_epochs), desc="Training"):
            random.shuffle(training_data)
            
            for batch_idx in range(0, len(training_data), batch_size):
                sample = training_data[batch_idx:batch_idx+batch_size]
                states, target_policies, target_values = zip(*sample)

                states = torch.from_numpy(np.array(states)).float().to(device)
                target_policies = torch.from_numpy(np.array(target_policies)).float().to(device)
                target_values = torch.from_numpy(np.array(target_values)).float().to(device).unsqueeze(1)

                pred_policies, pred_values = net(states)

                policy_loss = -torch.sum(target_policies * pred_policies) / target_policies.size(0)
                value_loss = torch.nn.functional.mse_loss(pred_values, target_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Iteration {i} complete. Final loss: {loss.item():.4f}")

        # --- MODIFIED: Save checkpoint with optimizer state and iteration number ---
        checkpoint_data = {
            'iteration': i,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint_data, f"checkpoints/model_{i}.pth")
        print(f"Saved checkpoint model_{i}.pth")
        print("-" * 50)


if __name__ == "__main__":
    train()