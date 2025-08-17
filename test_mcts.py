# gomoku-rl/test_mcts.py

import torch
import numpy as np

# Import our components from the gomoku_rl package
from gomoku_rl.game_logic import GomokuGame
from gomoku_rl.neural_net import GomokuNet
from gomoku_rl.mcts import MCTS
from gomoku_rl import config

def run_mcts_test():
    """
    Initializes all components and runs a single MCTS search to ensure they integrate correctly.
    """
    print("--- Starting MCTS Integration Test ---")

    # 1. Set up device (for Mac's GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize the game, network, and MCTS
    game = GomokuGame(board_size=config.BOARD_SIZE, win_condition=config.WIN_CONDITION)
    net = GomokuNet(board_size=config.BOARD_SIZE, 
                    num_residual_blocks=config.NUM_RESIDUAL_BLOCKS, 
                    num_channels=config.NUM_CHANNELS).to(device)
    
    # Let's add a few moves to the board to make the test more realistic
    game.make_move((6, 6)) # P1
    game.make_move((6, 7)) # P2
    print("Initialized Game, Network, and made a few moves.")
    game.display()

    # 3. Run the MCTS search
    print(f"Running MCTS with {config.MCTS_SIMULATIONS} simulations...")
    mcts = MCTS(game, net, device)
    move_probs = mcts.search(num_simulations=config.MCTS_SIMULATIONS)
    print("MCTS search completed without errors.")

    # 4. Verify the output
    print("\n--- Verifying Output ---")
    board_squares = config.BOARD_SIZE * config.BOARD_SIZE
    
    print(f"Output shape: {move_probs.shape}")
    assert move_probs.shape == (board_squares,), "Move probabilities shape is incorrect!"

    print(f"Sum of probabilities: {np.sum(move_probs):.4f}")
    assert np.isclose(np.sum(move_probs), 1.0), "Probabilities do not sum to 1!"
    
    print("A sample of non-zero move probabilities:")
    for i, prob in enumerate(move_probs):
        if prob > 0:
            row, col = i // config.BOARD_SIZE, i % config.BOARD_SIZE
            print(f"  Move ({row}, {col}): Probability {prob:.4f}")
    
    print("\n✅ MCTS Integration Test Passed!")


if __name__ == "__main__":
    run_mcts_test()