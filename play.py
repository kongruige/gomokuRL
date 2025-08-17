# gomoku-rl/play.py

import torch
import argparse

from gomoku_rl.game_logic import GomokuGame
from gomoku_rl.neural_net import GomokuNet
from gomoku_rl.agent import Agent
from gomoku_rl import config

def play():
    """
    Main function to run a game between a human player and the AI agent.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Play against a trained Gomoku AI.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file.")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained network
    net = GomokuNet(board_size=config.BOARD_SIZE, 
                    num_residual_blocks=config.NUM_RESIDUAL_BLOCKS, 
                    num_channels=config.NUM_CHANNELS).to(device)
    
    net.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    net.eval() # Set the network to evaluation mode
    print("Model loaded successfully from:", args.checkpoint_path)

    # Create the AI agent
    ai_agent = Agent(net, device)
    
    # --- Game Loop ---
    game = GomokuGame(board_size=config.BOARD_SIZE, win_condition=config.WIN_CONDITION)
    human_player = 1 # Let's say human is Player 1 (X)

    while not game.game_over:
        game.display()

        if game.current_player == human_player:
            # Human's turn
            try:
                move_str = input("Enter your move (row,col): ")
                row, col = map(int, move_str.split(','))
                move = (row, col)

                if move not in game.get_valid_moves():
                    print("Invalid move. Try again.")
                    continue
                
                game.make_move(move)

            except ValueError:
                print("Invalid format. Please use 'row,col'.")
                continue
        else:
            # AI's turn
            print("AI is thinking...")
            ai_move = ai_agent.choose_move(game)
            print(f"AI plays: {ai_move}")
            game.make_move(ai_move)

    # --- Game Over ---
    game.display()
    if game.winner == 0:
        print("The game is a draw!")
    elif game.winner == human_player:
        print("Congratulations, you won!")
    else:
        print("The AI wins! Better luck next time.")

if __name__ == "__main__":
    play()