# gomoku-rl/gomoku_rl/config.py

# Game settings
BOARD_SIZE = 13
WIN_CONDITION = 5

# Neural Network settings
NUM_RESIDUAL_BLOCKS = 3
NUM_CHANNELS = 64

# MCTS settings
MCTS_SIMULATIONS = 100 # Number of simulations per move
CPUCT = 1.0 # Exploration constant