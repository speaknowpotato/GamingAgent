import torch
import numpy as np
import random
from dqn_model import DQN
from rl_env import Game2048Env


class RLAgent:
    def __init__(
        self,
        model_path="/home/bricy/workspace/GamingAgent/games/game_2048/2048_dqn.pth",
    ):
        self.env = Game2048Env()
        self.model = DQN()
        self.direction_map = ["w", "d", "s", "a"]  # up, right, down, left
        self.last_moves = []  # Track recent moves to detect loops
        self.stuck_threshold = 5  # How many repeated moves before considering "stuck"

        # Load the model from checkpoint format
        try:
            # Try loading as a checkpoint dictionary first
            checkpoint = torch.load(model_path)
            if "model_state_dict" in checkpoint:
                # This is a checkpoint dictionary
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model checkpoint from {model_path}")
            else:
                # This is a direct state dict
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model state dict from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing with random weights")

        self.model.eval()

    def get_move(self, board):
        # Convert the board to the state format expected by the model
        state = np.array(board, dtype=np.float32)
        # Apply log2 transformation to non-zero values
        state = np.log2(
            state, out=np.zeros_like(state, dtype=np.float32), where=(state > 0)
        )
        # Normalize
        state = state / 16.0

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)

            # Get all actions sorted by Q-value (best first)
            sorted_actions = q_values.squeeze().argsort(descending=True)

            # Check if we're stuck in a loop of the same move
            if len(self.last_moves) >= self.stuck_threshold and all(
                move == self.last_moves[0] for move in self.last_moves
            ):
                print("AI detected stuck in loop, trying alternative move")
                # Try the second-best action instead
                if len(sorted_actions) > 1:
                    action = sorted_actions[1].item()
                else:
                    # If somehow we only have one action, choose randomly
                    action = random.randint(0, 3)
            else:
                # Use the best action
                action = sorted_actions[0].item()

            # Update last moves history
            self.last_moves.append(action)
            if len(self.last_moves) > self.stuck_threshold:
                self.last_moves.pop(0)

            # Check if the move would actually change the board
            test_board = self.simulate_move(board, self.direction_map[action])
            if test_board == board:
                # If the best move doesn't change the board, try other moves
                for alt_action in sorted_actions[1:]:
                    alt_action = alt_action.item()
                    test_board = self.simulate_move(
                        board, self.direction_map[alt_action]
                    )
                    if test_board != board:
                        action = alt_action
                        break

            return self.direction_map[action]

    def simulate_move(self, board, direction):
        """Simulate a move without actually making it"""
        from logic import move
        from copy import deepcopy

        return move(direction, deepcopy(board))
