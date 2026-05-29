"""
Example: Training a TD(lambda) agent to play the game.

This demonstrates how to use the TDAgent class for learning from game play.
"""

import sys
sys.path.insert(0, 'GameFinal')

from environment import GoBoard, compute_reward
from agent import TDAgent, RandomAgent, MoveAction
from players import Player, PlayerInfo


def train_td_agent(agent: TDAgent, opponent: TDAgent = None, num_episodes: int = 100, boardsize: int = 5):
    """Train a TD agent through self-play episodes.
    
    Args:
        agent: TDAgent to train (plays as black)
        opponent: TDAgent as opponent (plays as white). If None, uses RandomAgent.
        num_episodes: Number of training episodes to run
        boardsize: Size of the game board
    """
    
    for episode in range(num_episodes):
        board = GoBoard(boardsize)
        current_player = 'b'  # Start with black (agent)
        pass_count = 0
        
        # Reset eligibility traces at start of episode
        agent.reset_traces()
        if opponent and isinstance(opponent, TDAgent):
            opponent.reset_traces()
        
        # Play one game
        while pass_count < 2:  # Game ends after 2 consecutive passes
            
            # Get current state
            if current_player == 'b':
                state_id = agent.get_state_id(board, 'b')
                agent.visit_state(state_id)
                
                # Choose move (epsilon-greedy)
                action = agent.get_learned_move(board, 'b', epsilon=0.1)
            else:
                state_id = agent.get_state_id(board, 'w')
                
                if opponent and isinstance(opponent, TDAgent):
                    opponent.visit_state(state_id)
                    action = opponent.get_learned_move(board, 'w', epsilon=0.1)
                else:
                    # Play against random agent
                    action = RandomAgent.get_random_move(board, 'w')
            
            # Execute move
            if action.action_type == "pass":
                pass_count += 1
            else:
                pass_count = 0
                
                if action.action_type == "place":
                    board.place_stone(action.end_pos, current_player)
                elif action.action_type == "move":
                    board.move_stone(action.start_pos, action.end_pos, current_player)
            
            # Switch player
            current_player = 'w' if current_player == 'b' else 'b'
        
        # Game over: compute final reward
        game_result = board.get_game_result()
        
        # Update agent value functions
        black_reward = compute_reward(game_result, 'b')
        white_reward = compute_reward(game_result, 'w')
        
        # Update black agent (main learning agent)
        agent.update_value(state_id, black_reward, done=True)
        
        # Update white agent if it's also a TDAgent
        if opponent and isinstance(opponent, TDAgent):
            opponent.update_value(state_id, white_reward, done=True)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            result_str = f"{game_result[0].upper()}+{game_result[1]}" if game_result[0] != 'tie' else "Tie"
            print(f"Episode {episode + 1}/{num_episodes}: {result_str} | "
                  f"Agent values: {len(agent.value_function)} states learned")


def demonstrate_learning():
    """Simple demonstration of TD learning setup."""
    print("TD(lambda) Learning Agent Demonstration\n")
    
    # Create learning agents
    black_agent = TDAgent(learning_rate=0.1, gamma=0.99, lambda_trace=0.9)
    white_agent = TDAgent(learning_rate=0.1, gamma=0.99, lambda_trace=0.9)
    
    print(f"Training two TD agents...")
    print(f"- Learning rate (alpha): 0.1")
    print(f"- Discount factor (gamma): 0.99")
    print(f"- Lambda (eligibility trace decay): 0.9")
    print(f"- Board size: 5x5\n")
    
    # Train agents through self-play
    # Uncomment below to actually run training (takes a while):
    # train_td_agent(black_agent, opponent=white_agent, num_episodes=100, boardsize=5)
    
    print("To train agents, call:")
    print("  train_td_agent(agent, opponent=opponent, num_episodes=100, boardsize=5)")
    print("\nAgent methods available:")
    print("  - get_state_id(board, color): Get hashable state representation")
    print("  - get_value(state_id): Get learned value estimate")
    print("  - visit_state(state_id): Mark state as visited (increment eligibility)")
    print("  - update_value(state_id, reward, next_state_id, done): TD update step")
    print("  - get_learned_move(board, color, epsilon): Choose move using learned values")
    print("  - reset_traces(): Reset eligibility traces for new episode")


if __name__ == "__main__":
    demonstrate_learning()
