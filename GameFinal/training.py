from environment import GoBoard, compute_reward
from agent import TDAgent, MoveAction
import time


def play_training_episode(agent: TDAgent, opponent: TDAgent, board_size: int, epsilon: float) -> float:
    board = GoBoard(board_size)
    agent.reset_traces()
    opponent.reset_traces()

    pass_count = 0

    while pass_count < 2:
        for current_agent, color, opp_color in [
            (agent,    "b", "w"),
            (opponent, "w", "b"),
        ]:
            if pass_count >= 2:
                break

            state_id = current_agent.get_state_id(board, color)
            current_agent.visit_state(state_id)

            action = current_agent.get_learned_move(board, color, epsilon=epsilon)

            if action.action_type == "pass":
                pass_count += 1
            else:
                pass_count = 0
                if action.action_type == "place":
                    board.place_stone(action.end_pos, color)
                elif action.action_type == "move":
                    board.move_stone(action.start_pos, action.end_pos, color)

            next_state_id = current_agent.get_state_id(board, color)
            current_agent.update_value(state_id, reward=0.0, next_state_id=next_state_id, done=False)
            current_agent.decay_traces()

    game_result = board.get_game_result()

    agent_reward    = compute_reward(game_result, "b")
    opponent_reward = compute_reward(game_result, "w")

    agent.update_value(agent.get_state_id(board, "b"),       reward=agent_reward,    next_state_id=None, done=True)
    opponent.update_value(opponent.get_state_id(board, "w"), reward=opponent_reward, next_state_id=None, done=True)

    return agent_reward


def train(
    episodes: int   = 10_000,
    board_size: int = 9,
    epsilon_start: float = 1.0,
    epsilon_end:   float = 0.05,
    save_every:    int   = 100,
):
    start_time = time.time()
    agent    = TDAgent(learning_rate=0.1, gamma=0.99, lambda_trace=0.9)
    opponent = TDAgent(learning_rate=0.1, gamma=0.99, lambda_trace=0.9)
    agent.load("values_black.json")
    opponent.load("values_white.json")

    # Draws really should be impossible. If you see any in the result something unexpected happened.
    wins, losses, draws = 0, 0, 0

    for episode in range(1, episodes + 1):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / episodes))

        reward = play_training_episode(agent, opponent, board_size, epsilon)

        if reward > 0:   wins   += 1
        elif reward < 0: losses += 1
        else:            draws  += 1

        if episode % 100 == 0:
            elapsed = time.time() - start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            total    = wins + losses + draws
            win_rate = wins / total * 100
            print(f"Episode {episode:>6} | ε={epsilon:.3f} | "
                f"W {wins} / L {losses} / D {draws} | "
                f"Win rate: {win_rate:.1f}% | "
                f"States known: {len(agent.value_function)} | "
                f"Elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            print(f"Episode {episode:>6} | ε={epsilon:.3f} | "
                  f"W {wins} / L {losses} / D {draws} | "
                  f"Win rate: {win_rate:.1f}% | "
                  f"States known: {len(agent.value_function)}")
            wins, losses, draws = 0, 0, 0

        if episode % save_every == 0:
            agent.save("values_black.json")
            opponent.save("values_white.json")
            print(f"  → Checkpoint saved at episode {episode}")

    agent.save("values_black.json")
    opponent.save("values_white.json")
    print("Training complete.")


if __name__ == "__main__":
    train(episodes=1000, board_size=9)