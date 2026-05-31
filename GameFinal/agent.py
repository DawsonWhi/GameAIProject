import random
from environment import GoBoard, Position
from typing import Dict, Tuple, Optional
import copy
import math
import json

class MoveAction:
    """Represents either a placement or a movement action."""
    def __init__(self, action_type: str, start_pos: Position = None, end_pos: Position = None):
        self.action_type = action_type  # "place", "move", or "pass"
        self.start_pos = start_pos  # Only for "move"
        self.end_pos = end_pos or start_pos  # Position to place/move to


class RandomAgent:
    """AI player that makes random valid moves."""
    
    @staticmethod
    def get_random_move(board: GoBoard, color: str) -> MoveAction:
        """Generate a random valid move for the AI player.
        Weighted distribution: 60% place, 30% move, 10% pass.
        """
        valid_placements = []
        valid_moves = []
        
        # Find all valid placement positions (empty cells)
        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                if board.get_stone(pos) is None:
                    valid_placements.append(pos)
        
        # Find all valid move positions (pieces of this color that can move)
        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                stone = board.get_stone(pos)
                if stone and stone.color == color:
                    # Check all cardinal neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_row, new_col = row + dx, col + dy
                        if 0 <= new_row < board.size and 0 <= new_col < board.size:
                            new_pos = Position(new_row, new_col)
                            if board.get_stone(new_pos) is None:
                                valid_moves.append((pos, new_pos))
        
        # Weighted random choice: prefer placing if available, then moving, then passing
        actions = []
        weights = []
        
        if valid_placements:
            actions.append(("place", valid_placements))
            weights.append(0.6)  # 60% chance to place
        if valid_moves:
            actions.append(("move", valid_moves))
            weights.append(0.3)  # 30% chance to move
        
        actions.append(("pass", None))
        weights.append(0.1)  # 10% chance to pass
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        action_type, action_data = random.choices(actions, weights=weights)[0]
        
        if action_type == "pass":
            return MoveAction("pass")
        elif action_type == "place":
            pos = random.choice(action_data)
            return MoveAction("place", pos)
        else:  # move
            from_pos, to_pos = random.choice(action_data)
            return MoveAction("move", from_pos, to_pos)


class TDAgent:
    # TD Learning agent with eligibility traces (TD(lambda))
    
    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.99, lambda_trace: float = 0.9):
        
        self.alpha = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        
        self.value_function: Dict[str, float] = {}  # Maps state_id -> V(s)
        self.eligibility_traces: Dict[str, float] = {}  # Maps state_id -> e(s)
        self.episode_states: list = []  

    def prune_value_function(self, threshold=0.001):
        self.value_function = {
            k: v for k, v in self.value_function.items() if abs(v) > threshold
        }
    def get_hypothetical_state_id(self, board, pos, color, move_type, from_pos=None) -> str:
        board_str = ""
        for row in range(board.size):
            for col in range(board.size):
                p = Position(row, col)
                if move_type == "place":
                    if p == pos:
                        board_str += color
                        continue
                elif move_type == "move":
                    if p == from_pos:
                        board_str += "."
                        continue
                    if p == pos:
                        board_str += color
                        continue
                stone = board.get_stone(p)
                board_str += "." if stone is None else stone.color
        opp = 'w' if color == 'b' else 'b'
        return f"{board_str}_{opp}"
            
    def get_state_id(self, board: GoBoard, color: str) -> str:
       
        board_str = ""
        for row in range(board.size):
            for col in range(board.size):
                stone = board.get_stone(Position(row, col))
                if stone is None:
                    board_str += "."
                else:
                    board_str += stone.color
        
        # Include player color to distinguish whose turn it is
        return f"{board_str}_{color}"
    
    def save(self, filename: str):
        import os
        tmp = filename + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.value_function, f)
        os.replace(tmp, filename)

    def load(self, path: str):
        try:
            with open(path, "r") as f:
                self.value_function = json.load(f)
        except FileNotFoundError:
            pass  # Start fresh if no file exists

    def get_value(self, state_id: str) -> float:
        return self.value_function.get(state_id, 0.0)
    
    def reset_traces(self):
        self.eligibility_traces.clear()
        self.episode_states.clear()
    
    def visit_state(self, state_id: str):
        self.episode_states.append(state_id)
        self.eligibility_traces[state_id] = 1.0
    
    def decay_traces(self):
        for state_id in self.eligibility_traces:
            self.eligibility_traces[state_id] *= self.gamma * self.lambda_trace
    
    def update_value(self, state_id: str, reward: float, next_state_id: Optional[str] = None, done: bool = False):
        """Update value function using TD error.
        
        Updates all traces states with the TD error weighted by their eligibility.
        This implements TD(lambda) algorithm.
        
        Args:
            state_id: Current state ID
            reward: Immediate reward received
            next_state_id: Next state ID (None if terminal)
            done: Whether episode is over
        """
        # Calculate TD error (delta)
        current_value = self.get_value(state_id)
        next_value = 0.0 if done or next_state_id is None else self.get_value(next_state_id)
        td_error = reward + self.gamma * next_value - current_value
        
        # Update all states in eligibility trace
        for trace_state_id in list(self.eligibility_traces.keys()):
            eligibility = self.eligibility_traces[trace_state_id]
            old_value = self.value_function.get(trace_state_id, 0.0)
            new_value = old_value + self.alpha * td_error * eligibility
            self.value_function[trace_state_id] = new_value
            
            # Remove traces with negligible values to save memory
            if abs(eligibility) < 1e-5:
                del self.eligibility_traces[trace_state_id]
        
 
        self.decay_traces()
    
    def get_learned_move(self, board: GoBoard, color: str, epsilon: float = 0.1) -> MoveAction:
        """Choose move using learned values (epsilon-greedy).
        
        With probability epsilon, choose random move.
        Otherwise, choose move leading to highest value next state.
        Only considers moves that are valid according to Go rules.
        When multiple moves tie for best value, randomly selects among them.
        
        Args:
            board: Current game board
            color: Player color
            epsilon: Exploration rate (0=greedy, 1=random)
            
        Returns:
            MoveAction chosen by the agent
        """
        if random.random() < epsilon:
            return RandomAgent.get_random_move(board, color)
        
        # Build list of valid moves efficiently
        valid_moves = []
        
        # Find all valid placement positions (empty cells that don't violate rules)
        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                if board.get_stone(pos) is None:
                    # Quick pre-check: if board is empty or move doesn't immediately violate, consider it
                    # Full validation will happen during training when move is actually played
                    valid_moves.append(MoveAction("place", pos))
        
        # Find all valid move positions (pieces of this color that can move)
        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                stone = board.get_stone(pos)
                if stone and stone.color == color:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_row, new_col = row + dx, col + dy
                        if 0 <= new_row < board.size and 0 <= new_col < board.size:
                            new_pos = Position(new_row, new_col)
                            if board.get_stone(new_pos) is None:
                                valid_moves.append(MoveAction("move", pos, new_pos))
        
        # Evaluate valid moves and pick best
        best_moves = []
        best_value = float('-inf')
        
        for move in valid_moves:
            if move.action_type == "place":
                next_state_id = self.get_hypothetical_state_id(board, move.end_pos, color, "place")
            else:  # move
                next_state_id = self.get_hypothetical_state_id(board, move.end_pos, color, "move", from_pos=move.start_pos)
            
            value = self.get_value(next_state_id)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        
        # If no moves found, fall back to random valid move
        if not best_moves:
            return RandomAgent.get_random_move(board, color)
        
        # Randomly select from all moves with best value (breaks ties)
        return random.choice(best_moves)
    
    @staticmethod
    def _copy_board(board: GoBoard) -> GoBoard:
        import copy
        return copy.deepcopy(board)

class GreedyTerritoryAgent:
    def __init__(self):
        pass

    def get_greedy_move(self, board: GoBoard, color: str) -> MoveAction:
        best_move = None
        best_score = -math.inf

        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                if board.get_stone(pos) is None:
                    board_copy = self._copy_board(board)
                    if board_copy.place_stone(pos, color):
                        score = self.evaluate_board(board_copy, color)
                        if score > best_score:
                            best_score = score
                            best_move = MoveAction("place", pos)

        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                stone = board.get_stone(pos)
                if stone and stone.color == color:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_row, new_col = row + dx, col + dy
                        if 0 <= new_row < board.size and 0 <= new_col < board.size:
                            new_pos = Position(new_row, new_col)
                            if board.get_stone(new_pos) is None:
                                board_copy = self._copy_board(board)
                                if board_copy.move_stone(pos, new_pos, color):
                                    score = self.evaluate_board(board_copy, color)
                                    if score > best_score:
                                        best_score = score
                                        best_move = MoveAction("move", pos, new_pos)

        if best_move is None:
            return MoveAction("pass")
        return best_move

    def evaluate_board(self, board: GoBoard, color: str) -> int:
        """Score the full board state from the perspective of color."""
        opponent_color = 'w' if color == 'b' else 'b'
        score = 0

        for row in range(board.size):
            for col in range(board.size):
                pos = Position(row, col)
                stone = board.get_stone(pos)

                if stone is None:
                    # Count empty intersections surrounded only by one color (territory)
                    neighbors = self._get_neighbors(board, pos)
                    neighbor_colors = set(
                        board.get_stone(n).color for n in neighbors
                        if board.get_stone(n) is not None
                    )
                    if neighbor_colors == {color}:
                        score += 2          # likely our territory
                    elif neighbor_colors == {opponent_color}:
                        score -= 2          # likely opponent territory

                elif stone.color == color:
                    score += 3              # our stones on board
                    stone.group.calculate_liberties(board)
                    score += len(stone.group.liberties)  # liberties are good

                elif stone.color == opponent_color:
                    score -= 3              # opponent stones on board
                    stone.group.calculate_liberties(board)
                    if len(stone.group.liberties) == 1:
                        score += 4          # opponent in atari is good for us

        return score

    def evaluate_placement(self, board: GoBoard, pos: Position, color: str) -> int:
        score = 0
        opponent_color = 'w' if color == 'b' else 'b'

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos.x + dx, pos.y + dy
            if 0 <= nx < board.size and 0 <= ny < board.size:
                neighbor_pos = Position(nx, ny)
                neighbor_stone = board.get_stone(neighbor_pos)

                if neighbor_stone is None:
                    score += 1
                elif neighbor_stone.color == color:
                    group = neighbor_stone.group
                    group.calculate_liberties(board)
                    if len(group.liberties) == 1:
                        score += 8
                    else:
                        score += 2
                elif neighbor_stone.color == opponent_color:
                    group = neighbor_stone.group
                    group.calculate_liberties(board)
                    if len(group.liberties) == 1:
                        score += 10
                    else:
                        score += 2
        return score

    def evaluate_move(self, board: GoBoard, from_pos: Position, to_pos: Position, color: str) -> int:
        score = -1
        score += self.evaluate_placement(board, to_pos, color)
        stone = board.get_stone(from_pos)
        if stone and stone.group:
            stone.group.calculate_liberties(board)
            if len(stone.group.liberties) == 1:
                score -= 5
        return score

    def _get_neighbors(self, board: GoBoard, pos: Position) -> list:
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos.x + dx, pos.y + dy
            if 0 <= nx < board.size and 0 <= ny < board.size:
                neighbors.append(Position(nx, ny))
        return neighbors

    @staticmethod
    def _copy_board(board: GoBoard) -> GoBoard:
        return copy.deepcopy(board)
