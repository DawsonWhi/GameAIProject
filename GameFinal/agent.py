import random
from environment import GoBoard, Position


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
        
        Args:
            board: The game board
            color: Player color ('b' or 'w')
        
        Returns:
            MoveAction representing the chosen move
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
