from collections import namedtuple
from typing import List, Optional, Set, Dict
from players import Player, Point
from scoring import compute_game_result

BOARDSIZE = 19
Position = namedtuple("Position", ["x", "y"])


class Stone:
    def __init__(self, color: str, position: Position):
        self.color = color
        self.position = position
        self.group = None  # Will be set when added to a group

    def __str__(self):
        return self.color


class StoneGroup:
    def __init__(self, color: str):
        self.color = color
        self.stones = set()
        self.liberties = set()

    def add_stone(self, stone: Stone):
        self.stones.add(stone)
        stone.group = self

    def remove_stone(self, stone: Stone):
        if stone in self.stones:
            self.stones.remove(stone)
            stone.group = None

    def merge(self, other_group):
        if self.color != other_group.color:
            raise ValueError("Cannot merge groups of different colors")
        for stone in other_group.stones:
            self.add_stone(stone)
        other_group.stones = set()

    def calculate_liberties(self, board: "GoBoard"):
        """Calculate all liberties for this group"""
        self.liberties = set()
        for stone in self.stones:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Orthogonal neighbors
                pos = Position(stone.position.x + dx, stone.position.y + dy)
                if board.is_valid_position(pos) and board.get_stone(pos) is None:
                    self.liberties.add(pos)

    def has_liberties(self, board: "GoBoard") -> bool:
        self.calculate_liberties(board)
        return len(self.liberties) > 0

    def __len__(self):
        return len(self.stones)


class GoBoard:
    def __init__(self, size: int = BOARDSIZE):
        self.winner = None
        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.num_rows = size
        self.num_cols = size
        self.groups = []
        self.previous_state = None  # For ko rule
        self._grid = {}

    def get_color(self, point: Position) -> Optional[str]:
        if not (0 <= point.x < self.size and 0 <= point.y < self.size):
            return None
        stone = self.board[point.x][point.y]  # x=row, y=col
        return stone.color if stone else None

    def _replace_group(self, new_group):
        for stone in new_group.stones:
            self._grid[stone.position] = new_group

    def _remove_group(self, group: StoneGroup):
        for stone in group.stones:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_pos = Position(stone.position.x + dx, stone.position.y + dy)
                if self.is_valid_position(neighbor_pos):
                    neighbor_group = self._grid.get(neighbor_pos)
                    if neighbor_group and neighbor_group != group:
                        neighbor_group.liberties.add(stone.position)
            self._grid.pop(stone.position, None)

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row < self.size and 1 <= point.col < self.size

    def get(self, point: Point) -> Optional[Stone]:
        if not self.is_on_grid(point):
            return None
        stone = self.board[point.row-1][point.col-1]  # Convert to 0-based
        if stone is None:
            return None
        return Player.black if stone.color == 'b' else Player.white

    def place_stone(self, position: Position, color: str) -> bool:
        """Place a stone and handle all Go rules, including captures and ko"""
        if not self.is_valid_position(position) or self.get_stone(position) is not None:
            return False

        # Create new stone and temporary group
        new_stone = Stone(color, position)
        new_group = StoneGroup(color)
        new_group.add_stone(new_stone)
        self.board[position.x][position.y] = new_stone
        self._grid[position] = new_group

        adjacent_groups = self._get_adjacent_groups(position)

        # Merge with friendly groups
        for group in adjacent_groups:
            if group.color == color:
                new_group.merge(group)
                self.groups.remove(group)
                self._replace_group(new_group)
        self.groups.append(new_group)

        # Check for captures in opponent groups
        captured_groups = []
        for group in self._get_adjacent_groups(position):
            if group.color != color and not group.has_liberties(self):
                captured_groups.append(group)

        # Check for ko
        if len(captured_groups) == 1 and len(captured_groups[0].stones) == 1:
            captured_pos = next(iter(captured_groups[0].stones)).position
            if self.previous_state and self.previous_state == (captured_pos, color):
                # Ko violation - undo the move
                # This can also get AI stuck in an infinite loop
                self._remove_group(new_group)
                self.board[position.x][position.y] = None
                self._grid.pop(position, None)
                return False

        # Perform captures
        for group in captured_groups:
            self._remove_group(group)

        # Check if new move has liberties (suicide prevention)
        if not new_group.has_liberties(self) and not captured_groups:
            self._remove_group(new_group)
            self.board[position.x][position.y] = None
            self._grid.pop(position, None)
            return False

        # Update previous state for ko
        if len(captured_groups) == 1 and len(captured_groups[0].stones) == 1:
            self.previous_state = (position, color)
        else:
            self.previous_state = None

        return True

    def _get_adjacent_groups(self, position: Position) -> List[StoneGroup]:
        """Get all unique groups adjacent to a position"""
        groups = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = position.x + dx, position.y + dy
            if self.is_valid_position(Position(x, y)):
                stone = self.get_stone(Position(x, y))
                if stone and stone.group and stone.group not in groups:
                    groups.add(stone.group)
        return list(groups)

    def _remove_group(self, group: StoneGroup):
        """Remove a group from the board"""
        for stone in group.stones:
            self.board[stone.position.x][stone.position.y] = None
            self._grid.pop(stone.position, None)
        if group in self.groups:
            self.groups.remove(group)

    def is_valid_position(self, position: Position) -> bool:
        if position.x == -1 and position.y == -1:
            return True
        return 0 <= position.x < self.size and 0 <= position.y < self.size

    def get_stone(self, position: Position) -> Optional[Stone]:
        if not self.is_valid_position(position):
            return None
        return self.board[position.x][position.y]

    def move_stone(self, from_pos: Position, to_pos: Position, color: str) -> bool:
        """Move a stone from from_pos to to_pos. Only cardinal moves allowed.
        Returns True if successful, False otherwise."""
        
        # Validate source position
        if not self.is_valid_position(from_pos):
            return False
        source_stone = self.get_stone(from_pos)
        if source_stone is None or source_stone.color != color:
            return False
        
        # Validate destination position
        if not self.is_valid_position(to_pos):
            return False
        if self.get_stone(to_pos) is not None:
            return False
        
        # Validate cardinal movement (one step up/down/left/right)
        dx = abs(to_pos.x - from_pos.x)
        dy = abs(to_pos.y - from_pos.y)
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
            return False
        
        # Remove stone from source
        old_group = source_stone.group
        if old_group:
            old_group.remove_stone(source_stone)
            if len(old_group) == 0:
                self._remove_group(old_group)
            else:
                # Recalculate liberties for the remaining group
                old_group.calculate_liberties(self)
        
        # Place stone at destination
        source_stone.position = to_pos
        self.board[from_pos.x][from_pos.y] = None
        self._grid.pop(from_pos, None)
        self.board[to_pos.x][to_pos.y] = source_stone
        self._grid[to_pos] = old_group
        
        return True

    def display(self):
        """Print a simple ASCII representation of the board"""
        for row_idx in range(self.size):
            row = []
            for col_idx in range(self.size):
                stone = self.board[row_idx][col_idx]
                row.append(str(stone) if stone else ".")
            print(" ".join(row))
    
    def get_game_result(self) -> tuple:
        """Compute game result and return (winner, score_diff).
        
        Returns:
            Tuple of (winner_color, score_difference) where:
            - winner_color: 'b' or 'w' or 'tie'
            - score_difference: abs(black_score - white_score)
        """
        result_str = compute_game_result(self)
        
        # This should be impossible with our scoring function. If you're seeing ties in scoring something needs to change
        if "Tie" in str(result_str):
            return ("tie", 0.0)
        
        result_str = str(result_str)
        if "+" in result_str:
            parts = result_str.split("+")
            winner_color = parts[0].lower()
            score_diff = float(parts[1]) if len(parts) > 1 else 0.0
            return (winner_color, score_diff)
        
        return ("tie", 0.0)


def compute_reward(game_result: tuple, player_color: str) -> float:
    winner_color = game_result
    
    if winner_color == "tie":
        return 0.0
    elif winner_color == player_color:
        return 1.0
    else:
        return -1.0



def get_neighbors(row, col, board):
    neighbors = []
    if row > 0:
        neighbors.append((row - 1, col))  # up
    if row < len(board) - 1:
        neighbors.append((row + 1, col))  # down
    if col > 0:
        neighbors.append((row, col - 1))  # left
    if col < len(board[0]) - 1:
        neighbors.append((row, col + 1))  # right
    return neighbors


def is_surrounded_by_color(row, col, color, board):
    """Check if an empty spot is completely surrounded by one color"""
    if board[row][col] != None:  # not empty
        return False

    print("test")

    # check direct neighbors
    for nr, nc in get_neighbors(row, col, board):
        if board[nr][nc] != color and board[nr][nc] != None:
            print("test 2")
            return False  # false if it found diff color
    return True