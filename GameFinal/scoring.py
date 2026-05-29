# Calculates the score of a Go game based on the final board state.
#         self.komi = 6.5  # Standard komi for 19x19 Go
#         self.territory = None  # Territory will be calculated at the end of the game

from collections import namedtuple
from players import Player, Point

class Territory(object):
    def __init__(self, territory_map):
        self.num_b_territory = 0
        self.num_w_territory = 0
        self.num_b_stones = 0
        self.num_w_stones = 0        
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_b_stones += 1
            elif status == Player.white:
                self.num_w_stones += 1
            elif status == 'territory_b':
                self.num_b_territory += 1
            elif status == 'territory_w':
                self.num_w_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)

class GameResults(namedtuple('GameResult', 'b w komi')):
    @property
    def winner(self):
        if self.b > self.w + self.komi:
            return Player.black
        else:
            return Player.white
        
    def __str__(self):
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w,)
        return 'W+%.1f' % (w - self.b,)
        
def _collect_region(start_pos, board, visited=None):
    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]
    all_boarders = set()
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point(row = start_pos.row + delta_r, col = start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, boarder = _collect_region(next_p, board, visited)
            all_points += points
            all_boarders |= boarder
        else:
            all_boarders.add(neighbor)
    return all_points, all_boarders
    
def evaluate_territory(board):
    status = {}
    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            p = Point(row=r, col=c)
            if p in status:
                continue
            stone = board.get(p)
            if stone is not None:
                status[p] = stone.color
            else:
                points, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = f'territory_{stone_str}'
                else:
                    fill_with = 'dame'
                for point in points:
                    status[point] = fill_with
    return Territory(status)

        
def compute_game_result(board):
    territory = evaluate_territory(board)
    return GameResults(
        b=territory.num_b_stones + territory.num_b_territory,
        w=territory.num_w_stones + territory.num_w_territory,
        komi=6.5
    )