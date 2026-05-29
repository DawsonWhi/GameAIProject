import enum
from collections import namedtuple
from typing import Optional


class Player(enum.Enum):
    """Color enum preserved for backwards compatibility."""
    black = 1
    white = 2

    @property
    def color(self) -> str:
        return 'b' if self == Player.black else 'w'

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class PlayerInfo:
    """Represents a player (human or AI) in a game.

    Attributes:
        color_enum: Player enum value (Player.black or Player.white)
        color: single-char color used by the board ('b' or 'w')
        name: display name
        is_human: whether this player is human-controlled
    """

    def __init__(self, color_enum: Player, name: Optional[str] = None, is_human: bool = True):
        self.color_enum = color_enum
        self.color = color_enum.color
        self.name = name or ("Black" if color_enum == Player.black else "White")
        self.is_human = is_human

    def set_human_or_ai(self, is_human: bool):
        self.is_human = is_human


class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]