from environment import *
from players import Player, PlayerInfo
from scoring import compute_game_result
from agent import MoveAction, RandomAgent
from collections import namedtuple


class Game:

    # Takes a Position and color of player and returns a string to give to the neural net
    def input_to_move(self, move, color):
        row = move[0] + 97
        col = move[1] + 97
        move_string = color.upper() + "[" + chr(row) + chr(col) + "]" 
        return move_string

    # Takes the string outputted by the neural net, parses it, and returns a Position
    def move_to_input(self, move) -> Position:
        for index, char in enumerate(move):
            row = move[index+1]
            col = move[index+2]
            if char == "[" and row.islower() and row.isalpha() and col.islower() and col.isalpha():
                return Position(ord(row)-97, ord(col)-97)

                

    def get_formatted_move(self, boardsize: int) -> MoveAction:
        """Get a move from the player. Accepts:
        - 'row,col' to place a piece
        - 'move row1,col1 to row2,col2' to move a piece
        - 'pass' to pass
        """
        while True:
            user_input = input(f"Enter move as 'row,col' (place) or 'move row1,col1 to row2,col2' or 'pass': ")
            user_input = user_input.strip().lower()
            
            if user_input == "pass":
                return MoveAction("pass")
            
            # Handle move command
            if user_input.startswith("move "):
                try:
                    # Parse "move row1,col1 to row2,col2"
                    parts = user_input[5:].split(" to ")
                    if len(parts) != 2:
                        print("Invalid format. Use: move row1,col1 to row2,col2")
                        continue
                    
                    start_parts = parts[0].strip().split(",")
                    end_parts = parts[1].strip().split(",")
                    
                    start_row, start_col = int(start_parts[0]) - 1, int(start_parts[1]) - 1
                    end_row, end_col = int(end_parts[0]) - 1, int(end_parts[1]) - 1
                    
                    if all(0 <= pos < boardsize for pos in [start_row, start_col, end_row, end_col]):
                        return MoveAction("move", 
                                        Position(start_row, start_col), 
                                        Position(end_row, end_col))
                    else:
                        print(f"Both coordinates must be between 1 and {boardsize}.")
                except (ValueError, IndexError):
                    print("Invalid format. Use: move row1,col1 to row2,col2")
            else:
                # Handle place command
                try:
                    row_str, col_str = user_input.split(",")
                    row, col = int(row_str) - 1, int(col_str) - 1
                    if 0 <= row < boardsize and 0 <= col < boardsize:
                        return MoveAction("place", Position(row, col))
                    else:
                        print(f"Both numbers must be between 1 and {boardsize}.")
                except ValueError:
                    print("Invalid format. Use 'row,col' or 'move row1,col1 to row2,col2'.")

    def play_game(self, player1, player2, BOARDSIZE=19):
        board = GoBoard(BOARDSIZE)
        current_player = player1
        game_over = False
        valid_move = False
        pass_flag = 0
        prev_move = None

        while not (game_over or pass_flag == 2):  # Game continues until two consecutive passes

            while not valid_move:  # valid_move is True when a valid move is made
                board.display()
                print(f"{current_player.name}'s turn ({current_player.color})")
                if current_player.is_human:
                    action = self.get_formatted_move(BOARDSIZE)
                    
                    if action.action_type == "pass":
                        print(f"{current_player.name} passes.")
                        valid_move = True
                        pass_flag += 1
                    elif action.action_type == "place":
                        valid_move = board.place_stone(action.end_pos, current_player.color)
                        if valid_move:
                            prev_move = ("place", action.end_pos)
                            pass_flag = 0
                        else:
                            print("Invalid placement. Try again.")
                    elif action.action_type == "move":
                        valid_move = board.move_stone(action.start_pos, action.end_pos, current_player.color)
                        if valid_move:
                            prev_move = ("move", action.start_pos, action.end_pos)
                            pass_flag = 0
                        else:
                            print("Invalid move. Piece must be yours, destination empty, and move cardinal (1 step up/down/left/right). Try again.")
                else:
                    # AI player makes a random move
                    action = RandomAgent.get_random_move(board, current_player.color)
                    print(f"{current_player.name} (AI) chose: {action.action_type}")
                    
                    if action.action_type == "pass":
                        print(f"{current_player.name} passes.")
                        valid_move = True
                        pass_flag += 1
                    elif action.action_type == "place":
                        valid_move = board.place_stone(action.end_pos, current_player.color)
                        if valid_move:
                            prev_move = ("place", action.end_pos)
                            pass_flag = 0
                    elif action.action_type == "move":
                        valid_move = board.move_stone(action.start_pos, action.end_pos, current_player.color)
                        if valid_move:
                            prev_move = ("move", action.start_pos, action.end_pos)
                            pass_flag = 0

            if current_player == player1:
                current_player = player2
            else:
                current_player = player1
            valid_move = False
        result = compute_game_result(board)
        print("Game over!")
        print(result)


def get_board_size() -> int:
    while True:
        try:
            size = int(input("Enter board size (3-19, standard is 19): "))
            if 3 <= size <= 19:
                return size
            else:
                print("Board size must be between 3 and 19.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_game_mode() -> tuple:
    """Prompt user to select game mode and return (player1, player2) tuple."""
    print("\nGame Mode Selection:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    
    while True:
        choice = input("Select mode (1-3): ").strip()
        
        if choice == "1":
            p1 = PlayerInfo(Player.black, name="Player 1 (Black)", is_human=True)
            p2 = PlayerInfo(Player.white, name="Player 2 (White)", is_human=True)
            return p1, p2
        elif choice == "2":
            p1 = PlayerInfo(Player.black, name="You (Black)", is_human=True)
            p2 = PlayerInfo(Player.white, name="AI (White)", is_human=False)
            return p1, p2
        elif choice == "3":
            p1 = PlayerInfo(Player.black, name="AI 1 (Black)", is_human=False)
            p2 = PlayerInfo(Player.white, name="AI 2 (White)", is_human=False)
            return p1, p2
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    board_size = get_board_size()
    player1, player2 = get_game_mode()
    test_game = Game()
    test_game.play_game(player1, player2, BOARDSIZE=board_size)