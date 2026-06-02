from environment import *
from players import Player, PlayerInfo
from scoring import compute_game_result
from agent import MoveAction, RandomAgent, TDAgent, GreedyTerritoryAgent, MCTSAgent
from collections import namedtuple
import random


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
                return Position(ord(col)-97, ord(row)-97)

                

    def get_formatted_move(self, boardsize: int) -> MoveAction:
        """Get a move from the player. Accepts:
        - 'row,col' to place a piece
        - 'move row1,col1 to row2,col2' to move a piece
        - 'pass' to pass
        """
        while True:
            user_input = input(f"Enter move as 'row, col' (place) or 'move row1,col1 to row2,col2' or 'pass': ")
            user_input = user_input.strip().lower()
            
            if user_input == "pass":
                return MoveAction("pass")
            
            # Handle move command
            if user_input.startswith("move "):
                try:
                    parts = user_input[5:].split(" to ")
                    if len(parts) != 2:
                        print("Invalid format. Use: move row1,col1 to row2,col2")
                        continue
                     
                    start_parts = parts[0].strip().split(",")
                    end_parts = parts[1].strip().split(",")
                    
                    start_x, start_y = int(start_parts[0]) - 1, int(start_parts[1]) - 1
                    end_x, end_y = int(end_parts[0]) - 1, int(end_parts[1]) - 1
                    
                    if all(0 <= pos < boardsize for pos in [start_x, start_y, end_x, end_y]):
                        return MoveAction("move",
                                        Position(start_x, start_y),
                                        Position(end_x, end_y))
                    else:
                        print(f"Both coordinates must be between 1 and {boardsize}.")
                except (ValueError, IndexError):
                    print("Invalid format. Use: move row1,col1 to row2,col2")
            else:
                # Handle place command
                try:
                    x_str, y_str = user_input.split(",")
                    x, y = int(x_str) - 1, int(y_str) - 1
                    if 0 <= x < boardsize and 0 <= y < boardsize:
                        return MoveAction("place", Position(x, y))
                    else:
                        print(f"Both numbers must be between 1 and {boardsize}.")
                except ValueError:
                    print("Invalid format. Use 'row,col' or 'move row1,col1 to row2,col2'.")


    def play_game(self, player1, player2, BOARDSIZE=19, ai_agent: TDAgent = None, mcts_agent: MCTSAgent = None, greedy_agent = None):
        board = GoBoard(BOARDSIZE)
        current_player = player1
        game_over = False
        valid_move = False
        pass_flag = 0
        if ai_agent is None:
            ai_agent = TDAgent()
            ai_agent.load("values_black.msgpack.gz")
        if mcts_agent is None:
            mcts_agent = MCTSAgent(iterations=50)
        if greedy_agent is None:
            greedy_agent = GreedyTerritoryAgent()

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
                            pass_flag = 0
                        else:
                            print("Invalid placement. Try again.")
                    elif action.action_type == "move":
                        valid_move = board.move_stone(action.start_pos, action.end_pos, current_player.color)
                        if valid_move:
                            pass_flag = 0
                        else:
                            print("Invalid move. Piece must be yours, destination empty, and move cardinal (1 step up/down/left/right). Try again.")
                else:
                    # AI players - select agent based on agent_type
                    action = None
                    
                    if current_player.agent_type == "td":
                        action = ai_agent.get_learned_move(board, current_player.color, epsilon=0.0)
                    elif current_player.agent_type == "mcts":
                        action = mcts_agent.get_best_move(board, current_player.color)
                    elif current_player.agent_type == "greedy":
                        action = greedy_agent.get_greedy_move(board, current_player.color)
                    else:
                        # Default to pass if agent type not recognized
                        action = MoveAction("pass")
                    
                    print(f"{current_player.name} chose: {action.action_type}")
                    
                    if action.action_type == "pass":
                        print(f"{current_player.name} passes.")
                        valid_move = True
                        pass_flag += 1
                    elif action.action_type == "place":
                        if board.place_stone(action.end_pos, current_player.color):
                            valid_move = True
                            pass_flag = 0
                        else:
                            # Invalid move, fallback to pass
                            print(f"Invalid placement, {current_player.name} passes instead.")
                            valid_move = True
                            pass_flag += 1
                    elif action.action_type == "move":
                        if board.move_stone(action.start_pos, action.end_pos, current_player.color):
                            valid_move = True
                            pass_flag = 0
                        else:
                            # Invalid move, fallback to pass
                            print(f"Invalid move, {current_player.name} passes instead.")
                            valid_move = True
                            pass_flag += 1

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
    print("2. Human vs TD Agent")
    print("3. Human vs MCTS")
    print("4. Human vs Greedy")
    print("5. TD Agent vs MCTS")
    print("6. Greedy vs MCTS")
    
    while True:
        choice = input("Select mode (1-6): ").strip()
        
        if choice == "1":
            p1 = PlayerInfo(Player.black, name="Player 1 (Black)", is_human=True)
            p2 = PlayerInfo(Player.white, name="Player 2 (White)", is_human=True)
            return p1, p2
        elif choice == "2":
            # Randomly decide if human goes first
            if random.choice([True, False]):
                p1 = PlayerInfo(Player.black, name="You (Black)", is_human=True)
                p2 = PlayerInfo(Player.white, name="TD Agent (White)", is_human=False, agent_type="td")
                print("You go first (Black)")
            else:
                p1 = PlayerInfo(Player.black, name="TD Agent (Black)", is_human=False, agent_type="td")
                p2 = PlayerInfo(Player.white, name="You (White)", is_human=True)
                print("TD Agent goes first (Black)")
            return p1, p2
        elif choice == "3":
            # Randomly decide if human goes first
            if random.choice([True, False]):
                p1 = PlayerInfo(Player.black, name="You (Black)", is_human=True)
                p2 = PlayerInfo(Player.white, name="MCTS (White)", is_human=False, agent_type="mcts")
                print("You go first (Black)")
            else:
                p1 = PlayerInfo(Player.black, name="MCTS (Black)", is_human=False, agent_type="mcts")
                p2 = PlayerInfo(Player.white, name="You (White)", is_human=True)
                print("MCTS goes first (Black)")
            return p1, p2
        elif choice == "4":
            # Randomly decide if human goes first
            if random.choice([True, False]):
                p1 = PlayerInfo(Player.black, name="You (Black)", is_human=True)
                p2 = PlayerInfo(Player.white, name="Greedy (White)", is_human=False, agent_type="greedy")
                print("You go first (Black)")
            else:
                p1 = PlayerInfo(Player.black, name="Greedy (Black)", is_human=False, agent_type="greedy")
                p2 = PlayerInfo(Player.white, name="You (White)", is_human=True)
                print("Greedy goes first (Black)")
            return p1, p2
        elif choice == "5":
            p1 = PlayerInfo(Player.black, name="TD Agent (Black)", is_human=False, agent_type="td")
            p2 = PlayerInfo(Player.white, name="MCTS (White)", is_human=False, agent_type="mcts")
            return p1, p2
        elif choice == "6":
            p1 = PlayerInfo(Player.black, name="Greedy (Black)", is_human=False, agent_type="greedy")
            p2 = PlayerInfo(Player.white, name="MCTS (White)", is_human=False, agent_type="mcts")
            return p1, p2
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    td_agent = TDAgent()
    td_agent.load("values_black.msgpack.gz")  # Load pre-trained values for TD Agent
    mcts_agent = MCTSAgent(iterations=50)  # Create MCTS agent
    greedy_agent = GreedyTerritoryAgent()  # Create Greedy Territory agent
    board_size = get_board_size()
    player1, player2 = get_game_mode()
    test_game = Game()
    test_game.play_game(player1, player2, BOARDSIZE=board_size, ai_agent=td_agent, mcts_agent=mcts_agent, greedy_agent=greedy_agent)