import json
from utils import board_to_key, valid_moves, valid_pos
import copy

ROWS, COLS = 3, 3

INITIAL_BOARD = [
    [2, 2, 2],
    [0, 0, 0],
    [1, 1, 1],
]

class BruteForceSearch:
    def __init__(self):
        self.table = [{},{}]
        self.define()


    def define(self):
        self.valid_moves = valid_moves
        self.valid_pos = valid_pos
        self.get_state = board_to_key

    def save_game(self, game_data, winner):
        k = winner - 1
        for key in game_data.keys():
            if key not in self.table[k]:
                self.table[k][key] = []
            for action in game_data[key]:
                if action not in self.table[k][key]:
                    self.table[k][key].append(action)
                
        

    def check_winner(self,board,player_turn):
        for col in range(COLS):
            if board[0][col] == 1:
                return 1
            if board[ROWS - 1][col] == 2:
                return 2

        p1_moves = sum(
            len(self.valid_moves(board,r, c, 1)) for r in range(ROWS) for c in range(COLS) if board[r][c] == 1
        )
        p2_moves = sum(
            len(self.valid_moves(board,r, c, 2)) for r in range(ROWS) for c in range(COLS) if board[r][c] == 2
        )
        valid_moves = [p1_moves,p2_moves]
        if valid_moves[~(player_turn-1)] == 0:
            return player_turn

        return None
        

    def recursive_search(self, board, player_turn, game_data=None):
        
        state = self.get_state(board)
        print(f"Exploring state: {state} for player {player_turn}")
        valid_positions = self.valid_pos(gameinfo=(board,player_turn))
        for pos in valid_positions:
            sr, sc = pos
            valid_moves = self.valid_moves(board, sr, sc, player_turn)
            for move in valid_moves:
                er, ec = move
                new_board=copy.deepcopy(board)
                new_board[sr][sc] = 0
                new_board[er][ec] = player_turn
                new_game_data = game_data if game_data is not None else {}
                if state not in new_game_data:
                    new_game_data[state] = []
                new_game_data[state].append((sr, sc, er, ec))
                winner = self.check_winner(board,player_turn)
                if winner:
                    self.save_game(new_game_data, winner)
                else:
                    self.recursive_search(new_board, 3 - player_turn, new_game_data)


if __name__ == "__main__":
    agent = BruteForceSearch()
    agent.recursive_search(INITIAL_BOARD, 1)
    
    with open("brute_force_table_1.json", "w") as f:
        json.dump(agent.table[0], f, indent=4)

    with open("brute_force_table_2.json", "w") as f:
        json.dump(agent.table[1], f, indent=4)
    
    print("Brute force search completed and data saved to brute_force_table_2.json")

    

    