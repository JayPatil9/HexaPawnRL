import json
import random

class BruteForceAgent:
    def __init__(self):
        self.table_1 = json.load(open("brute_force_table_1.json"))
        self.table_2 = json.load(open("brute_force_table_2.json"))

    def define(self, valid_moves, valid_pos, board_to_key):
        self.valid_moves = valid_moves
        self.valid_pos = valid_pos
        self.board_to_key = board_to_key

    
    def get_action(self, board, player_turn):
        board = board
        board_key = self.board_to_key(board)

        
        if player_turn == 1:
            if board_key in self.table_1:
                valid_moves = self.table_1[board_key]
            else:
                valid_moves = None
        else:
            if board_key in self.table_2:
                valid_moves = self.table_2[board_key]
            else:
                valid_moves = None
        
        if not valid_moves:
            return self.get_random_action(board, player_turn)
        
        move = random.choice(valid_moves)
        return (move[i] for i in range(4))
    
    def get_random_action(self, board, player_turn):
        valid_pos = self.valid_pos(gameinfo=(board, player_turn))
        
        while True:
            start = random.randint(0, len(valid_pos) - 1) if len(valid_pos) > 1 else 0
            sr, sc = valid_pos[start]

            valid_moves = self.valid_moves(board, sr, sc, player_turn)
            if len(valid_moves) == 0:
                continue
            end = random.randint(0, len(valid_moves) - 1) if len(valid_moves) > 1 else 0
            er, ec = valid_moves[end]
            break

        return (sr, sc, er, ec)
