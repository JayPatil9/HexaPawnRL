import json
import ast
from utils import board_to_key, valid_moves, valid_pos
import copy
import random

ROWS, COLS = 3, 3

INITIAL_BOARD = [
    [2, 2, 2],
    [0, 0, 0],
    [1, 1, 1],
]

class HERTrainer:

    def __init__(self):
        self.game_data = {}
        self.valid_moves = valid_moves
        self.valid_pos = valid_pos   
        self.get_state = board_to_key
        self.game_data = copy.deepcopy(self.load_states())

    def load_states(self):
        with open("states.txt", "r") as f:
            states_dict = ast.literal_eval(f.read())
            for state in states_dict:
                states_dict[state] = [tuple(move) for move in states_dict[state]]
            return states_dict
    
    def select_random_move(self, board, player_turn):
        if player_turn == 2:
            state = self.get_state(board)
            if state in self.game_data and len(self.game_data[state]) > 0:
                return random.choice(self.game_data[state])
        
        elif player_turn == 1:
            possible_pos = self.valid_pos(gameinfo=(board, player_turn))
            possible_moves = []
            for sr, sc in possible_pos:
                moves = self.valid_moves(board, sr, sc, player_turn)
                for er, ec in moves:
                    possible_moves.append((sr, sc, er, ec))
            if possible_moves:
                return random.choice(possible_moves)
        return None
        
    def play_move(self, board, move):
        sr, sc, er, ec = move
        new_board = copy.deepcopy(board)
        new_board[er][ec] = new_board[sr][sc]
        new_board[sr][sc] = 0
        return new_board

    def delete_losing_moves(self, last_move):
        state, move = last_move
        if state in self.game_data and move in self.game_data[state]:
            self.game_data[state].remove(move)
        if state in self.game_data and len(self.game_data[state]) == 0:
            del self.game_data[state]
        
    def check_winner(self, board, player_turn):
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
        valid_moves = [p1_moves, p2_moves]
        opponent = 2 - player_turn
        if(valid_moves[opponent] == 0):
            return player_turn
        else:
            return None
        
    def play_episode(self):
        board = copy.deepcopy(INITIAL_BOARD)
        player_turn = 1
        game_history = []

        while True:
            state = self.get_state(board)
            if player_turn == 2:
                move = self.select_random_move(board, player_turn)
                if move is None:
                    return False
                board = self.play_move(board, move)
                game_history.append((state, move))
            else:
                move = self.select_random_move(board, player_turn)
                if move is None:
                    return True
                board = self.play_move(board, move)
            winner = self.check_winner(board, player_turn)
            if winner is not None:
                if winner == 1:
                    self.delete_losing_moves(game_history[-1])
                    return False
                else: 
                    return True
            player_turn = 3 - player_turn
            
                
    def train(self, num_episodes=100):
        trainer_wins = 0
        trainer_losses = 0

        for episode in range(num_episodes):
            trainer_won = self.play_episode()
            if trainer_won:
                trainer_wins += 1
            else:
                trainer_losses += 1

        print(f"\n=== Training Complete ===")
        print(f"Total Wins: {trainer_wins}")
        print(f"Total Losses: {trainer_losses}")
        print(f"Win Rate: {trainer_wins / num_episodes * 100:.2f}%")

    def save_states(self, filename="states.json"):
        with open(filename, "w") as f:
            json.dump(self.game_data, f, indent=4)

if __name__ == "__main__":
    trainer = HERTrainer()
    trainer.train()
    trainer.save_states()