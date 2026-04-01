import json
import random
from utils import board_to_key, valid_moves, valid_pos
from train import HERTrainer


class HERAgent:

    def __init__(self, player=2, learned_file="states.json"):
        self.player = player
        self.learned_moves = {}
        try:
            with open(learned_file, "r") as f:
                data = json.load(f)
                for state, moves in data.items():
                    self.learned_moves[state] = [tuple(move) if isinstance(move, list) else move for move in moves]
        except FileNotFoundError:
            print(f"Warning: {learned_file} not found. Initiating training: ")
            self.train()
        
        self.valid_moves = valid_moves
        self.valid_pos = valid_pos
        self.board_to_key = board_to_key

    def train(self):
        trainer = HERTrainer()
        trainer.train()
        trainer.save_states()
        self.game_data = trainer.game_data
        for state, moves in self.game_data.items():
            self.learned_moves[state] = [tuple(move) for move in moves]

    def get_action(self, board, player_turn):
        if player_turn != self.player:
            return None
        
        state = self.board_to_key(board)
        if state in self.learned_moves and len(self.learned_moves[state]) > 0:
            move = random.choice(self.learned_moves[state])
            return move

        return self.get_random_action(board, player_turn)

    def get_random_action(self, board, player_turn):
        print("Selecting random action...")
        possible_pos = self.valid_pos(gameinfo=(board, player_turn))
        possible_moves = []
        for sr, sc in possible_pos:
            moves = self.valid_moves(board, sr, sc, player_turn)
            for er, ec in moves:
                possible_moves.append((sr, sc, er, ec))
        if possible_moves:
            return random.choice(possible_moves)