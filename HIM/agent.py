import json
import random
from utils import board_to_key, valid_moves, valid_pos
from train import HIMTrainer


class HIMAgent:

    def __init__(self, player=2, learned_file="states.json"):
        self.player = player
        self.learned_moves = {}
        try:
            with open(learned_file, "r") as f:
                self.learned_moves = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {learned_file} not found. Initiating training: ")
            self.train()
        
        self.valid_moves = valid_moves
        self.valid_pos = valid_pos
        self.board_to_key = board_to_key

    def train(self):
        trainer = HIMTrainer()
        trainer.train()
        trainer.save_states()
        self.learned_moves = trainer.game_data

    def get_action(self, board, player_turn):
        if player_turn != self.player:
            return None
        
        state = self.board_to_key(board)
        if state in self.learned_moves and len(self.learned_moves[state]) > 0:
            state_data = self.learned_moves[state]
            moves = list(state_data.keys())
            weights = list(state_data.values())
            move_str = random.choices(moves, weights=weights, k=1)[0]
            move = tuple(map(int, move_str))
            return move
        else:
            return self.get_random_action(board, player_turn)

    def get_random_action(self, board, player_turn):
        possible_pos = self.valid_pos(gameinfo=(board, player_turn))
        possible_moves = []
        for sr, sc in possible_pos:
            moves = self.valid_moves(board, sr, sc, player_turn)
            for er, ec in moves:
                possible_moves.append((sr, sc, er, ec))
        if possible_moves:
            return random.choice(possible_moves)