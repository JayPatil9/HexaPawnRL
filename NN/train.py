import copy
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random



class HexaPawnNet(nn.Module):
    
    def __init__(self, input_size=9, hidden_size=64, output_size=4):
        super(HexaPawnNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
class HexaPawnAgent:
    
    def __init__(self, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = HexaPawnNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
    def board_to_tensor(self, board):
        flat_board = np.array(board).flatten()
        return torch.FloatTensor(flat_board).unsqueeze(0).to(self.device)
    
    def get_valid_moves(self, game):
        valid_moves = []
        valid_positions = game.valid_pos(gameinfo=(game.board, game.player_turn))
        for row, col in valid_positions:
            moves = game.valid_moves(game.board, row, col, game.player_turn)
            for move in moves:
                valid_moves.append((row, col, move[0], move[1]))
        return valid_moves
    
    def action_to_tensor(self, action):
        return torch.FloatTensor(action).to(self.device)
    
    def tensor_to_action(self, tensor):
        values = torch.clamp(tensor, 0, 2).round().int().cpu().numpy()
        return tuple(values)
    
    def choose_action(self, game):
        valid_moves = self.get_valid_moves(game)
        
        if not valid_moves:
            return None
        
        # Exploration: random move
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Exploitation: use neural network
        board_tensor = self.board_to_tensor(game.board)
        
        with torch.no_grad():
            predicted_action = self.net(board_tensor).cpu().numpy()[0]
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            move_array = np.array(move)
            score = -np.sum((predicted_action - move_array) ** 2)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.cat([self.board_to_tensor(s) for s, _, _, _, _ in batch])
        actions = torch.stack([self.action_to_tensor(a) for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        
        targets = actions.clone()
        
        for i in range(len(batch)):
            reward_scale = 1.0 + 0.1 * rewards[i]
            targets[i] = actions[i] * reward_scale
            targets[i] = torch.clamp(targets[i], 0, 2)
        
        current_predictions = self.net(states)
        loss = self.criterion(current_predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    

class HexaPawnTrainer:
    
    def __init__(self, game_class):
        self.game_class = game_class
        self.agent1 = HexaPawnAgent()
        self.agent2 = HexaPawnAgent()
        
        self.scores = []
        self.losses = []
        self.win_rates = []

    def play_game(self, agent1, agent2, train=True):
        game = self.game_class(delay=2)
        states = []
        actions = []
        
        while True:
            current_player = game.player_turn
            current_agent = agent1 if current_player == 1 else agent2

            current_state = copy.deepcopy(game.board)

            action = current_agent.choose_action(game)
            if action is None:
                break
            
            states.append((current_state, action, current_player))

            game_over = game.play_step(action)
            
            if game_over:
                winner = game.check_winner()
                
                if train:
                    self.assign_rewards_and_train(states, winner, agent1, agent2)
                
                return winner
            
    def assign_rewards_and_train(self, states, winner, agent1, agent2):
        for i, (state, action, player) in enumerate(states):
            if winner == player:
                reward = 1.0  # Win
            else:
                reward = -1.0  # Loss
            
            next_state = states[i + 1][0] if i + 1 < len(states) else state
            done = (i == len(states) - 1)
            agent = agent1 if player == 1 else agent2
            agent.remember(state, action, reward, next_state, done)

    def train(self, episodes=1000):
        print(f"Starting training for {episodes} episodes...")
        
        recent_winners = deque(maxlen=100)
        
        for episode in range(episodes):
            winner = self.play_game(self.agent1, self.agent2, train=True)
            recent_winners.append(winner)
            
            loss1 = self.agent1.replay()
            loss2 = self.agent2.replay()
            
            if episode % 100 == 0:
                win_rate_1 = sum(1 for w in recent_winners if w == 1) / len(recent_winners) if recent_winners else 0
                win_rate_2 = sum(1 for w in recent_winners if w == 2) / len(recent_winners) if recent_winners else 0
                
                self.win_rates.append((win_rate_1, win_rate_2))
                
                avg_loss = np.mean([l for l in [loss1, loss2] if l is not None])
                if avg_loss:
                    self.losses.append(avg_loss)
                
                print(f"Episode {episode}: P1 Win Rate: {win_rate_1:.3f}, "
                      f"P2 Win Rate: {win_rate_2:.3f}"
                      f"Epsilon: {self.agent1.epsilon:.3f}")
                
    def evaluate_against_random(self, games=100):
        wins = 0
        losses = 0
        
        for _ in range(games):
            game = self.game_class()
            
            while True:
                if game.player_turn == 2:
                    action = self.agent2.choose_action(game)
                else:
                    valid_moves = self.agent1.get_valid_moves(game)
                    action = random.choice(valid_moves) if valid_moves else None
                
                if action is None:
                    break
                
                game_over = game.play_step(action)
                if game_over:
                    winner = game.check_winner()
                    if winner == 2:
                        wins += 1
                    else:
                        losses += 1
                    break
        
        print(f"Evaluation against random player: Wins: {wins}, Losses: {losses}")
        return wins / games
    
    def plot_training_progress(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        if self.win_rates:
            episodes = range(0, len(self.win_rates) * 100, 100)
            win_rates_p1, win_rates_p2 = zip(*self.win_rates)
            
            ax1.plot(episodes, win_rates_p1, label='Player 1', color='blue')
            ax1.plot(episodes, win_rates_p2, label='Player 2', color='red')
            ax1.set_title('Win Rates Over Time')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Win Rate')
            ax1.legend()
            ax1.grid(True)
        
        if self.losses:
            ax2.plot(range(0, len(self.losses) * 100, 100), self.losses)
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        episodes_range = range(1000)
        epsilon_values = [max(0.01, 1.0 * (0.995 ** ep)) for ep in episodes_range]
        ax3.plot(episodes_range, epsilon_values)
        ax3.set_title('Epsilon Decay (Exploration Rate)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True)
        
        random_performance = self.evaluate_against_random()
        ax4.bar(['Random Baseline', 'Trained Agent'], [1-random_performance, random_performance])
        ax4.set_title('Performance vs Random Player')
        ax4.set_ylabel('Win Rate')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_testing_progress(self, games=100):
        random_performance = self.evaluate_against_random(games)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Random Player', 'Trained Agent'], [1 - random_performance, random_performance])
        ax.set_title('Performance vs Random Player')
        ax.set_ylabel('Win Rate')
        ax.grid(True)
        plt.show()

    def save_model(self, filepath):
        torch.save({
            'agent1_state_dict': self.agent1.net.state_dict(),
            'agent2_state_dict': self.agent2.net.state_dict(),
            'agent1_optimizer': self.agent1.optimizer.state_dict(),
            'agent2_optimizer': self.agent2.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.agent1.net.load_state_dict(checkpoint['agent1_state_dict'])
        self.agent2.net.load_state_dict(checkpoint['agent2_state_dict'])
        self.agent1.optimizer.load_state_dict(checkpoint['agent1_optimizer'])
        self.agent2.optimizer.load_state_dict(checkpoint['agent2_optimizer'])
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    from game import HexapawnGame
    trainer = HexaPawnTrainer(HexapawnGame)


    #Training
    load = input("Do you want to load a pre-trained model? (y/n): ")
    if load.lower() == 'y':
        try:
            trainer.load_model("hexapawn_model.pth")
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting training from scratch.")
    
    trainer.train(episodes=200)
    print("Training completed.")
    save = input("Do you want to save the model? (y/n): ")
    if save.lower() == 'y':
        trainer.save_model("hexapawn_model.pth")
    
    trainer.plot_training_progress()

    #Testing
    try:
        trainer.load_model("hexapawn_model.pth")
        print("Model loaded successfully.")
        trainer.plot_testing_progress(games=100)
    except FileNotFoundError:
        print("No pre-trained model found. Starting training from scratch.")
    


    
    


