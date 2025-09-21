import pygame
import sys
from utils import board_to_key, valid_moves, valid_pos
from agent import BruteForceAgent

pygame.init()

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 3, 3
SQUARE_SIZE = WIDTH // COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (185, 185, 185, 0.1)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

PLAYER1_COLOR = BLUE
PLAYER2_COLOR = RED

class HexapawnGame:
    def __init__(self,game=None):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hexapawn")
        self.font = pygame.font.Font(None, 50)
        self.valid_moves = valid_moves
        self.valid_pos = valid_pos
        self.board_to_key = board_to_key
        if game:
            self.board = game.board
            self.selected = game.selected
            self.player_turn = game.player_turn
        else:
            self.reset()

    def reset(self, delay=500):
        self.board = [
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
        ]
        self.selected = None
        self.player_turn = 1
        self.winner = None
        self.draw_board()
        self.delay = delay
        

    def draw_board(self):
        self.screen.fill(WHITE)
        for row in range(ROWS):
            for col in range(COLS):
                
                pygame.draw.rect(
                    self.screen, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2
                )
                
                if self.board[row][col] == 1:
                    pygame.draw.circle(
                        self.screen,
                        PLAYER1_COLOR,
                        (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                        SQUARE_SIZE // 3,
                    )
                
                elif self.board[row][col] == 2:
                    pygame.draw.circle(
                        self.screen,
                        PLAYER2_COLOR,
                        (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                        SQUARE_SIZE // 3,
                    )


    def move_pawn(self, start, end,board=None):
        if board is None:
            board = self.board
        sr, sc = start
        er, ec = end
        board[er][ec] = board[sr][sc]
        board[sr][sc] = 0

    def check_winner(self):
        for col in range(COLS):
            if self.board[0][col] == 1:
                return 1
            if self.board[ROWS - 1][col] == 2:
                return 2

        p1_moves = sum(
            len(self.valid_moves(self.board,r, c, 1)) for r in range(ROWS) for c in range(COLS) if self.board[r][c] == 1
        )
        p2_moves = sum(
            len(self.valid_moves(self.board,r, c, 2)) for r in range(ROWS) for c in range(COLS) if self.board[r][c] == 2
        )
        valid_moves = [p1_moves,p2_moves]
        if valid_moves[~(self.player_turn-1)] == 0:
            return self.player_turn

        return None

    def draw_winner(self, winner):
        self.winner = winner
        text = self.font.render(f"Player {winner} Wins!", True, BLACK)
        self.screen.fill(WHITE)
        self.screen.blit(
            text,
            (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2),
        )
        pygame.display.flip()
        pygame.time.delay(self.delay)
    
    def invalid_move(self):
        text = self.font.render(f"Invalid Move!", True, RED)
        self.screen.fill(WHITE)
        self.screen.blit(
            text,
            (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2),
        )
        pygame.display.flip()
        pygame.time.delay(self.delay)


    def play_step(self, action):
        if action:
            sr, sc, er, ec = action
            if (er, ec) in self.valid_moves(self.board,sr, sc, self.player_turn):
                self.move_pawn((sr, sc), (er, ec))
                winner = self.check_winner()
                if winner:
                    self.draw_winner(winner)
                    return False
                self.player_turn = 3 - self.player_turn
            
            return True
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col, row = x // SQUARE_SIZE, y // SQUARE_SIZE

                    if self.selected:
                        
                        sr, sc = self.selected
                        if (row, col) in self.valid_moves(self.board,sr, sc, self.player_turn):
                            self.move_pawn(self.selected, (row, col))
                            winner = self.check_winner()
                            if winner:
                                self.draw_winner(winner)
                                return False
                            self.player_turn = 3 - self.player_turn
                        else:
                            self.invalid_move()
                        self.selected = None
                    else:
                        
                        if self.board[row][col] == self.player_turn:
                            self.selected = (row, col)
            return True
    


    def run(self,agent=None):
        running = True
        action = None
        while running:
            if agent:
                action = agent.get_action(self.board, self.player_turn)
            running = self.play_step(action)
            if not running:
                break
            self.draw_board()
            pygame.time.delay(self.delay)
            pygame.display.flip()

        
        


def AIvsAI():
    game = HexapawnGame()
    agent = BruteForceAgent()
    agent.define(valid_moves, valid_pos, board_to_key)
    n = 10000
    player_1, player_2 = 0, 0
    while n > 0:
        game.reset(2)
        running = True
        action = None
        while running:
            if agent:
                action = agent.get_action(game.board, game.player_turn)
            running = game.play_step(action)
            if not running:
                break
            game.draw_board()
            pygame.time.delay(game.delay)
            pygame.display.flip()
        winner = game.winner
        if winner == 1:
            player_1 += 1
        elif winner == 2:
            player_2 += 1
        n -= 1
    pygame.quit()

    print(f"Player 1 wins: {player_1} times")
    print(f"Player 2 wins: {player_2} times")

def HumanVsAI():
    game = HexapawnGame()
    agent = BruteForceAgent()
    agent.define(valid_moves, valid_pos, board_to_key)
    game.reset(10)
    running = True
    while running:
        action = None
        if agent and game.player_turn == 2:
            action = agent.get_action(game.board, game.player_turn)
        running = game.play_step(action)
        if not running:
            break
        game.draw_board()
        pygame.time.delay(game.delay)
        pygame.display.flip()
    pygame.quit()

def HumanVsHuman():
    game = HexapawnGame()
    game.reset(10)
    running = True
    action = None
    while running:
        running = game.play_step(action)
        if not running:
            break
        game.draw_board()
        pygame.time.delay(game.delay)
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    # AIvsAI()
    HumanVsAI()
    # HumanVsHuman()