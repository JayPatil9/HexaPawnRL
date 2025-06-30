import pygame
import sys
from utils import valid_moves

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
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hexapawn")
        self.font = pygame.font.Font(None, 50)
        self.valid_moves = valid_moves
        self.reset()

    def reset(self):
        self.board = [
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
        ]
        self.selected = None
        self.player_turn = 1
        self.draw_board()

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


    def move_pawn(self, start, end):
        sr, sc = start
        er, ec = end
        self.board[er][ec] = self.board[sr][sc]
        self.board[sr][sc] = 0

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
        text = self.font.render(f"Player {winner} Wins!", True, BLACK)
        self.screen.fill(WHITE)
        self.screen.blit(
            text,
            (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2),
        )
        pygame.display.flip()
        pygame.time.delay(3000)
    
    def invalid_move(self):
        text = self.font.render(f"Invalid Move!", True, RED)
        self.screen.fill(WHITE)
        self.screen.blit(
            text,
            (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2),
        )
        pygame.display.flip()
        pygame.time.delay(500)


    def play_step(self, action):
        if action:
            sr, sc, er, ec = action
            if (er, ec) in self.valid_moves(self.board,sr, sc, self.player_turn):
                self.move_pawn((sr, sc), (er, ec))
                winner = self.check_winner()
                if winner:
                    return winner
                self.player_turn = 3 - self.player_turn
            
            return None
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


    def run(self):
        running = True

        while running:
            running = self.play_step(None)
            if not running:
                break
            self.draw_board()
            pygame.display.flip()

        pygame.quit()
        sys.exit()
        


if __name__ == "__main__":
    game = HexapawnGame()
    game.run()