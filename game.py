import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 3, 3
SQUARE_SIZE = WIDTH // COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (185, 185, 185, 0.1)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BROWN = (195, 176, 145)
GREEN = (120, 150, 120)

class HexapawnGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("HexaPawn")
        self.font = pygame.font.Font(None, 50)
        icon = pygame.image.load('./assets/icon.png')
        pygame.display.set_icon(icon)

        self.pawn1_img = pygame.image.load('./assets/white_pawn.png').convert_alpha()
        self.pawn2_img = pygame.image.load('./assets/black_pawn.png').convert_alpha()
        img_size = (int(SQUARE_SIZE * 0.8), int(SQUARE_SIZE * 0.8))
        self.pawn1_img = pygame.transform.scale(self.pawn1_img, img_size)
        self.pawn2_img = pygame.transform.scale(self.pawn2_img, img_size)

        self.board = [
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
        ]
        self.selected = None
        self.player_turn = 1

    def draw_board(self):
        offset = (SQUARE_SIZE - self.pawn1_img.get_width()) // 2

        for row in range(ROWS):
            for col in range(COLS):
                
                pygame.draw.rect(
                    self.screen,
                    BROWN if (row + col) % 2 == 0 else GREEN,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )
                
                if self.board[row][col] == 1:
                    # pygame.draw.circle(
                    #     self.screen,
                    #     PLAYER1_COLOR,
                    #     (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    #     SQUARE_SIZE // 3,
                    # )
                    self.screen.blit(self.pawn1_img, (col * SQUARE_SIZE + offset, row * SQUARE_SIZE + offset))
                
                elif self.board[row][col] == 2:
                    # pygame.draw.circle(
                    #     self.screen,
                    #     PLAYER2_COLOR,
                    #     (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    #     SQUARE_SIZE // 3,
                    # )
                    self.screen.blit(self.pawn2_img, (col * SQUARE_SIZE + offset, row * SQUARE_SIZE + offset))

    def valid_moves(self, row, col, player):
        moves = []
        direction = -1 if player == 1 else 1
        new_row = row + direction

        if 0 <= new_row < ROWS and self.board[new_row][col] == 0:
            moves.append((new_row, col))


        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < ROWS and 0 <= new_col < COLS:
                if self.board[new_row][new_col] != 0 and self.board[new_row][new_col] != player:
                    moves.append((new_row, new_col))
        return moves

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
            len(self.valid_moves(r, c, 1)) for r in range(ROWS) for c in range(COLS) if self.board[r][c] == 1
        )
        p2_moves = sum(
            len(self.valid_moves(r, c, 2)) for r in range(ROWS) for c in range(COLS) if self.board[r][c] == 2
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

    def run(self):
        running = True

        while running:
            self.draw_board()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col, row = x // SQUARE_SIZE, y // SQUARE_SIZE

                    if self.selected:
                        
                        sr, sc = self.selected
                        if (row, col) in self.valid_moves(sr, sc, self.player_turn):
                            self.move_pawn(self.selected, (row, col))
                            winner = self.check_winner()
                            if winner:
                                self.draw_winner(winner)
                                running = False
                            self.player_turn = 3 - self.player_turn
                        else:
                            self.invalid_move()
                        self.selected = None
                    else:
                        
                        if self.board[row][col] == self.player_turn:
                            self.selected = (row, col)

        pygame.quit()
        sys.exit()
        


if __name__ == "__main__":
    game = HexapawnGame()
    game.run()