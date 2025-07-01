ROWS, COLS = 3, 3

def board_to_key(board):
    return ''.join(str(cell) for row in board for cell in row)

def valid_pos(game=None, gameinfo=None):
    if game:
        board = game.board
        player = game.player_turn
    elif gameinfo:
        board, player = gameinfo
    pos = []
    for x in range(3):
        for y in range(3):
            if board[x][y] == player:
                pos.append((x,y))
    return pos

def valid_moves(board, row, col, player):
    moves = []
    direction = -1 if player == 1 else 1
    new_row = row + direction

    if 0 <= new_row < ROWS and board[new_row][col] == 0:
        moves.append((new_row, col))


    for dc in [-1, 1]:
        new_col = col + dc
        if 0 <= new_row < ROWS and 0 <= new_col < COLS:
            if board[new_row][new_col] != 0 and board[new_row][new_col] != player:
                moves.append((new_row, new_col))
    return moves