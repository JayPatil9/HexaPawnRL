# Brute Force Method

This folder contains the brute force implementation for solving Hexapawn using exhaustive game tree search. The algorithm generates a complete lookup table of optimal moves for every possible game state.

## 🧠 Algorithm Overview

The brute force approach uses **exhaustive recursive search** to:
1. Explore all possible game states from the initial position
2. Track winning and losing paths for both players
3. Generate optimal move tables for each player
4. Create a complete strategy database

### Key Features
- **Complete game tree traversal**: Every possible game state is analyzed
- **Optimal play guarantee**: Generated moves are mathematically optimal
- **Dual player tables**: Separate strategy tables for Player 1 and Player 2
- **Perfect information**: Leverages complete game state visibility

## 📁 Files

- **`train.py`**: Main training script that generates the lookup tables
- **`agent.py`**: Agent implementation using the generated tables
- **`game.py`**: Game environment specific to brute force approach
- **`utils.py`**: Utility functions for board manipulation and validation

## 🚀 Training Instructions

### Running the Training

1. **Execute the training script**:
```bash
python train.py
```

2. **Training process**:
   - Starts from the initial Hexapawn board position
   - Recursively explores all possible game paths
   - Tracks moves that lead to winning outcomes
   - Saves optimal strategies to JSON files

3. **Output files**:
   - `brute_force_table_1.json`: Optimal moves for Player 1
   - `brute_force_table_2.json`: Optimal moves for Player 2

### Training Parameters

The training uses these default settings:
- **Board size**: 3×3 grid
- **Initial setup**: Standard Hexapawn starting position
- **Search depth**: Complete (until game termination)
- **Players**: 2 (alternating turns)

## 🔧 Algorithm Details

### State Representation
- Board states are converted to unique string keys using `board_to_key()`
- Each state maps to a list of winning moves: `(start_row, start_col, end_row, end_col)`

### Search Strategy
```python
def recursive_search(board, player_turn, game_data):
    1. Get current state key
    2. Find all valid positions for current player
    3. For each position, find all valid moves
    4. Apply each move to create new board state
    5. Check if move results in immediate win
    6. If win: save the path to lookup table
    7. If not: recursively search opponent's response
```

### Win Conditions
- **Pawn promotion**: Getting a pawn to the opposite end
- **Capture all**: Eliminating all opponent pawns
- **Blockade**: Opponent has no legal moves

## 📊 Performance Characteristics

### Advantages
- **Perfect play**: Always makes optimal moves
- **Fast lookup**: Immediate move selection during gameplay
- **Complete coverage**: Handles all possible game states

### Limitations
- **Memory intensive**: Stores large lookup tables
- **Preprocessing required**: Must generate tables before play
- **Static strategy**: Cannot adapt or learn from experience
- **Limited scalability**: Exponential growth with board size

<!-- ## 🎯 Usage Example

After training, the generated tables can be used as follows:

```bash

``` -->

---

**Note**: This brute force approach provides the theoretical foundation for optimal Hexapawn play and serves as a benchmark for other AI methods in this project.