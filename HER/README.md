# HER (Hexapawn Educable Robot)

This folder contains a punishment-based learning implementation for Hexapawn.

---

## Algorithm Overview

1. Start with an initial set of known moves for each game state
2. Play games using random move selection from known states
3. When the agent loses, "punish" it by removing only the last move that caused the loss
4. Gradually eliminate bad moves through repeated gameplay
5. Build a refined strategy table of winning moves

### Key Features
- **Learning from failure**: Moves are eliminated only when they directly lead to loss
- **Incremental improvement**: Win rate increases as bad moves are removed
- **Memory efficient**: Stores only explored states and their valid moves
- **Single agent focus**: Primarily optimizes Player 2's strategy
- **No preprocessing**: Learns through actual gameplay experience

---

## Files

- **`train.py`**: Main training script implementing punishment-based learning
- **`agent.py`**: Agent implementation using learned move tables
- **`game.py`**: Game environment for Hexapawn
- **`utils.py`**: Utility functions for board manipulation and validation
- **`states.txt`**: Initial move set for all reachable Hexapawn states

---

## Training Instructions

1. **Prepare initial states**:
   - Ensure `states.txt` contains all valid moves for each state
   - This file is typically generated from the brute force method

2. **Execute the training script**:
```bash
python train.py
```

3. **Training process**:
   - Loads initial known moves from `states.txt`
   - Plays episodes where Player 2 uses learned moves
   - Player 1 plays optimally (random valid moves)
   - When Player 2 loses: deletes the last move from that state
   - Tracks wins and losses over all episodes
   - Saves refined strategy to JSON file

4. **Output file**:
   - `states.json`: Refined move table for Player 2 after training
   - Shows moves that survived the learning process (good moves)

### Training Parameters

The training uses these default settings:
- **Board size**: 3×3 grid
- **Episodes**: 100 (default, configurable)
- **Initial states**: Loaded from `states.txt`
- **Learning rate**: 1 bad move deleted per loss

## Usage 

After training, you can run the game using the learned strategy:
```bash
python game.py
```

---

## Comparison with Brute Force

| Aspect | Trial-and-Error | Brute Force |
|--------|---|---|
| **Approach** | Learning through play | Complete tree search |
| **Speed** | Faster | Slower |
| **Memory** | Grows with learning | Fixed large table |
| **Optimality** | Approximate → Optimal | Guaranteed optimal |
| **Adaptability** | Improves over time | Static strategy |

---
