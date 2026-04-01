# HIM (Hexapawn Instructable Matchboxes)

This folder contains a matchbox-style learning implementation for Hexapawn that both rewards successful moves and punishes losing moves.

---

## Algorithm Overview

1. Start with an initial set of valid moves for each reachable game state (from `states.txt`).
2. Play self-play episodes where Player 2 (HIM) uses weighted random selection from its matchboxes and Player 1 plays available moves randomly.
3. When HIM wins: add beads (increase counts) for every move it made during that game so those moves become more likely in the future.
4. When HIM loses: remove/penalize beads (decrease counts, not below 1) for every move it made during that game.
5. Repeat to bias the strategy toward sequences that lead to wins while retaining diversity from earlier rewards.

### Key Features
- **Reward and punish**: Wins add beads to all moves used in that win; losses decrement counts for moves used in that loss.
- **Probability based**: Matchbox counts are used as weights for random.choices; higher-count moves are more likely.
- **Sequence-level learning**: Rewards/punishments apply to entire game histories, not only the final move.
- **Persistent state**: Learned move weights are saved as JSON for reuse.

---

## Files

- **`train.py`**: HIM training implementation.
- **`game.py`**: Game environment used for interactive play or visualization.
- **`utils.py`**: Board utilities and move validators.
- **`states.txt`**: Initial move table for all reachable Hexapawn states (source of initial matchboxes).

---

## Training Instructions

1. Ensure `states.txt` exists in this folder (generated via brute-force enumeration or provided).

2. **Execute the training script**:
```bash
python train.py
```

3`. Training process:
- Loads initial move counts from `states.txt` into `self.game_data`.
- HIM selects moves using weights stored per-state in `self.game_data`.
- On win: every `(state, move)` from the game history gets its count increased (reward).
- On loss: every `(state, move)` from the game history gets its count decreased down to a minimum of 1 (punish).

4. Output file:
- `states.json`: learned move-weight table after training.

---

### Default Parameters
- **Board size**: 3×3
- **Episodes**: configurable; `train.py` defaults to 1000 in the example run
- **Reward amount**: configurable in code; current default increments by 1 per rewarded move
- **Punish amount**: decrements by 1 per punished move, with lower bound 1


---

## Usage 

After training, you can run the game using the learned strategy:
```bash
python game.py
```

---
