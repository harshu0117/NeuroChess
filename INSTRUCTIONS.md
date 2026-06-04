# ChessRL: AlphaZero-Inspired Chess Engine on Consumer Hardware
## Master Project Specification & 7-Day Execution Blueprint

This document serves as the master blueprint and complete technical specification for **ChessRL**. It contains all architectural layout patterns, scope constraints, low-level mathematical frameworks, and day-by-day sprint objectives required to execute the project successfully within 7 days on commodity hardware.

---

## 1. Project Vision & Research Framework

### Core Goal
To build, optimize, and empirically analyze a hybrid chess engine that bridges classical brute-force tree search and modern deep reinforcement learning methods under tight consumer-grade hardware constraints.

### The Research Question
> *"How effective are neural-network-guided search methods compared to classical search methods when deployed and evaluated on commodity, consumer-grade hardware?"*

### Core Evaluation Sub-questions:
1. **Search vs. Learning:** Can a small, highly specialized neural network compensate for a $1,000\times$ reduction in Nodes Per Second (NPS) compared to brute-force Alpha-Beta pruning?
2. **Compute vs. Performance:** What are the exact scaling dynamics of Elo rating relative to time constraints, MCTS simulation counts, and neural network parameter size?
3. **Ablation Validity:** How much standalone value do individual modern algorithmic components (Transposition Tables, Iterative Deepening, PUCT-guided MCTS) add under limited execution times?

---

## 2. Resource Constraints & Scope Reductions

### Target Hardware Environment
* **CPU:** Intel i7 (or equivalent consumer multi-core processor)
* **RAM:** 16 GB 
* **GPU:** Integrated Graphics (Intel Iris Xe or similar). *Note: Neural training will be offloaded to a free cloud tier (Google Colab/Kaggle T4 GPU) for 2–6 hours; local search inference must run efficiently on a laptop CPU.*

### Critical Scope Cuts (To Ensure 7-Day Completion)
* **NO Custom Move Generation:** Implementing bug-free bitboards covering castling, en passant, promotions, check validation, and sliding piece pin-detection takes weeks. You must strictly use the pre-built `python-chess` library for board state updates and legal move generation.
* **NO From-Scratch AlphaZero Self-Play:** Pure self-play RL takes millions of games on massive TPU clusters to learn fundamental concepts. You will pivot to **Supervised Behavioral Cloning** on pre-existing chess datasets to build a strong neural network baseline on Day 3. 
* **Micro-Scale RL Only:** A small reinforcement self-play loop (50–100 games) will be built on Day 6 strictly to prove the mathematical correctness of your pipeline, not for massive Elo accumulation.

---

## 3. System Architecture & Directory Tree

Create the workspace layout precisely as defined below:

```text
ChessRL/
├── engine/
│   ├── __init__.py
│   ├── encoder.py         # State tensorization (8x8x13)
│   ├── evaluation.py      # Handcrafted evaluation baseline
│   ├── transposition.py   # Zobrist-hashed cache table
│   ├── alphabeta.py       # Classical Engine (Minimax, Alpha-Beta, ID)
│   └── mcts.py            # Neural Engine (MCTS with PUCT)
├── neural/
│   ├── __init__.py
│   ├── model.py           # Dual-Headed ResNet (Policy & Value)
│   ├── dataset.py         # Lichess/Stockfish CSV parser & DataLoader
│   └── train.py           # Supervised training loop
├── rl/
│   ├── __init__.py
│   └── self_play.py       # Short-run data-generation & fine-tuning loop
├── benchmark/
│   ├── __init__.py
│   ├── elo.py             # Match evaluator
│   ├── experiments.py     # Automated execution & log parser
│   └── results.csv        # Profiling database
├── web/
│   └── streamlit_app.py   # Human interaction & analysis UI
├── models/                # Saved weights (.pt files)
├── notebooks/             # Scratchpads / Colab training source scripts
└── INSTRUCTIONS.md        # Reference specifications


# ChessRL: Technical Specifications & Implementation Guide

## 4. Comprehensive File Specifications

### 4.1 `engine/encoder.py`
* **Objective:** Encodes the `chess.Board` state into an $8 \times 8 \times 13$ binary tensor.
* **Perspective Optimization:** To maximize sample efficiency, the board must always be encoded from the perspective of the **player whose turn it is to move**. If it is Black's turn, mirror the board vertically before tensorizing.
* **Tensor Array Mapping:** Shape `(13, 8, 8)` float32.
  * Channels 0–5: Current player's pieces (Pawn, Knight, Bishop, Rook, Queen, King).
  * Channels 6–11: Opponent player's pieces (Pawn, Knight, Bishop, Rook, Queen, King).
  * Channel 12: Constant plane filled entirely with $1.0$ if White to move, or $0.0$ if Black to move (assists the neural net in parsing orientation-dependent metadata like castling rights).
* **Action Space Mapping:** Flat array of size **4,096** ($64 \times 64$).
  $$\text{Action Index} = (\text{from\_square} \times 64) + \text{to\_square}$$
  *Promotion Handling:* Default all pawn promotion selections to Queens to save policy action space dimensions during constrained computing.

### 4.2 `engine/evaluation.py`
* **Objective:** Handcrafted positional evaluation for Engine A (Classical).
* **Material Values:** `{'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}`.
* **Positional Matrices (Piece-Square Tables):** Provide explicit 64-element arrays penalizing or rewarding piece placement (e.g., knights penalized on corners, pawns rewarded for advancing and holding the center).
* **Perspective Calculation:** $$\text{Static Score} = (\text{White Material} - \text{Black Material}) + (\text{White Positional Bonus} - \text{Black Positional Bonus})$$
  The score must always return relative to White's advantage (positive values favor White, negative values favor Black).

### 4.3 `engine/transposition.py`
* **Objective:** Cache table saving evaluated nodes for Alpha-Beta search.
* **Implementation:** Wrap a Python dictionary with a capacity threshold ($10^6$ entries) utilizing a First-In, First-Out (FIFO) eviction guard.
* **Hashing Key:** Access `chess.polyglot.zobrist_hash(board)` directly from the library to prevent hash collision defects.
* **Payload Dict Structure:**
  ```python
  {
      'depth': int,        # Search depth when stored
      'flag': str,         # 'EXACT', 'LOWERBOUND' (Beta cutoff), 'UPPERBOUND' (Alpha fail-low)
      'value': float,      # Calculated evaluation score
      'best_move': str     # UCI representation string of optimal child branch
  }
  ```

### 4.4 `engine/alphabeta.py`
* **Objective:** Robust classical search loop combining tree search optimizations.
* **Core Structural Requirements:**
  1. **Iterative Deepening Wrapper:** Loop depth from $d=1$ through $d=\text{Target}$. Track execution time limits. Cache and update a global variable `best_move_so_far` at the end of each completed depth iteration to guarantee safe moves during an execution timeout.
  2. **Move Ordering:** Before scanning child branches, sort legal moves:
     * Priority 1: Transposition Table recommendation for the current hash position.
     * Priority 2: Tactical Captures sorted via MVV-LVA (*Most Valuable Victim - Least Valuable Aggressor*).
     * Priority 3: Check delivery and pawn promotions.
  3. **Quiescence Search:** When reaching `depth == 0`, transition to a selective search that strictly parses only legal tactical captures. Halt and evaluate only when the board reaches a stable position (no valid captures remaining). This mitigates the "horizon effect."

### 4.5 `neural/model.py`
* **Objective:** Compact Dual-Headed Residual Network optimized for fast CPU inference.
* **Architecture Pipeline:**
  * **Input Block:** `Conv2d(13 -> 64, kernel=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU`.
  * **Residual Core:** 4 sequential Residual Blocks. Each block contains:
    $$\text{X}_{\text{out}} = \text{ReLU}(\text{X} + \text{BatchNorm}(\text{Conv2d}(\text{ReLU}(\text{BatchNorm}(\text{Conv2d}(\text{X}))))))$$
    Using 64 filters, kernel size 3, padding 1.
  * **Policy Head:** `Conv2d(64 -> 2, kernel=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(128 -> 4096)`. Outputs raw unnormalized logits.
  * **Value Head:** `Conv2d(64 -> 1, kernel=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(1)` $\rightarrow$ `Tanh`. Outputs a positional health score bounded strictly within $[-1.0, 1.0]$.

### 4.6 `engine/mcts.py`
* **Objective:** Monte Carlo Tree Search using neural guidance rather than random rollouts.
* **Node Variables:** $N$ (visit count), $W$ (total action value), $Q$ (mean action value = $W/N$), $P$ (prior probability array from model Policy head).
* **PUCT Selection Formula:** Select child action $a$ maximizing:
  $$\text{argmax}_a \left( Q(s, a) + C_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right)$$
  Set $C_{puct} = 1.4$.
* **The Legality Mask:** When processing the model's 4,096 Policy vector output, assign all illegal move indices to $-\infty$, then calculate the `Softmax` across the remaining valid legal actions.
* **Leaf Evaluation:** When expanding an unvisited leaf node, do not perform random play simulations. Pass the tensorized state directly to your neural network. Use the Value head output ($v$) to update the node metrics, backpropagating the result up the selection path while flipping the sign of $v$ at each level to account for the alternating player perspective.

### 4.7 `neural/dataset.py` & `neural/train.py`
* **Data Core:** Load static chess tabular position datasets (FEN strings paired with high-Elo move selections and definitive game outcomes).
* **Multi-Task Loss Evaluation:**
  $$\text{Loss} = (z - v)^2 - \pi^T \log(p) + c ||\theta||^2$$
  * $(z - v)^2$: Mean Squared Error minimizing value predictions vs actual game output $z \in \{-1, 0, 1\}$.
  * $-\pi^T \log(p)$: Categorical Cross-Entropy tracking policy logits $p$ against one-hot encoded target moves $\pi$.
  * $c ||\theta||^2$: Weight decay set to `1e-4` for L2 regularized structural convergence.

### 4.8 `rl/self_play.py`
* **Objective:** Verification loop proving reinforcement data pipelines function correctly.
* **Loop Mechanics:** Run automated matches where the engine plays against itself. For every move execution, invoke `MCTS` for 50–100 iterations. Record `[state_tensor, search_visit_distribution, None]` steps to a temporary storage stack. Upon game termination (checkmate or draw), assign the true reward value $z \in \{-1, 0, 1\}$ across all saved steps and append them to the `ReplayBuffer`. Run a single batch optimization step using Adam to verify that model parameters successfully adapt.

### 4.9 `benchmark/experiments.py`
* **Objective:** Automated profiling framework.
* **Execution Logic:** Use Python's `subprocess` engine to spin up background local match series via `cutechess-cli`.
* **Sample Configuration Subprocess Command:**
  ```bash
  cutechess-cli -engine cmd=python_alphabeta_bot.py name=Classical -engine cmd=python_mcts_bot.py name=AlphaMini -each proto=uci tc=40/2 -games 100 -pgn results.pgn
  ```
* Save the computed performance logs into `results.csv` tracking: engine configuration types, nodes searched, computed average NPS values, and relative Elo ratings calculated via the `Ordo` evaluation program.

### 4.10 `web/streamlit_app.py`
* **Objective:** Interactive visualization platform.
* **Features:**
  * Render an interactive chess board using SVG strings (`chess.svg.board(board=board)`).
  * Feature configuration toggles to choose whether the user plays against the Classical Alpha-Beta engine or the Neural MCTS engine.
  * Include an analytical sidebar tracking real-time engine telemetry: evaluated nodes count, active search depth, a live gauge displaying the Value head score, and a bar chart detailing the top 5 legal options ordered by MCTS visit distributions ($N$).

---

## 5. The 7-Day Agile Execution Plan

### Day 1: Core Infrastructure & Classical Baseline
* **Goal:** Establish environment, wrap board structures, and build standard lookahead frameworks.
* **Tasks:** Set up directory configurations. Standardize `encoder.py` plane indexing standards. Implement basic `Minimax` and layer in Alpha-Beta Pruning inside `alphabeta.py`.
* **Deliverable:** A functional classical search script making valid legal moves based on Material evaluations.

### Day 2: Classical Enhancements & State Encoding
* **Goal:** Maximize classical engine search depth and finalize the state tensorizer.
* **Tasks:** Write the Iterative Deepening wrapper. Implement the Transposition Table and integrate MVV-LVA Move Ordering. Finalize `encoder.py` verifying that its perspective tensor tracking matches orientations accurately.
* **Deliverable:** An optimized classical engine reaching search depths of 6–8 efficiently on a laptop CPU.

### Day 3: Neural Network Definition & Supervised Seeding
* **Goal:** Define the model structure and launch supervised training.
* **Tasks:** Implement `ChessResNet` inside `model.py` using PyTorch. Build the custom dataset parser. Upload your scripts to Google Colab/Kaggle, attach the dataset, and launch the multi-task loss optimization process overnight.
* **Deliverable:** Trained weights file (`weights/supervised_base.pt`) achieving high validation accuracy for both heads.

### Day 4: Monte Carlo Tree Search Implementation
* **Goal:** Implement the tree framework and hook it up to your neural network.
* **Tasks:** Construct the MCTS Node class. Build selection structures utilizing the PUCT mathematical formula. Integrate the legal move policy mask. Load your trained weights from Day 3 and route leaf expansions directly through the network's value evaluations.
* **Deliverable:** A fully integrated Neural MCTS Engine capable of playing games without random rollouts.

### Day 5: Empirical Benchmarking & Ablation Tests
* **Goal:** Collect all research data and profile engine performance.
* **Tasks:** Write and execute the automated `cutechess_runner.py` framework. Run Experiment 1 (Classical Alpha-Beta vs Neural MCTS) and Experiment 2 (MCTS Simulation Counts Ablation: 50 vs 100 vs 200 vs 400 simulations). Export the runtime metrics, nodes evaluated, and win-rate profiles to `results.csv`.
* **Deliverable:** Complete dataset matrices and charts mapping out performance vs compute trade-offs.

### Day 6: Web Dashboard & RL Pipeline Verification
* **Goal:** Build the interactive interface and complete the reinforcement training code.
* **Tasks:** Code the `streamlit_app.py` dashboard with SVG rendering and explainability analysis panes. Implement `rl/self_play.py` and execute a short 50-game verification run to ensure backpropagation metrics flow correctly through the self-play loop.
* **Deliverable:** A clean UI app and verified, error-free RL fine-tuning loops.

### Day 7: Automated Report Generation
* **Goal:** Author the final research paper document.
* **Tasks:** Compile your statistical performance logs from Day 5. Plot training loss trends, Elo trajectory trends, and NPS vs Elo curves. Write the 8–10 page report under the title: *"Comparative Analysis of Search-Based and Learning-Based Chess Engines on Consumer Hardware"*.
* **Deliverable:** A submission-ready PDF research report and a structured GitHub repository.

---

## 6. Target Milestone Checkpoints

| Performance Metric | Minimum Success Criterion | Elite Success Target |
| :--- | :--- | :--- |
| **Classical Engine Depth** | Depth 5 steady search | Depth 8+ with full TT caching |
| **Neural Search NPS** | 100–500 simulations/sec | 1,000+ simulations/sec on CPU |
| **Model Size** | 2 Residual Blocks (~1M parameters) | 4–6 Residual Blocks (~3M parameters) |
| **Benchmarking Engine Baseline** | Beat Random Player 100% of matches | Match or defeat Stockfish Level 3–4 |
| **Estimated Match Elo** | ~1400 Local Elo Rating | 1800+ Bounded Local Elo Rating |

---

## 7. Immediate Environment Launch Sequence

Execute the following commands inside your shell terminal to activate the miniconda workspace and verify dependency versions before commanding your agent:

```bash
# 1. Env update
chessenv is already activated take it from there okay, 

# 2. Install pinned dependencies 
pip install python-chess==1.10.0 torch==2.2.0 numpy==1.26.4 streamlit==1.31.0 fastapi==0.109.0 uvicorn==0.27.0

# 3. Create file structure framework
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['engine', 'neural', 'rl', 'benchmark', 'web', 'models', 'reports']]"
python -c "import os; [open(f, 'w').close() for f in ['engine/encoder.py', 'engine/evaluation.py', 'engine/transposition.py', 'engine/alphabeta.py', 'engine/mcts.py', 'neural/model.py', 'neural/dataset.py', 'neural/train.py', 'rl/self_play.py', 'benchmark/experiments.py', 'web/streamlit_app.py']]"
```