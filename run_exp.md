# Chess Research Experiment Notebook (V3 - Professional Benchmark)

This notebook includes the professional benchmarking suite using UCI, Stockfish, and Cutechess.

## Cell 1: Setup and Initialization
```python
# Pull latest code
!git pull

# Install dependencies
!pip install -r requirements.txt

import os
import subprocess
import sys
import time
import json
import psutil
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import numpy as np
import chess
from tqdm.notebook import tqdm

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Setup Complete.")
```

## Cell 2: Phase 1 - Supervised Training
```python
# Import native training logic
sys.path.append(os.getcwd())
from neural.model import ChessResNet
from neural.dataset import ChessDataset

def train_supervised(csv_path, model_save_path, epochs=12, batch_size=2048, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = ChessResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for states, policy_targets, value_targets in pbar:
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            p_out, v_out = model(states)
            p_loss = policy_criterion(p_out, policy_targets)
            v_loss = value_criterion(v_out.view(-1), value_targets)
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), model_save_path)

DATA_PATH = 'data/maia_chess.csv'
MODEL_PATH = 'models/supervised_base.pt'

if not os.path.exists(DATA_PATH):
    !python -m data.fetch_data

train_supervised(DATA_PATH, MODEL_PATH)
```

## Cell 3: Phase 2 - RL Fine-Tuning
```python
from rl.self_play import run_rl_finetuning
# Run 200 games of self-play with the perspective-aware encoder
run_rl_finetuning(num_games=200, train_frequency=10)
```

## Cell 4: Phase 3 - Professional Elo Benchmarking
```python
# This cell runs the new professional tester which:
# 1. Downloads Stockfish and Cutechess-CLI
# 2. Wraps our model as a UCI engine (engine/uci_wrapper.py)
# 3. Runs 1000 games (200 per Stockfish level: 1200, 1500, 1800, 2000, 2200)
# 4. Calculates a standardized Elo rating
!python benchmark/professional_tester.py --games-per-level 200
```

## Cell 5: Results Analysis
```python
import json
if os.path.exists("reports/professional_benchmark.json"):
    with open("reports/professional_benchmark.json", "r") as f:
        data = json.load(f)
    print("="*40)
    print(f"🏆 FINAL ESTIMATED ELO: {data['final_estimate']:.0f}")
    print("="*40)
    
    for elo, res in data['results'].items():
        print(f"Vs Stockfish {elo}: {res['wins']}W - {res['losses']}L - {res['draws']}D (Perf: {res['performance_elo']:.0f})")
```
