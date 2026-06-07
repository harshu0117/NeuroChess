# 🏁 How to Run Chess Research on Kaggle

Follow these steps to run the complete experiment, from training to professional Elo benchmarking, in a Kaggle Notebook.

## 1. Create a New Kaggle Notebook
*   Go to [Kaggle](https://www.kaggle.com/) and create a new Python notebook.
*   **Settings:** Ensure **Internet** is turned ON and **GPU T4 x2** (or P100) is enabled for faster training.

## 2. Step-by-Step Execution

### Step 2.1: Clone and Setup
```python
# Clone your repository
!git clone <YOUR_GITHUB_REPO_URL>
%cd chess

# Install requirements
!pip install -r requirements.txt
```

### Step 2.2: Fetch Training Data
```python
# This downloads the Maia Chess dataset (2M games)
!python -m data.fetch_data
```

### Step 2.3: Supervised Training
```python
import os
sys.path.append(os.getcwd())
from neural.train import train

# Run training (optimized for Kaggle GPU)
train(
    csv_path='data/maia_chess.csv', 
    model_save_path='models/supervised_base.pt', 
    epochs=12, 
    batch_size=2048
)
```

### Step 2.4: RL Fine-Tuning (Self-Play)
```python
from rl.self_play import run_rl_finetuning
# Run 200 games of self-play to improve tactical depth
run_rl_finetuning(num_games=200, train_frequency=10)
```

### Step 2.5: Professional Elo Benchmarking
```python
# This cell handles everything automatically:
# 1. Downloads Stockfish and Cutechess-CLI
# 2. Runs 1000 games across 5 Elo levels (1200-2200)
# 3. Monitors hardware telemetry (CPU/RAM/GPU)
# 4. Calculates your final estimated Elo rating
!python benchmark/professional_tester.py --games-per-level 200
```

## 3. Viewing Results

### Check the Final Report
```python
import json
with open("reports/professional_benchmark.json", "r") as f:
    report = json.load(f)

print(f"🏆 FINAL ESTIMATED ELO: {report['final_estimate']:.0f}")
```

### Analyze Hardware Efficiency
The `professional_benchmark.json` file also contains telemetry for every match. You can see how much GPU memory your model used vs. how many games it won!

---
**Tip:** If the notebook crashes or times out, you can skip Step 2.3 and 2.4 if the `.pt` models are already in your `models/` folder. Step 2.5 will pick up wherever you left off.
