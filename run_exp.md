# Chess Research Experiment Notebook

This file contains the content of `run_experiments.py` and core training logic partitioned into cells for use in a Kaggle or Jupyter notebook.

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
from tqdm.notebook import tqdm

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Hardware Info Logging
master_log = {
    "hardware_info": {
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    },
    "experiments": {},
    "elo_ratings": {"classical": 1500} # Classical is our anchor/baseline
}

print("Setup Complete.")
print(f"Hardware: {master_log['hardware_info']}")
```

## Cell 2: Resource Monitor and Utilities
```python
class ResourceMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.running = True
        self.history = []
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.handle = None
        else:
            self.handle = None

    def run(self):
        while self.running:
            stats = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3)
            }
            if self.handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    stats["gpu_percent"] = util.gpu
                    stats["gpu_mem_percent"] = (mem.used / mem.total) * 100
                except:
                    pass
            self.history.append(stats)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if HAS_NVML:
            try: pynvml.nvmlShutdown()
            except: pass

def run_command_with_telemetry(cmd, description):
    print(f"\n{'='*60}")
    print(f" STARTING: {description}")
    print(f" COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    monitor = ResourceMonitor()
    monitor.start()
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.communicate()
    end_time = time.time()
    
    monitor.stop()
    monitor.join()
    
    duration = end_time - start_time
    
    if monitor.history:
        avg_cpu = sum(s["cpu_percent"] for s in monitor.history) / len(monitor.history)
        avg_mem = sum(s["memory_used_gb"] for s in monitor.history) / len(monitor.history)
        max_gpu = max([s.get("gpu_percent", 0) for s in monitor.history]) if monitor.handle else 0
    else:
        avg_cpu, avg_mem, max_gpu = 0, 0, 0

    print(f"\n TELEMETRY for {description}:")
    print(f" - Duration: {duration:.2f}s")
    print(f" - Avg CPU Usage: {avg_cpu:.1f}%")
    print(f" - Avg RAM Usage: {avg_mem:.2f} GB")
    if monitor.handle:
        print(f" - Max GPU Usage: {max_gpu:.1f}%")
    
    return {
        "duration": duration,
        "avg_cpu": avg_cpu,
        "avg_ram": avg_mem,
        "max_gpu": max_gpu,
        "telemetry_history": monitor.history
    }
```

## Cell 3: Native Training Function
```python
# Import project modules (ensure they are in path)
sys.path.append(os.getcwd())
from neural.model import ChessResNet
from neural.dataset import ChessDataset

def train_supervised(csv_path, model_save_path, epochs=12, batch_size=2048, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize components
    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = ChessResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    model.train()
    
    start_time = time.time()
    monitor = ResourceMonitor()
    monitor.start()

    try:
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for states, policy_targets, value_targets in pbar:
                states = states.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                
                optimizer.zero_grad()
                policy_out, value_out = model(states)
                
                p_loss = policy_criterion(policy_out, policy_targets)
                v_loss = value_criterion(value_out.view(-1), value_targets)
                loss = p_loss + v_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(dataloader):.4f}")
            torch.save(model.state_dict(), model_save_path)
    finally:
        monitor.stop()
        monitor.join()

    end_time = time.time()
    return {
        "duration": end_time - start_time,
        "telemetry_history": monitor.history
    }
```

## Cell 4: Phase 1 - Supervised Training
```python
DATA_PATH = 'data/maia_chess.csv'
MODEL_PATH = 'models/supervised_base.pt'

if os.path.exists(MODEL_PATH):
    print("\n[!] Supervised model exists. Skipping training.")
else:
    if not os.path.exists(DATA_PATH):
        print("Fetching data...")
        !python -m data.fetch_data
    
    master_log["experiments"]["supervised_training"] = train_supervised(
        csv_path=DATA_PATH,
        model_save_path=MODEL_PATH
    )
```

## Cell 5: Phase 2 - RL Training (Self-Play)
```python
if os.path.exists("models/rl_finetuned.pt"):
    print("\n[!] RL model exists. Skipping RL training.")
elif os.path.exists("rl/self_play.py"):
    # Running RL via subprocess as it likely manages its own complex state/loops
    master_log["experiments"]["rl_training"] = run_command_with_telemetry(
        [sys.executable, "-m", "rl.self_play"],
        "Reinforcement Learning (Self-Play)"
    )
```

## Cell 6: Phase 3 - Benchmarking (Tournament)
```python
# Simulation parameters
GAMES = 100
CONCURRENCY = 4
TIME_CONTROL = 0.1

opponents = ["neural", "rl", "hybrid"]

for eng in opponents:
    name = f"benchmark_{eng}_vs_classical"
    pgn_file = f"reports/{name}.pgn"
    
    if eng == "neural" and not os.path.exists("models/supervised_base.pt"): continue
    if eng == "rl" and not os.path.exists("models/rl_finetuned.pt"): continue
    
    res = run_command_with_telemetry(
        [
            sys.executable, "-m", "benchmark.professional_tester",
            "--engine", eng,
            "--base", "classical",
            "--games", str(GAMES),
            "--concurrency", str(CONCURRENCY),
            "--time", str(TIME_CONTROL),
            "--pgn", pgn_file
        ],
        f"Establishing Elo: {eng} vs classical"
    )
    master_log["experiments"][name] = res
```

## Cell 7: Final Reporting
```python
with open("reports/complete_research_data.json", "w") as f:
    json.dump(master_log, f, indent=4)
print("All experiments logged to reports/complete_research_data.json")
```
