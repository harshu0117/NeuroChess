import subprocess
import sys
import argparse
import os
import time
import json
import psutil
import threading
import torch
import math
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

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

def calculate_elo_diff(win_rate):
    if win_rate <= 0: return -1000
    if win_rate >= 1: return 1000
    return -400 * math.log10(1.0 / win_rate - 1.0)

def main():
    parser = argparse.ArgumentParser(description="Unified Research Experiment Runner with Telemetry")
    parser.add_argument("--mode", choices=["all", "train", "benchmark", "rl"], default="all")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--time", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Pre-flight Check
    DATA_PATH = 'data/maia_chess.csv'
    if args.mode in ["all", "train"] and not os.path.exists(DATA_PATH):
        print(f"\n[!] ERROR: Dataset not found at {DATA_PATH}")
        print("[!] Please run 'python -m data.fetch_data' before starting the experiment.")
        sys.exit(1)

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

    # Phase 1: Training (Supervised)
    if args.mode in ["all", "train"]:
        if os.path.exists("models/supervised_base.pt"):
            print("\n[!] Supervised model exists. Skipping training.")
        else:
            master_log["experiments"]["supervised_training"] = run_command_with_telemetry(
                [sys.executable, "-m", "neural.train"],
                "Supervised Training (2M Games)"
            )

    # Phase 2: RL Training
    if args.mode in ["all", "rl"]:
        if os.path.exists("models/rl_finetuned.pt"):
            print("\n[!] RL model exists. Skipping RL training.")
        elif os.path.exists("rl/self_play.py"):
            master_log["experiments"]["rl_training"] = run_command_with_telemetry(
                [sys.executable, "-m", "rl.self_play"],
                "Reinforcement Learning (Self-Play)"
            )

    # Phase 3: Benchmarking (The Tournament)
    if args.mode in ["all", "benchmark"]:
        # Profiles:
        # A: classical
        # B: neural (supervised_base.pt)
        # C: rl (rl_finetuned.pt)
        # D: hybrid (supervised_base.pt + classical)
        
        # Test everyone against the Classical Baseline to establish Elo
        opponents = ["neural", "rl", "hybrid"]
        
        for eng in opponents:
            name = f"benchmark_{eng}_vs_classical"
            pgn_file = f"reports/{name}.pgn"
            
            # Check if models exist for neural/rl
            if eng == "neural" and not os.path.exists("models/supervised_base.pt"): continue
            if eng == "rl" and not os.path.exists("models/rl_finetuned.pt"): continue
            
            res = run_command_with_telemetry(
                [
                    sys.executable, "-m", "benchmark.professional_tester",
                    "--engine", eng,
                    "--base", "classical",
                    "--games", str(args.games),
                    "--concurrency", str(args.concurrency),
                    "--time", str(args.time),
                    "--pgn", pgn_file
                ],
                f"Establishing Elo: {eng} vs classical"
            )
            master_log["experiments"][name] = res
            
            # Estimate Elo relative to classical (1500)
            # In a real OpenBench, this is more complex, but for research:
            # We'll parse the last line of the output if we were capturing it, 
            # or just calculate it if we had the W/L/D here.
            # Since run_command prints to sys.stdout, let's assume we need to parse.
            # For now, let's add a placeholder and explain how to read the PGNs.
            print(f">>> {eng} vs classical completed. Check {pgn_file} for Elo analysis.")

    # Save complete telemetry and results
    with open("reports/complete_research_data.json", "w") as f:
        json.dump(master_log, f, indent=4)
    
    print("\n" + "="*60)
    print(" ALL EXPERIMENTS COMPLETE")
    print(" 1. Classical: 1500 Elo (Baseline)")
    print(" 2. Neural:    Check reports/benchmark_neural_vs_classical.pgn")
    print(" 3. RL:        Check reports/benchmark_rl_vs_classical.pgn")
    print(" 4. Hybrid:    Check reports/benchmark_hybrid_vs_classical.pgn")
    print("="*60)

if __name__ == "__main__":
    main()
