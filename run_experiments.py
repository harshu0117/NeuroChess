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
import requests

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# --- TELEMETRY ---

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
        max_gpu = max([s.get("gpu_percent", 0) for s in monitor.history]) if HAS_NVML and monitor.handle else 0
    else:
        avg_cpu, avg_mem, max_gpu = 0, 0, 0

    print(f"\n TELEMETRY for {description}:")
    print(f" - Duration: {duration:.2f}s")
    print(f" - Avg CPU: {avg_cpu:.1f}%")
    print(f" - Avg RAM: {avg_mem:.2f} GB")
    if HAS_NVML: print(f" - Max GPU: {max_gpu:.1f}%")
    
    return {
        "duration": duration,
        "avg_cpu": avg_cpu,
        "avg_ram": avg_mem,
        "max_gpu": max_gpu
    }

# --- AUTOMATION TOOLS ---

def download_benchmarking_tools():
    """Download Stockfish and Cutechess-CLI for professional benchmarking."""
    if not os.path.exists("./stockfish"):
        print("📥 Downloading Stockfish...")
        sf_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar"
        os.system(f"wget -q {sf_url} -O stockfish.tar && tar -xf stockfish.tar && mv stockfish-ubuntu-x86-64-avx2 stockfish && chmod +x stockfish")

    if not os.path.exists("./cutechess-cli"):
        print("📥 Downloading Cutechess-CLI...")
        cc_url = "https://github.com/cutechess/cutechess/releases/download/1.3.1/cutechess-1.3.1-linux64.tar.gz"
        os.system(f"wget -q {cc_url} -O cutechess.tar.gz && tar -xzf cutechess.tar.gz && mv cutechess-1.3.1-linux64/cutechess-cli . && chmod +x cutechess-cli")

# --- MAIN EXPERIMENT LOOP ---

def main():
    parser = argparse.ArgumentParser(description="Unified Chess Research Pipeline")
    parser.add_argument("--mode", choices=["all", "train", "rl", "benchmark", "fetch"], default="all")
    parser.add_argument("--games-per-level", type=int, default=200)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    master_log = {
        "hardware_info": {
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        },
        "experiments": {}
    }

    # 1. Fetch Data
    if args.mode in ["all", "fetch"]:
        DATA_PATH = 'data/maia_chess.csv'
        if not os.path.exists(DATA_PATH):
            run_command_with_telemetry([sys.executable, "-m", "data.fetch_data"], "Fetching Training Data")

    # 2. Supervised Training
    if args.mode in ["all", "train"]:
        if os.path.exists("models/supervised_base.pt"):
            print("\n[!] Supervised model exists. Skipping.")
        else:
            master_log["experiments"]["supervised_training"] = run_command_with_telemetry(
                [sys.executable, "-m", "neural.train"],
                "Phase 1: Supervised Training"
            )

    # 3. RL Fine-Tuning
    if args.mode in ["all", "rl"]:
        if os.path.exists("models/rl_finetuned.pt"):
            print("\n[!] RL model exists. Skipping.")
        else:
            master_log["experiments"]["rl_training"] = run_command_with_telemetry(
                [sys.executable, "-m", "rl.self_play"],
                "Phase 2: Reinforcement Learning"
            )

    # 4. Professional Benchmarking
    if args.mode in ["all", "benchmark"]:
        download_benchmarking_tools()
        master_log["experiments"]["professional_benchmark"] = run_command_with_telemetry(
            [sys.executable, "benchmark/professional_tester.py", "--games-per-level", str(args.games_per_level)],
            "Phase 3: Professional Elo Benchmarking"
        )
        
        # Load the detailed Elo report if it exists
        if os.path.exists("reports/professional_benchmark.json"):
            with open("reports/professional_benchmark.json", "r") as f:
                elo_data = json.load(f)
            master_log["elo_results"] = elo_data

    # Save final unified report
    with open("reports/unified_experiment_report.json", "w") as f:
        json.dump(master_log, f, indent=4)
    
    print("\n" + "="*60)
    print(" 🏁 ALL PHASES COMPLETE")
    if "elo_results" in master_log:
        print(f" 🏆 Estimated Elo: {master_log['elo_results']['final_estimate']:.0f}")
    print(" 📂 Report saved to: reports/unified_experiment_report.json")
    print("="*60)

if __name__ == "__main__":
    main()
