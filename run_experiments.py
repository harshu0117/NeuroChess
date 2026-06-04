import subprocess
import sys
import argparse
import os
import time
import json
import psutil
import threading
import torch
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
    
    # Analyze telemetry
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

def main():
    parser = argparse.ArgumentParser(description="Unified Research Experiment Runner with Telemetry")
    parser.add_argument("--mode", choices=["all", "train", "benchmark", "rl"], default="all")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--time", type=float, default=0.1)
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

    # Phase 1: Training
    if args.mode in ["all", "train"]:
        master_log["experiments"]["supervised_training"] = run_command_with_telemetry(
            [sys.executable, "-m", "neural.train"],
            "Supervised Training (2M Games)"
        )

    # Phase 2: RL
    if args.mode in ["all", "rl"]:
        if os.path.exists("rl/self_play.py"):
            master_log["experiments"]["rl_training"] = run_command_with_telemetry(
                [sys.executable, "-m", "rl.self_play"],
                "Reinforcement Learning (Self-Play)"
            )

    # Phase 3: Benchmarking
    if args.mode in ["all", "benchmark"]:
        pairs = [
            ("hybrid", "classical"),
            ("neural", "classical"),
            ("hybrid", "neural"),
        ]

        for engine, base in pairs:
            name = f"benchmark_{engine}_vs_{base}"
            pgn_file = f"reports/{name}.pgn"
            master_log["experiments"][name] = run_command_with_telemetry(
                [
                    sys.executable, "-m", "benchmark.professional_tester",
                    "--engine", engine,
                    "--base", base,
                    "--games", str(args.games),
                    "--concurrency", str(args.concurrency),
                    "--time", str(args.time),
                    "--pgn", pgn_file
                ],
                f"Benchmark: {engine} vs {base}"
            )

    # Save complete telemetry and results
    with open("reports/complete_research_data.json", "w") as f:
        json.dump(master_log, f, indent=4)
    
    print("\n" + "="*60)
    print(" ALL EXPERIMENTS COMPLETE")
    print(" Telemetry and results saved to reports/complete_research_data.json")
    print("="*60)

if __name__ == "__main__":
    main()
