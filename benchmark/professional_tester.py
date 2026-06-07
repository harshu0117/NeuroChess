import os
import subprocess
import sys
import argparse
import json
import math
import requests
import zipfile
import shutil
import time
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

def download_tools():
    """
    Downloads Stockfish and Cutechess-CLI if they aren't present.
    """
    if not os.path.exists("./stockfish"):
        print("📥 Downloading Stockfish...")
        sf_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar"
        os.system(f"wget -q {sf_url} -O stockfish.tar && tar -xf stockfish.tar && mv stockfish-ubuntu-x86-64-avx2 stockfish && chmod +x stockfish")

    if not os.path.exists("./cutechess-cli"):
        print("📥 Downloading Cutechess-CLI...")
        cc_url = "https://github.com/cutechess/cutechess/releases/download/1.3.1/cutechess-1.3.1-linux64.tar.gz"
        os.system(f"wget -q {cc_url} -O cutechess.tar.gz && tar -xzf cutechess.tar.gz && mv cutechess-1.3.1-linux64/cutechess-cli . && chmod +x cutechess-cli")

def run_match(engine_path, sf_path, sf_elo, games=200):
    """
    Runs a match with telemetry monitoring.
    """
    report_file = f"reports/match_sf_{sf_elo}.txt"
    print(f"\n{'='*60}")
    print(f" ⚔️ MATCH: ResNetEngine vs Stockfish {sf_elo}")
    print(f" GAMES: {games}")
    print(f"{'='*60}")
    
    monitor = ResourceMonitor()
    monitor.start()
    
    start_time = time.time()
    
    cmd = [
        "./cutechess-cli",
        "-engine", f"name=ResNetEngine", f"cmd={sys.executable}", f"arg=engine/uci_wrapper.py",
        "-engine", f"name=Stockfish_{sf_elo}", f"cmd={sf_path}", f"option.UCI_LimitStrength=true", f"option.UCI_Elo={sf_elo}",
        "-each", "proto=uci", "tc=0.1",
        "-games", str(games),
        "-rounds", "1",
        "-repeat",
        "-concurrency", "4",
        "-pgn", f"reports/sf_{sf_elo}.pgn"
    ]
    
    with open(report_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        process.wait()
    
    end_time = time.time()
    monitor.stop()
    monitor.join()
    
    duration = end_time - start_time
    
    # Calculate telemetry averages
    if monitor.history:
        avg_cpu = sum(s["cpu_percent"] for s in monitor.history) / len(monitor.history)
        avg_mem = sum(s["memory_used_gb"] for s in monitor.history) / len(monitor.history)
        max_gpu = max([s.get("gpu_percent", 0) for s in monitor.history]) if HAS_NVML and monitor.handle else 0
    else:
        avg_cpu, avg_mem, max_gpu = 0, 0, 0

    print(f"\n 📊 TELEMETRY for SF_{sf_elo}:")
    print(f" - Duration: {duration:.2f}s")
    print(f" - Avg CPU: {avg_cpu:.1f}%")
    print(f" - Avg RAM: {avg_mem:.2f} GB")
    if HAS_NVML: print(f" - Max GPU: {max_gpu:.1f}%")
    
    results = parse_results(report_file)
    if results:
        results["telemetry"] = {
            "duration": duration,
            "avg_cpu": avg_cpu,
            "avg_ram": avg_mem,
            "max_gpu": max_gpu
        }
    return results

def parse_results(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    import re
    match = re.search(r"Finished match between .*?: (\d+)-(\d+)-(\d+)", content)
    if match:
        w, l, d = map(int, match.groups())
        return {"wins": w, "losses": l, "draws": d, "total": w+l+d}
    return None

def calculate_elo(sf_elo, wins, losses, draws):
    total = wins + losses + draws
    if total == 0: return 0
    win_rate = (wins + 0.5 * draws) / total
    if win_rate <= 0: return sf_elo - 800
    if win_rate >= 1: return sf_elo + 800
    elo_diff = -400 * math.log10(1.0 / win_rate - 1.0)
    return sf_elo + elo_diff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-level", type=int, default=200)
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)
    download_tools()
    
    levels = [1200, 1500, 1800, 2000, 2200]
    final_results = {}
    
    for elo in levels:
        res = run_match(
            engine_path="engine/uci_wrapper.py",
            sf_path="./stockfish",
            sf_elo=elo,
            games=args.games_per_level
        )
        if res:
            perf_elo = calculate_elo(elo, res['wins'], res['losses'], res['draws'])
            res['performance_elo'] = perf_elo
            final_results[elo] = res
            print(f"✅ Performance against SF {elo}: {perf_elo:.0f} Elo")

    if final_results:
        avg_elo = sum(r['performance_elo'] for r in final_results.values()) / len(final_results)
        print("\n" + "="*40)
        print(f"🏆 FINAL ESTIMATED ELO: {avg_elo:.0f}")
        print("="*40)
        
        with open("reports/professional_benchmark.json", "w") as f:
            json.dump({
                "results": final_results, 
                "final_estimate": avg_elo,
                "timestamp": time.ctime()
            }, f, indent=4)

if __name__ == "__main__":
    main()
