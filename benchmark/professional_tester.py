import os
import subprocess
import sys
import argparse
import json
import math
import requests
import zipfile
import shutil

def download_tools():
    """
    Downloads Stockfish and Cutechess-CLI if they aren't present.
    Note: Paths are optimized for Linux/Kaggle environments.
    """
    # 1. Download Stockfish (Linux version for Kaggle)
    if not os.path.exists("./stockfish"):
        print("📥 Downloading Stockfish...")
        sf_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar"
        # Simplification for script: assuming wget/curl is available in Kaggle
        os.system(f"wget -q {sf_url} -O stockfish.tar && tar -xf stockfish.tar && mv stockfish-ubuntu-x86-64-avx2 stockfish && chmod +x stockfish")

    # 2. Download Cutechess-CLI
    if not os.path.exists("./cutechess-cli"):
        print("📥 Downloading Cutechess-CLI...")
        # URL for a portable Linux x64 build
        cc_url = "https://github.com/cutechess/cutechess/releases/download/1.3.1/cutechess-1.3.1-linux64.tar.gz"
        os.system(f"wget -q {cc_url} -O cutechess.tar.gz && tar -xzf cutechess.tar.gz && mv cutechess-1.3.1-linux64/cutechess-cli . && chmod +x cutechess-cli")

def run_match(engine_path, sf_path, sf_elo, games=200):
    """
    Runs a match between our engine and Stockfish at a specific Elo.
    """
    report_file = f"reports/match_sf_{sf_elo}.txt"
    print(f"⚔️ Starting match against Stockfish {sf_elo} Elo ({games} games)...")
    
    # Construct cutechess-cli command
    # Using internal Stockfish UCI 'Skill Level' or 'UCI_Elo' if supported
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
    
    return parse_results(report_file)

def parse_results(file_path):
    """
    Extracts W/L/D from cutechess output.
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    # Look for "Finished match" line
    # Finished match between ResNetEngine and Stockfish_1200: 45-140-15 [0.262] 200
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

    # Aggregate Final Elo Estimate
    if final_results:
        avg_elo = sum(r['performance_elo'] for r in final_results.values()) / len(final_results)
        print("\n" + "="*40)
        print(f"🏆 FINAL ESTIMATED ELO: {avg_elo:.0f}")
        print("="*40)
        
        with open("reports/professional_benchmark.json", "w") as f:
            json.dump({"results": final_results, "final_estimate": avg_elo}, f, indent=4)

if __name__ == "__main__":
    main()
