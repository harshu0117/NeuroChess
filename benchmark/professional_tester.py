import chess
import chess.engine
import chess.pgn
import math
import time
import argparse
import sys
import os
import datetime
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Professional Elo calculation with error margins
class EloCalculator:
    @staticmethod
    def get_elo_diff(win_rate: float) -> float:
        if win_rate <= 0: return -1000
        if win_rate >= 1: return 1000
        return -400 * math.log10(1.0 / win_rate - 1.0)

    @staticmethod
    def get_error_margin(wins: int, losses: int, draws: int) -> float:
        total = wins + losses + draws
        if total <= 1: return 0
        
        mu = (wins + 0.5 * draws) / total
        stdev = math.sqrt((wins * (1.0 - mu)**2 + losses * (0.0 - mu)**2 + draws * (0.5 - mu)**2) / total)
        
        # 95% confidence interval
        if mu <= 0 or mu >= 1: return 0
        return 1.96 * stdev / math.sqrt(total) * 400 / (math.log(10) * mu * (1.0 - mu))

class SPRT:
    """Sequential Probability Ratio Test (SPRT) using Wald's method."""
    def __init__(self, elo0: float, elo1: float, alpha: float = 0.05, beta: float = 0.05):
        self.elo0 = elo0
        self.elo1 = elo1
        self.alpha = alpha
        self.beta = beta
        
        # Target probabilities
        self.p0 = 1.0 / (1.0 + 10**(-elo0 / 400.0))
        self.p1 = 1.0 / (1.0 + 10**(-elo1 / 400.0))
        
        # Thresholds
        self.la = math.log(beta / (1.0 - alpha))
        self.lb = math.log((1.0 - beta) / alpha)

    def status(self, wins: int, losses: int, draws: int) -> str:
        total = wins + losses + draws
        if total == 0: return "PENDING"
        
        if wins + losses == 0: return "PENDING"
        
        # Log-likelihood ratio for Bernoulli trials
        # This is an approximation for chess results (W/L/D)
        llr = wins * math.log(self.p1 / self.p0) + losses * math.log((1 - self.p1) / (1 - self.p0))
        
        if llr >= self.lb: return "ACCEPTED (H1)"
        if llr <= self.la: return "REJECTED (H0)"
        return "PENDING"

def play_game(engine1_cmd: List[str], engine2_cmd: List[str], 
              engine1_name: str, engine2_name: str,
              fen: str, time_limit: float, game_id: int) -> Dict:
    """Plays a single game and returns results and PGN string."""
    try:
        e1 = chess.engine.SimpleEngine.popen_uci(engine1_cmd)
        e2 = chess.engine.SimpleEngine.popen_uci(engine2_cmd)
        
        board = chess.Board(fen)
        game = chess.pgn.Game()
        game.headers["Event"] = "Professional Benchmark"
        game.headers["White"] = engine1_name
        game.headers["Black"] = engine2_name
        game.headers["FEN"] = fen
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        
        node = game
        
        while not board.is_game_over():
            current_engine = e1 if board.turn == chess.WHITE else e2
            result = current_engine.play(board, chess.engine.Limit(time=time_limit))
            board.push(result.move)
            node = node.add_main_variation(result.move)
            
        res = board.result()
        game.headers["Result"] = res
        
        e1.quit()
        e2.quit()
        
        return {
            "result": res,
            "pgn": str(game),
            "game_id": game_id
        }
    except Exception as e:
        return {
            "result": "1/2-1/2",
            "pgn": f"Error: {e}",
            "game_id": game_id,
            "error": str(e)
        }

def load_openings(epd_file: Optional[str]) -> List[str]:
    default_openings = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d3 0 2",
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq c3 0 1",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 1",
    ]
    
    if not epd_file or not os.path.exists(epd_file):
        return default_openings
    
    try:
        with open(epd_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            return lines if lines else default_openings
    except:
        return default_openings

def run_benchmark():
    parser = argparse.ArgumentParser(description="Professional Chess Engine Benchmarker")
    parser.add_argument("--engine", choices=["hybrid", "neural", "classical", "rl"], default="hybrid", help="Engine to test")
    parser.add_argument("--base", choices=["hybrid", "neural", "classical", "rl"], default="classical", help="Baseline engine")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel games")
    parser.add_argument("--time", type=float, default=0.1, help="Time limit per move (seconds)")
    parser.add_argument("--epd", type=str, help="Path to EPD opening book")
    parser.add_argument("--pgn", type=str, default="benchmark_results.pgn", help="File to save PGN results")
    parser.add_argument("--sprt", nargs=2, type=float, default=[0, 10], help="SPRT Elo bounds (elo0 elo1)")
    
    args = parser.parse_args()
    
    cmd_map = {
        "hybrid": [sys.executable, "-m", "engine.hybrid_mcts_wrapper"],
        "neural": [sys.executable, "-m", "engine.mcts_wrapper"],
        "classical": [sys.executable, "-m", "engine.alphabeta_wrapper"],
        "rl": [sys.executable, "-m", "engine.rl_mcts_wrapper"]
    }
    
    engine_cmd = cmd_map[args.engine]
    base_cmd = cmd_map[args.base]
    
    openings = load_openings(args.epd)
    sprt = SPRT(args.sprt[0], args.sprt[1])
    
    print(f"\n{'='*60}")
    print(f" PROFESSIONAL BENCHMARK: {args.engine.upper()} vs {args.base.upper()}")
    print(f"{'='*60}")
    print(f" Concurrency: {args.concurrency} | Time: {args.time}s | Games: {args.games}")
    print(f" SPRT Bounds: [{args.sprt[0]}, {args.sprt[1]}]")
    print(f"{'='*60}\n")
    
    wins, losses, draws = 0, 0, 0
    
    with open(args.pgn, "w") as pgn_file:
        with ProcessPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            for i in range(args.games):
                fen = openings[i % len(openings)]
                # Alternate colors: Engine 1 is White in even games
                if i % 2 == 0:
                    futures.append(executor.submit(play_game, engine_cmd, base_cmd, args.engine, args.base, fen, args.time, i))
                else:
                    futures.append(executor.submit(play_game, base_cmd, engine_cmd, args.base, args.engine, fen, args.time, i))
            
            for future in as_completed(futures):
                res_data = future.result()
                game_id = res_data["game_id"]
                res_str = res_data["result"]
                
                # Update statistics
                is_engine_white = (game_id % 2 == 0)
                if res_str == "1-0":
                    if is_engine_white: wins += 1
                    else: losses += 1
                elif res_str == "0-1":
                    if is_engine_white: losses += 1
                    else: wins += 1
                else:
                    draws += 1
                
                # Log PGN
                pgn_file.write(res_data["pgn"] + "\n\n")
                pgn_file.flush()
                
                # Calculate Stats
                total = wins + losses + draws
                win_rate = (wins + 0.5 * draws) / total
                elo_diff = EloCalculator.get_elo_diff(win_rate)
                error = EloCalculator.get_error_margin(wins, losses, draws)
                sprt_status = sprt.status(wins, losses, draws)
                
                # Professional Output Line
                sys.stdout.write(f"\r[{total:3d}/{args.games:3d}] W:{wins:2d} L:{losses:2d} D:{draws:2d} | Elo:{elo_diff:+7.2f} +/-{error:5.2f} | SPRT: {sprt_status}")
                sys.stdout.flush()
                
                if "ACCEPTED" in sprt_status or "REJECTED" in sprt_status:
                    print(f"\n\nStopping: SPRT {sprt_status} triggered.")
                    break

    print(f"\n\n{'='*60}")
    print(f" FINAL REPORT")
    print(f"{'='*60}")
    print(f" Score: {wins} - {losses} - {draws} ({win_rate:.2%})")
    print(f" Elo Difference: {elo_diff:+.2f} +/- {error:.2f}")
    print(f" PGNs saved to: {args.pgn}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_benchmark()
