import sys
import os
import chess
import torch
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from engine.encoder import BoardEncoder
from engine.mcts import MCTSEngine
from engine.hybrid_mcts import HybridMCTSEngine
from neural.model import ChessResNet

def uci_loop():
    """
    Standard UCI (Universal Chess Interface) loop for the experimental engine.
    This allows the engine to be used by Cutechess and other GUIs.
    """
    encoder = BoardEncoder()
    model = ChessResNet()
    
    # Try to load best model
    if os.path.exists("models/rl_finetuned.pt"):
        model.load_state_dict(torch.load("models/rl_finetuned.pt", map_location='cpu'))
    elif os.path.exists("models/supervised_base.pt"):
        model.load_state_dict(torch.load("models/supervised_base.pt", map_location='cpu'))
    
    # Default to Hybrid for best performance if available, otherwise MCTS
    # For a UCI engine, we'll use Hybrid as it's the strongest variant
    engine = HybridMCTSEngine(model, encoder)
    board = chess.Board()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            command = line.strip()
            
            if command == "uci":
                print("id name ChessResNet-Experimental")
                print("id author GeminiCLI")
                print("uciok")
                sys.stdout.flush()
            
            elif command == "isready":
                print("readyok")
                sys.stdout.flush()
            
            elif command.startswith("position"):
                # position [fen <fenstring> | startpos ]  moves <move1> .... <movei>
                parts = command.split(" ")
                if "startpos" in parts:
                    board = chess.Board()
                elif "fen" in parts:
                    fen_idx = parts.index("fen")
                    fen = " ".join(parts[fen_idx+1 : fen_idx+7])
                    board = chess.Board(fen)
                
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for move_uci in parts[moves_idx+1:]:
                        board.push_uci(move_uci)
            
            elif command.startswith("go"):
                # Simplistic 'go' handling: just search and move
                # In a real UCI engine we'd parse time controls (wtime, btime)
                move = engine.search(board, iterations=1600)
                print(f"bestmove {move.uci()}")
                sys.stdout.flush()
            
            elif command == "quit":
                break
            
            elif command == "ucinewgame":
                board = chess.Board()
                
        except EOFError:
            break
        except Exception as e:
            # Silently handle or log to stderr to avoid breaking UCI protocol
            sys.stderr.write(f"Error: {str(e)}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    uci_loop()
