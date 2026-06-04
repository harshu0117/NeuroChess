import sys
import chess
import torch
from engine.mcts import MCTSEngine
from engine.encoder import BoardEncoder
from neural.model import ChessResNet

def main():
    # Initialize Engine
    model = ChessResNet()
    encoder = BoardEncoder()
    
    # Load weights if available
    try:
        model.load_state_dict(torch.load("models/supervised_base.pt", map_location='cpu'))
    except:
        # For UCI compliance, we still need to run even if weights aren't there
        pass
    
    engine = MCTSEngine(model, encoder)
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        line = line.strip()
        parts = line.split()
        if not parts:
            continue

        command = parts[0]

        if command == "uci":
            print("id name MCTS_ChessRL")
            print("id author GeminiCLI")
            print("uciok")
            sys.stdout.flush()
        elif command == "isready":
            print("readyok")
            sys.stdout.flush()
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            if parts[1] == "startpos":
                board = chess.Board()
                if "moves" in parts:
                    move_index = parts.index("moves")
                    for move_uci in parts[move_index + 1:]:
                        board.push_uci(move_uci)
            elif parts[1] == "fen":
                fen = " ".join(parts[2:8])
                board = chess.Board(fen)
                if "moves" in parts:
                    move_index = parts.index("moves")
                    for move_uci in parts[move_index + 1:]:
                        board.push_uci(move_uci)
        elif command == "go":
            iterations = None
            time_limit = None
            for i, part in enumerate(parts):
                if part == "nodes":
                    iterations = int(parts[i+1])
                if part == "movetime":
                    time_limit = int(parts[i+1]) / 1000.0
            
            # Default to 0.1s if nothing specified for safety
            if iterations is None and time_limit is None:
                time_limit = 0.1
                    
            move = engine.search(board, iterations=iterations, time_limit=time_limit)
            if move:
                print(f"bestmove {move.uci()}")
            else:
                fallback = list(board.legal_moves)[0] if list(board.legal_moves) else None
                if fallback:
                    print(f"bestmove {fallback.uci()}")
            sys.stdout.flush()
        elif command == "quit":
            break

if __name__ == "__main__":
    main()
