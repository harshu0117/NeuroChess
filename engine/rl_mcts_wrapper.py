import sys
import chess
import torch
from engine.mcts import MCTSEngine
from engine.encoder import BoardEncoder
from neural.model import ChessResNet

def main():
    model = ChessResNet()
    encoder = BoardEncoder()
    
    try:
        model.load_state_dict(torch.load("models/rl_finetuned.pt", map_location='cpu'))
    except:
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
            print("id name RL_MCTS_ChessRL")
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
            iterations = 400
            for i, part in enumerate(parts):
                if part == "nodes":
                    iterations = int(parts[i+1])
            
            move = engine.search(board, iterations=iterations)
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
