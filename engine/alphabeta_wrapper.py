import sys
import chess
from engine.alphabeta import AlphaBetaEngine

def main():
    engine = AlphaBetaEngine()
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
            print("id name AlphaBeta_ChessRL")
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
            # Simple time management: use 1 second per move if not specified
            time_limit = 1.0
            for i, part in enumerate(parts):
                if part == "movetime":
                    time_limit = int(parts[i+1]) / 1000.0
            
            move = engine.search(board, max_depth=8, time_limit=time_limit)
            if move:
                print(f"bestmove {move.uci()}")
            else:
                # Fallback to any legal move
                fallback = list(board.legal_moves)[0] if list(board.legal_moves) else None
                if fallback:
                    print(f"bestmove {fallback.uci()}")
            sys.stdout.flush()
        elif command == "quit":
            break

if __name__ == "__main__":
    main()
