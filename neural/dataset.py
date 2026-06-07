import pandas as pd
import torch
from torch.utils.data import Dataset
import chess
from engine.encoder import BoardEncoder

class ChessDataset(Dataset):
    """
    Custom Dataset for training the ChessResNet.
    Expects a CSV with columns: 'fen', 'move', 'result'
    'move' should be in UCI format.
    'result' should be in [-1, 0, 1] relative to White.
    """
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.encoder = BoardEncoder()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        move_str = str(row['move'])
        
        # Result normalization: 
        # The dataset contains Stockfish evaluations (centipawns or mate strings).
        # We need to map this to [-1, 1].
        raw_eval = str(row['result'])
        if '#' in raw_eval:
            # Mate in X
            result = 1.0 if '+' in raw_eval else -1.0
        else:
            try:
                # Convert centipawns to a squashed value in [-1, 1]
                cp = float(raw_eval)
                result = torch.tanh(torch.tensor(cp / 300.0)).item()
            except ValueError:
                result = 0.0

        board = chess.Board(fen)
        
        # 1. Encode State (Tensor)
        state_tensor = self.encoder.encode(board)
        
        # 2. Encode Policy Target (Index)
        try:
            # Try parsing as UCI first
            move = chess.Move.from_uci(move_str)
        except ValueError:
            # Fallback to SAN (requires board context)
            try:
                move = board.parse_san(move_str)
            except ValueError:
                # If both fail, return a dummy or skip
                # For now, let's use the first legal move as a fallback
                move = list(board.legal_moves)[0]
                
        policy_target = self.encoder.encode_move(move, board)
        
        # 3. Encode Value Target (z)
        value_target = result if board.turn == chess.WHITE else -result
        
        return (
            torch.from_numpy(state_tensor),
            torch.tensor(policy_target, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32)
        )
