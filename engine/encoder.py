import numpy as np
import chess

class BoardEncoder:
    """
    Encodes chess.Board states into 8x8x13 tensors and manages move indexing.
    """
    
    def __init__(self):
        # Piece types in order for channel mapping
        self.piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING
        ]

    def encode(self, board: chess.Board):
        """
        Encodes the board state into a (13, 8, 8) float32 tensor.
        Perspective: The board is always viewed from the side of the player whose turn it is.
        """
        # Create empty tensor: 13 channels, 8x8 grid
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        
        # Determine if we need to flip the perspective (if it's Black's turn)
        is_black_turn = board.turn == chess.BLACK
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Calculate coordinates in the 8x8 grid
                # chess.SQUARES are 0-63, starting from A1 (0), B1 (1) ... H8 (63)
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Flip rank if it's Black's turn for perspective-neutrality
                if is_black_turn:
                    rank = 7 - rank
                
                # Determine channel index
                # Channels 0-5: Current player's pieces
                # Channels 6-11: Opponent's pieces
                if piece.color == board.turn:
                    channel = self.piece_types.index(piece.piece_type)
                else:
                    channel = self.piece_types.index(piece.piece_type) + 6
                
                tensor[channel, rank, file] = 1.0
        
        # Channel 12: Constant plane (1.0 for White to move, 0.0 for Black)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
        else:
            tensor[12, :, :] = 0.0
            
        return tensor

    def encode_move(self, move: chess.Move, board: chess.Board):
        """
        Maps a chess.Move to a flat index in the 4,096 action space.
        If it's Black's turn, we must flip the move coordinates to match the flipped board perspective.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        if board.turn == chess.BLACK:
            from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
            to_sq = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
            
        return (from_sq * 64) + to_sq

    def decode_move(self, action_index: int, board: chess.Board):
        """
        Maps a 4,096 action index back to a chess.Move object.
        If it's Black's turn, we must flip the move coordinates back.
        """
        from_sq = action_index // 64
        to_sq = action_index % 64
        
        if board.turn == chess.BLACK:
            from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
            to_sq = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
            
        move = chess.Move(from_sq, to_sq)
        
        # Handle Pawn Promotion
        if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
            if chess.square_rank(to_sq) in [0, 7]:
                move.promotion = chess.QUEEN
                
        return move
