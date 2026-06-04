import chess

# Material Values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables (PST)
# Tables are defined from White's perspective. 
# For Black, we will flip the rank index.
# Higher values = better position for the piece.

PST = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
}

def evaluate_board(board: chess.Board):
    """
    Calculates the static evaluation of the board.
    Positive values favor White, negative values favor Black.
    """
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999 # Black wins
        else:
            return 99999  # White wins
    
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition():
        return 0

    score = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            color = piece.color
            
            # Material value
            val = PIECE_VALUES[piece_type]
            
            # PST value
            # chess.SQUARES are indexed 0-63 (A1-H8). 
            # Our PST tables are indexed 0-63 (A8-H1 logically in typical array representation, 
            # but let's align them to chess.SQUARES: 0 is A1, 7 is H1, 56 is A8, 63 is H8).
            # Wait, standard PSTs in tutorials often use 0 as A8. 
            # Let's check: chess.SQUARES[0] is A1.
            # My PST lists above: PST[chess.PAWN][0] should be A8 value (0). 
            # Let's adjust access: rank 0 is index 56-63, rank 7 is index 0-7.
            
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            if color == chess.WHITE:
                pst_idx = (7 - rank) * 8 + file
                score += val + PST[piece_type][pst_idx]
            else:
                # For black, flip the rank to use the same PST
                pst_idx = rank * 8 + file
                score -= (val + PST[piece_type][pst_idx])
                
    return score
