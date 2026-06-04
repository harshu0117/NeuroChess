import chess
import time
from engine.evaluation import evaluate_board
from engine.transposition import TranspositionTable

class AlphaBetaEngine:
    def __init__(self, tt_capacity=10**6):
        self.tt = TranspositionTable(max_entries=tt_capacity)
        self.nodes_visited = 0
        self.best_move_so_far = None
        self.start_time = 0
        self.time_limit = 0
        
        # MVV-LVA values: [Aggressor][Victim]
        # P, N, B, R, Q, K
        self.mvv_lva_scores = [
            [15, 25, 35, 45, 55, 65], # Pawn aggressor
            [14, 24, 34, 44, 54, 64], # Knight aggressor
            [13, 23, 33, 43, 53, 63], # Bishop aggressor
            [12, 22, 32, 42, 52, 62], # Rook aggressor
            [11, 21, 31, 41, 51, 61], # Queen aggressor
            [10, 20, 30, 40, 50, 60]  # King aggressor
        ]
        self.piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

    def get_mvv_lva(self, board, move):
        victim = board.piece_at(move.to_square)
        if not victim:
            return 0
        aggressor = board.piece_at(move.from_square)
        return self.mvv_lva_scores[self.piece_idx[aggressor.piece_type]][self.piece_idx[victim.piece_type]]

    def sort_moves(self, board, depth, tt_move_uci=None):
        moves = list(board.legal_moves)
        move_scores = []
        
        for move in moves:
            score = 0
            move_uci = move.uci()
            
            # Priority 1: TT move
            if tt_move_uci and move_uci == tt_move_uci:
                score = 10000
            # Priority 2: Captures (MVV-LVA)
            elif board.is_capture(move):
                score = 1000 + self.get_mvv_lva(board, move)
            # Priority 3: Promotions and Checks
            elif move.promotion:
                score = 500
            elif board.gives_check(move):
                score = 100
            
            move_scores.append((score, move))
            
        # Sort descending by score
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in move_scores]

    def quiescence(self, board, alpha, beta):
        self.nodes_visited += 1
        
        # Periodic time check in quiescence too
        if self.nodes_visited % 1024 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
                
        stand_pat = evaluate_board(board)
        current_eval = stand_pat if board.turn == chess.WHITE else -stand_pat
        
        if current_eval >= beta:
            return beta
        if alpha < current_eval:
            alpha = current_eval
            
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                try:
                    score = -self.quiescence(board, -beta, -alpha)
                finally:
                    board.pop()
                
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha

    def alphabeta(self, board, depth, alpha, beta):
        self.nodes_visited += 1
        
        # Time check
        if self.nodes_visited % 1024 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()

        # TT Lookup
        tt_entry = self.tt.get(board)
        tt_move_uci = None
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['value']
            elif tt_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_entry['value'])
            
            if alpha >= beta:
                return tt_entry['value']
            tt_move_uci = tt_entry['best_move']

        if depth == 0:
            return self.quiescence(board, alpha, beta)
        
        if board.is_game_over():
            score = evaluate_board(board)
            return score if board.turn == chess.WHITE else -score

        best_move = None
        best_value = -float('inf')
        original_alpha = alpha
        
        sorted_moves = self.sort_moves(board, depth, tt_move_uci)
        
        for move in sorted_moves:
            board.push(move)
            try:
                value = -self.alphabeta(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, value)
            if alpha >= beta:
                break
                
        # TT Store
        flag = 'EXACT'
        if best_value <= original_alpha:
            flag = 'UPPERBOUND'
        elif best_value >= beta:
            flag = 'LOWERBOUND'
            
        self.tt.store(board, depth, flag, best_value, best_move)
        
        return best_value

    def search(self, board, max_depth=10, time_limit=5.0):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes_visited = 0
        self.best_move_so_far = None
        
        # Always have a fallback move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        self.best_move_so_far = legal_moves[0]

        try:
            for depth in range(1, max_depth + 1):
                best_val = -float('inf')
                alpha = -float('inf')
                beta = float('inf')
                
                # Use a local best move for this depth
                sorted_moves = self.sort_moves(board, depth)
                current_best_move = sorted_moves[0] if sorted_moves else None
                
                for move in sorted_moves:
                    board.push(move)
                    try:
                        val = -self.alphabeta(board, depth - 1, -beta, -alpha)
                    finally:
                        board.pop()
                    
                    if val > best_val:
                        best_val = val
                        current_best_move = move
                    alpha = max(alpha, val)
                
                # Only update the global best move if we finished the full depth
                if current_best_move:
                    self.best_move_so_far = current_best_move
                
        except TimeoutError:
            # Iterative deepening naturally handles timeouts by using the move from the last complete depth
            pass
            
        return self.best_move_so_far
