import chess
import chess.polyglot

class TranspositionTable:
    """
    A thread-safe-ish (for single process) transposition table for caching search results.
    Uses FIFO eviction when the capacity limit is reached.
    """
    
    def __init__(self, max_entries=10**6):
        self.max_entries = max_entries
        self.table = {}
        self.keys_fifo = [] # Track keys for FIFO eviction

    def get(self, board: chess.Board):
        """
        Retrieves an entry from the table using Zobrist hash.
        """
        key = chess.polyglot.zobrist_hash(board)
        return self.table.get(key)

    def store(self, board: chess.Board, depth, flag, value, best_move=None):
        """
        Stores an entry in the table.
        flag: 'EXACT', 'LOWERBOUND' (Beta cutoff), 'UPPERBOUND' (Alpha fail-low)
        """
        key = chess.polyglot.zobrist_hash(board)
        
        if key not in self.table:
            # Check for capacity and evict if necessary
            if len(self.table) >= self.max_entries:
                oldest_key = self.keys_fifo.pop(0)
                del self.table[oldest_key]
            
            self.keys_fifo.append(key)
        
        # Store or update the entry
        self.table[key] = {
            'depth': depth,
            'flag': flag,
            'value': value,
            'best_move': best_move.uci() if best_move else None
        }

    def clear(self):
        """
        Clears the transposition table.
        """
        self.table.clear()
        self.keys_fifo = []
