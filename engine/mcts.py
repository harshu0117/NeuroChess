import numpy as np
import torch
import chess
import math
from engine.encoder import BoardEncoder

class MCTSNode:
    def __init__(self, board, parent=None, action_index=None, prior=0):
        self.board = board
        self.parent = parent
        self.action_index = action_index # The move that led to this node
        self.children = {} # action_index -> MCTSNode
        
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.state_tensor = None # To be filled by encoder

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total_visits = math.sqrt(total_visits) if total_visits > 0 else 0

        for action, child in self.children.items():
            # PUCT Formula: Q + C_puct * P * (sqrt(sum(N)) / (1 + N))
            u_score = c_puct * child.prior * (sqrt_total_visits / (1 + child.visit_count))
            score = child.value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, board, action_probs):
        """
        Expands the node with legal moves and their prior probabilities.
        action_probs: numpy array of size 4096 (policy head output)
        """
        for move in board.legal_moves:
            action_index = (move.from_square * 64) + move.to_square
            prior = action_probs[action_index]
            
            # Create board for the child
            child_board = board.copy()
            child_board.push(move)
            
            self.children[action_index] = MCTSNode(
                child_board, 
                parent=self, 
                action_index=action_index, 
                prior=prior
            )

class MCTSEngine:
    def __init__(self, model, encoder, c_puct=1.4):
        self.model = model
        self.encoder = encoder
        self.c_puct = c_puct
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def search(self, board, iterations=400):
        root = MCTSNode(board.copy())
        
        # Initial expansion of root
        with torch.no_grad():
            state_tensor = torch.from_numpy(self.encoder.encode(board)).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(state_tensor)
            
            # Apply legality mask and softmax
            probs = self._get_legal_probs(board, policy_logits[0].cpu().numpy())
            root.expand(board, probs)

        for _ in range(iterations):
            node = root
            search_board = board.copy()

            # 1. Select
            while node.children:
                action, node = node.select_child(self.c_puct)
                search_board.push(self.encoder.decode_move(action, search_board))

            # 2. Expand & Evaluate (Leaf)
            if not search_board.is_game_over():
                with torch.no_grad():
                    state_tensor = torch.from_numpy(self.encoder.encode(search_board)).unsqueeze(0).to(self.device)
                    policy_logits, value_tensor = self.model(state_tensor)
                    
                    probs = self._get_legal_probs(search_board, policy_logits[0].cpu().numpy())
                    node.expand(search_board, probs)
                    value = value_tensor.item()
            else:
                # Terminal node evaluation
                res = search_board.result()
                if res == "1-0":
                    value = 1.0 if search_board.turn == chess.BLACK else -1.0
                elif res == "0-1":
                    value = -1.0 if search_board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0

            # 3. Backpropagate
            # Value is from the perspective of the player at the leaf node.
            # We need to flip it at each level of the tree.
            curr_value = value
            while node:
                node.value_sum += curr_value
                node.visit_count += 1
                curr_value = -curr_value # Flip perspective
                node = node.parent

        # Return the best move (most visited)
        best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        return self.encoder.decode_move(best_action, board)

    def _get_legal_probs(self, board, logits):
        """
        Masks illegal moves and applies softmax.
        """
        mask = np.zeros(4096, dtype=bool)
        for move in board.legal_moves:
            idx = (move.from_square * 64) + move.to_square
            mask[idx] = True
            
        # Set illegal logits to very small number
        logits[~mask] = -1e10
        
        # Softmax
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()
