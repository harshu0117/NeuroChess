import chess
import torch
import numpy as np
import sys
import os
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.getcwd())

from engine.mcts import MCTSEngine
from engine.encoder import BoardEncoder
from neural.model import ChessResNet

class SelfPlay:
    def __init__(self, model, encoder, iterations=50):
        self.model = model
        self.encoder = encoder
        self.iterations = iterations
        self.engine = MCTSEngine(model, encoder)
        self.memory = []

    def play_game(self):
        board = chess.Board()
        game_history = []
        max_moves = 100
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            move = self.engine.search(board.copy(), iterations=self.iterations)
            state_tensor = self.encoder.encode(board)
            game_history.append({
                'state': state_tensor,
                'turn': board.turn,
                'move_idx': self.encoder.encode_move(move)
            })
            board.push(move)
            move_count += 1
            
        result = board.result()
        if result == "1-0": z = 1.0
        elif result == "0-1": z = -1.0
        else: z = 0.0
            
        for step in game_history:
            reward = z if step['turn'] == chess.WHITE else -z
            self.memory.append((step['state'], step['move_idx'], reward))
        return result

    def train_on_memory(self, batch_size=32, lr=0.0001):
        if len(self.memory) < batch_size:
            return 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states = torch.tensor(np.array([b[0] for b in batch])).to(device)
        policy_targets = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
        value_targets = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
        optimizer.zero_grad()
        p_out, v_out = self.model(states)
        p_loss = torch.nn.functional.cross_entropy(p_out, policy_targets)
        v_loss = torch.nn.functional.mse_loss(v_out.view(-1), value_targets)
        loss = p_loss + v_loss
        loss.backward()
        optimizer.step()
        return loss.item()

def run_rl_finetuning(num_games=20, train_frequency=2, model_path="models/supervised_base.pt"):
    print(f"🌟 Starting Reinforcement Learning Fine-Tuning ({num_games} games)...")
    model = ChessResNet()
    encoder = BoardEncoder()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✅ Loaded base weights for fine-tuning.")
    except:
        print("⚠️ Warning: No base weights found.")
    
    sp = SelfPlay(model, encoder, iterations=100) 
    pbar = tqdm(total=num_games, desc="RL Progress", unit="game")
    
    for i in range(num_games):
        sp.play_game()
        pbar.update(1)
        if (i + 1) % train_frequency == 0:
            loss = sp.train_on_memory(batch_size=min(len(sp.memory), 64))
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Memory": len(sp.memory)})
            
    save_path = "models/rl_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    pbar.close()
    print(f"🎉 RL Fine-tuning complete! Weights saved to {save_path}")

if __name__ == "__main__":
    run_rl_finetuning(num_games=20)
