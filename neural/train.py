import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from neural.model import ChessResNet
from neural.dataset import ChessDataset

def train(csv_path, model_save_path, epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize components
    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = ChessResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for states, policy_targets, value_targets in pbar:
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_out, value_out = model(states)
            
            # Calculate losses
            p_loss = policy_criterion(policy_out, policy_targets)
            v_loss = value_criterion(value_out.view(-1), value_targets)
            
            loss = p_loss + v_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            policy_loss_sum += p_loss.item()
            value_loss_sum += v_loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} (Policy: {policy_loss_sum/len(dataloader):.4f}, Value: {value_loss_sum/len(dataloader):.4f})")
        
        # Save checkpoint
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    import os
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Configuration
    DATA_PATH = 'data/maia_chess.csv'
    SAVE_PATH = 'models/supervised_base.pt'
    
    if os.path.exists(DATA_PATH):
        print(f"Starting supervised training on {DATA_PATH}...")
        train(
            csv_path=DATA_PATH, 
            model_save_path=SAVE_PATH, 
            epochs=12,       # Trimmed to 12 based on plateau observation
            batch_size=2048, # Optimized for 15GB GPU
            lr=0.001
        )
    else:
        print(f"Error: Dataset not found at {DATA_PATH}. Please run data/fetch_data.py first.")
