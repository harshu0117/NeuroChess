from datasets import load_dataset
import pandas as pd
import os

def fetch_and_sample(output_path, num_samples=2000000):
    print(f"Streaming dataset from Hugging Face...")
    
    # Load the dataset in streaming mode to avoid downloading 9GB
    ds = load_dataset("bingbangboom/stockfish-evaluation-SAN", split="train", streaming=True)
    
    data = []
    count = 0
    
    print(f"Extracting first {num_samples} rows...")
    for entry in ds:
        # Map their column names to our expected format
        # Their columns: 'fen', 'best_move', 'evaluation'
        # Note: best_move is in SAN, we might need to convert it to UCI 
        # but our encoder/dataset can handle either if we adjust it.
        # However, it's safer to store it directly.
        
        # Simple evaluation normalization: 
        # Evaluation is often in centipawns. Let's keep it as is for now
        # and handle normalization in the DataLoader to keep this raw.
        
        data.append({
            'fen': entry['fen'],
            'move': entry['best_move'],
            'result': entry['evaluation']
        })
        
        count += 1
        if count % 10000 == 0:
            print(f"Progress: {count}/{num_samples}...")
        
        if count >= num_samples:
            break
            
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    # Path where we want the processed data
    fetch_and_sample('data/maia_chess.csv', num_samples=2000000)
