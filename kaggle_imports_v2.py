# Kaggle Setup & Import Cell (NumPy 2.0+ Compatible)
# This cell uses the latest versions of Torch, Pandas, and NumPy to avoid 'numpy.rec' errors.

# 1. Install Modern Dependencies
!pip install -q "numpy>=2.0.0" "torch>=2.3.0" "pandas>=2.2.2" "chess==1.10.0" "tqdm>=4.66.4" "datasets>=2.19.0" "psutil>=5.9.8"

import os
import sys
import numpy as np
import pandas as pd
import torch
import chess
from tqdm.notebook import tqdm

# 2. Verify Versions
print(f"✅ NumPy version: {np.__version__} (2.0+ ready)")
print(f"✅ Torch version: {torch.__version__} (NumPy 2.0 compatible)")
print(f"✅ Pandas version: {pd.__version__}")

# Setup local path
sys.path.append(os.getcwd())

print("\n🚀 Environment is fully modern and ready for Chess Research.")
if torch.cuda.is_available():
    print(f"GPU Active: {torch.cuda.get_device_name(0)}")
