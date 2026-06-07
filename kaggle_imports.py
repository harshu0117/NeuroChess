# Kaggle Import and Setup Cell
# This cell installs dependencies and imports all necessary modules for the project.

# 1. Install Dependencies
# !pip install -q chess torch numpy streamlit fastapi uvicorn pandas datasets psutil pynvml tqdm

import os
import sys
import subprocess
import time
import json
import math
import threading
import requests
import zipfile
import shutil
import re

# Science & ML
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Chess Logic
import chess
import chess.pgn
import chess.engine

# Progress Bars & Telemetry
from tqdm.notebook import tqdm
import psutil
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# Project Path Setup
# Ensure the current directory is in the system path for local imports
sys.path.append(os.getcwd())

print("✅ All libraries imported successfully.")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
