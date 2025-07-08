from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(data_dir: str = "data/log", test_size: float = 0.2, seed: int = 42):
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    
    if len(X) == 0:
        raise ValueError("Dataset is empty.")

    return train_test_split(X, y, test_size=test_size, random_state=seed)
