"""
Machine Learning data management for CIVI-GENESIS.
Handle training dataset creation, storage, and management.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLDataset:
    """
    Manages the collection and storage of training samples from LLM outputs.
    """
    
    def __init__(self):
        """Initialize an empty dataset."""
        self.X: List[np.ndarray] = []  # Feature vectors
        self.Y: List[np.ndarray] = []  # Target deltas [delta_h, delta_s, delta_i]
    
    def add_sample(self, features: np.ndarray, targets: np.ndarray):
        """
        Add a training sample to the dataset.
        
        Args:
            features: Feature vector (X).
            targets: Target deltas [delta_happiness, delta_support, delta_income] (Y).
        """
        self.X.append(features)
        self.Y.append(targets)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the dataset as numpy arrays.
        
        Returns:
            Tuple of (X, Y) as 2D numpy arrays.
        """
        if not self.X:
            return np.array([]), np.array([])
        
        X_array = np.vstack(self.X)
        Y_array = np.vstack(self.Y)
        return X_array, Y_array
    
    def size(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.X)
    
    def clear(self):
        """Clear all samples from the dataset."""
        self.X = []
        self.Y = []
    
    def save_to_csv(self, filepath: str):
        """
        Save the dataset to a CSV file.
        
        Args:
            filepath: Path to save the CSV file.
        """
        if not self.X:
            logger.warning("No data to save")
            return
        
        X_array, Y_array = self.get_arrays()
        
        # Create column names
        n_features = X_array.shape[1]
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        target_cols = ["delta_happiness", "delta_support", "delta_income"]
        
        # Combine X and Y
        data = np.hstack([X_array, Y_array])
        df = pd.DataFrame(data, columns=feature_cols + target_cols)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} samples to {filepath}")
    
    def load_from_csv(self, filepath: str):
        """
        Load the dataset from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
        """
        try:
            df = pd.read_csv(filepath)
            
            # Separate features and targets
            target_cols = ["delta_happiness", "delta_support", "delta_income"]
            feature_cols = [col for col in df.columns if col not in target_cols]
            
            X_array = df[feature_cols].values
            Y_array = df[target_cols].values
            
            # Convert to lists of arrays
            self.X = [X_array[i] for i in range(len(X_array))]
            self.Y = [Y_array[i] for i in range(len(Y_array))]
            
            logger.info(f"Loaded {len(self.X)} samples from {filepath}")
        
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
    
    def merge(self, other: 'MLDataset'):
        """
        Merge another dataset into this one.
        
        Args:
            other: Another MLDataset to merge.
        """
        self.X.extend(other.X)
        self.Y.extend(other.Y)


def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets.
    
    Args:
        X: Feature array.
        Y: Target array.
        test_size: Proportion of data for test set (default 0.2).
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test).
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


def normalize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Normalize features using StandardScaler.
    
    Args:
        X_train: Training features.
        X_test: Test features (optional).
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler).
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
