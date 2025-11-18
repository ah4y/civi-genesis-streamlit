"""
Neural Network model for CIVI-GENESIS.
Train and use an MLP to approximate LLM-generated citizen reactions.
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class CitizenReactionModel:
    """
    Neural network model that learns to approximate citizen reaction deltas.
    
    This model is trained on LLM-generated samples and can then predict
    reaction deltas for large populations without calling the LLM.
    """
    
    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 32),
        max_iter: int = 500,
        random_state: int = 42
    ):
        """
        Initialize the reaction model.
        
        Args:
            hidden_layers: Tuple specifying hidden layer sizes.
            max_iter: Maximum training iterations.
            random_state: Random seed for reproducibility.
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        self.scaler = None
        self.is_trained = False
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None
    ) -> dict:
        """
        Train the model on LLM-generated samples.
        
        Args:
            X_train: Training features.
            Y_train: Training targets [delta_h, delta_s, delta_i].
            X_val: Validation features (optional).
            Y_val: Validation targets (optional).
        
        Returns:
            Dict with training metrics.
        """
        logger.info(f"Training model on {len(X_train)} samples...")
        
        # Fit the model
        self.model.fit(X_train, Y_train)
        self.is_trained = True
        
        # Compute training metrics
        Y_train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(Y_train, Y_train_pred)
        train_mse = mean_squared_error(Y_train, Y_train_pred)
        
        metrics = {
            "train_mae": float(train_mae),
            "train_mse": float(train_mse),
            "n_samples": len(X_train),
            "n_iterations": self.model.n_iter_
        }
        
        # Compute validation metrics if provided
        if X_val is not None and Y_val is not None:
            Y_val_pred = self.model.predict(X_val)
            val_mae = mean_absolute_error(Y_val, Y_val_pred)
            val_mse = mean_squared_error(Y_val, Y_val_pred)
            metrics["val_mae"] = float(val_mae)
            metrics["val_mse"] = float(val_mse)
            
            logger.info(f"Training complete. Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        else:
            logger.info(f"Training complete. Train MAE: {train_mae:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict reaction deltas for given features.
        
        Args:
            X: Feature array (can be single sample or batch).
        
        Returns:
            Predicted deltas [delta_h, delta_s, delta_i].
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return self.model.predict(X)
    
    def save(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model.
            scaler_path: Path to save the scaler (optional).
        """
        if not self.is_trained:
            logger.warning("Attempting to save untrained model")
        
        # Create directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler if provided
        if scaler_path and self.scaler:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
    
    @classmethod
    def load(
        cls,
        model_path: str,
        scaler_path: Optional[str] = None
    ) -> Optional['CitizenReactionModel']:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model.
            scaler_path: Path to the saved scaler (optional).
        
        Returns:
            CitizenReactionModel instance or None if loading fails.
        """
        try:
            instance = cls()
            instance.model = joblib.load(model_path)
            instance.is_trained = True
            
            # Load scaler if provided
            if scaler_path and Path(scaler_path).exists():
                instance.scaler = joblib.load(scaler_path)
            
            logger.info(f"Model loaded from {model_path}")
            return instance
        
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None


def train_reaction_model(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.2,
    hidden_layers: Tuple[int, ...] = (64, 32),
    max_iter: int = 500,
    random_state: int = 42
) -> Tuple[CitizenReactionModel, dict]:
    """
    Train a citizen reaction model with automatic train/test split.
    
    Args:
        X: Feature array.
        Y: Target array.
        test_size: Proportion for validation set.
        hidden_layers: Hidden layer sizes.
        max_iter: Maximum iterations.
        random_state: Random seed.
    
    Returns:
        Tuple of (trained model, metrics dict).
    """
    from ml_data import split_dataset, normalize_features
    
    # Split data
    X_train, X_val, Y_train, Y_val = split_dataset(X, Y, test_size, random_state)
    
    # Normalize features
    X_train_scaled, X_val_scaled, scaler = normalize_features(X_train, X_val)
    
    # Create and train model
    model = CitizenReactionModel(hidden_layers, max_iter, random_state)
    model.scaler = scaler
    
    metrics = model.train(X_train_scaled, Y_train, X_val_scaled, Y_val)
    
    return model, metrics


def load_reaction_model(model_path: str = "models/citizen_reaction_mlp.joblib") -> Optional[CitizenReactionModel]:
    """
    Load a reaction model from the default path.
    
    Args:
        model_path: Path to the model file.
    
    Returns:
        Loaded model or None if not found.
    """
    if not Path(model_path).exists():
        logger.info(f"No model found at {model_path}")
        return None
    
    return CitizenReactionModel.load(model_path)
