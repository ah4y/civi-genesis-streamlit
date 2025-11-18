"""
Configuration module for CIVI-GENESIS.
Centralize configuration and environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings and configuration."""
    
    # API Keys
    gemini_api_key: str
    backup_api_keys: Optional[List[str]] = None  # Optional backup keys for rotation
    summary_api_key: Optional[str] = None  # Dedicated key for summaries
    
    # Default simulation parameters
    default_population_size: int = 1000
    default_steps: int = 5
    max_population_size: int = 50000
    
    # LLM sampling configuration
    llm_sample_size: int = 300  # Max citizens to sample with LLM per step
    
    # Neural network configuration
    nn_hidden_layers: Tuple[int, ...] = (64, 32)
    nn_max_iter: int = 500
    nn_min_training_samples: int = 500
    
    # Random seed (optional)
    random_seed: Optional[int] = None
    
    # Income distribution defaults
    low_income_share: float = 0.4
    middle_income_share: float = 0.4
    high_income_share: float = 0.2


def get_settings() -> Settings:
    """
    Get application settings from environment variables.
    
    Returns:
        Settings object with configuration values.
    
    Raises:
        ValueError: If required environment variables are missing.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required. "
            "Please set it in your environment or .env file."
        )
    
    # Load backup API keys (comma-separated)
    backup_keys_str = os.getenv("GEMINI_BACKUP_KEYS", "")
    backup_keys = [key.strip() for key in backup_keys_str.split(",") if key.strip()]
    
    # Load dedicated summary API key (optional)
    summary_api_key = os.getenv("GEMINI_SUMMARY_API_KEY", "").strip() or None
    
    return Settings(
        gemini_api_key=gemini_api_key,
        backup_api_keys=backup_keys if backup_keys else None,
        summary_api_key=summary_api_key,
        random_seed=int(os.getenv("RANDOM_SEED", 0)) or None
    )
