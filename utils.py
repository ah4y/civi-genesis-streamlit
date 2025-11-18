"""
Utility functions for CIVI-GENESIS.
Helper functions for data conversion, formatting, and feature engineering.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from data_models import Citizen, CitizenState, CITY_ZONES, INCOME_LEVELS, POLITICAL_VIEWS, DOMAINS


def citizen_to_dict(citizen: Citizen) -> Dict[str, Any]:
    """Convert a Citizen object to a dictionary."""
    return {
        "id": citizen.id,
        "age": citizen.age,
        "gender": citizen.gender,
        "city_zone": citizen.city_zone,
        "income_level": citizen.income_level,
        "education": citizen.education,
        "profession": citizen.profession,
        "family_size": citizen.family_size,
        "political_view": citizen.political_view,
        "risk_tolerance": citizen.risk_tolerance,
        "openness_to_change": citizen.openness_to_change,
        "base_happiness": citizen.base_happiness,
        "base_income": citizen.base_income
    }


def state_to_dict(state: CitizenState) -> Dict[str, Any]:
    """Convert a CitizenState object to a dictionary."""
    return {
        "citizen_id": state.citizen_id,
        "step": state.step,
        "happiness": state.happiness,
        "policy_support": state.policy_support,
        "income": state.income,
        "diary_entry": state.diary_entry,
        "llm_updated": state.llm_updated
    }


def citizens_to_dataframe(citizens: List[Citizen]) -> pd.DataFrame:
    """Convert list of citizens to a pandas DataFrame."""
    return pd.DataFrame([citizen_to_dict(c) for c in citizens])


def states_to_dataframe(states: List[CitizenState]) -> pd.DataFrame:
    """Convert list of citizen states to a pandas DataFrame."""
    return pd.DataFrame([state_to_dict(s) for s in states])


def format_support(support: float) -> str:
    """
    Format policy support value as percentage.
    
    Args:
        support: Support value from -1 to 1.
    
    Returns:
        Formatted string like "+45%" or "-23%".
    """
    percentage = support * 100
    sign = "+" if percentage >= 0 else ""
    return f"{sign}{percentage:.0f}%"


def format_income(income: float) -> str:
    """Format income value with currency symbol."""
    return f"${income:.0f}"


def encode_categorical_one_hot(value: str, categories: List[str]) -> List[float]:
    """
    One-hot encode a categorical value.
    
    Args:
        value: The categorical value to encode.
        categories: List of all possible categories.
    
    Returns:
        One-hot encoded list.
    """
    encoding = [0.0] * len(categories)
    if value in categories:
        idx = categories.index(value)
        encoding[idx] = 1.0
    return encoding


def build_feature_vector(
    citizen: Citizen,
    prev_state: CitizenState,
    policy_domain: str
) -> np.ndarray:
    """
    Build a feature vector for ML model input.
    
    Combines citizen static attributes, previous state, and policy domain
    into a single feature vector for the neural network.
    
    Args:
        citizen: Citizen object with demographic and personality data.
        prev_state: Previous CitizenState with happiness, support, income.
        policy_domain: Policy domain (Economy, Education, Social, Startup).
    
    Returns:
        NumPy array of features.
    """
    features = []
    
    # Citizen static features
    features.append(citizen.age / 100.0)  # Normalize age
    features.extend(encode_categorical_one_hot(citizen.income_level, INCOME_LEVELS))
    features.extend(encode_categorical_one_hot(citizen.city_zone, CITY_ZONES))
    features.extend(encode_categorical_one_hot(citizen.political_view, POLITICAL_VIEWS))
    features.append(citizen.risk_tolerance)
    features.append(citizen.openness_to_change)
    features.append(citizen.family_size / 10.0)  # Normalize family size
    
    # Previous state features
    features.append(prev_state.happiness)
    features.append(prev_state.policy_support)
    features.append(np.log1p(prev_state.income) / 10.0)  # Log-scaled income
    
    # Policy domain features
    features.extend(encode_categorical_one_hot(policy_domain, DOMAINS))
    
    return np.array(features, dtype=np.float32)


def get_feature_dimension() -> int:
    """
    Get the total dimension of feature vectors.
    
    Returns:
        Integer dimension of feature space.
    """
    # Age (1) + income_level (3) + city_zone (4) + political_view (3) +
    # risk_tolerance (1) + openness_to_change (1) + family_size (1) +
    # prev_happiness (1) + prev_support (1) + prev_income (1) + policy_domain (4)
    return 1 + 3 + 4 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 4


def compute_deltas(
    prev_state: CitizenState,
    new_happiness: float,
    new_support: float,
    new_income: float
) -> np.ndarray:
    """
    Compute deltas (changes) in citizen state.
    
    Args:
        prev_state: Previous state.
        new_happiness: New happiness value.
        new_support: New support value.
        new_income: New income value.
    
    Returns:
        NumPy array [delta_happiness, delta_support, delta_income].
    """
    delta_h = new_happiness - prev_state.happiness
    delta_s = new_support - prev_state.policy_support
    delta_i = new_income - prev_state.income
    
    return np.array([delta_h, delta_s, delta_i], dtype=np.float32)


def apply_deltas(
    prev_state: CitizenState,
    deltas: np.ndarray
) -> tuple[float, float, float]:
    """
    Apply deltas to previous state and clamp values.
    
    Args:
        prev_state: Previous CitizenState.
        deltas: NumPy array [delta_happiness, delta_support, delta_income].
    
    Returns:
        Tuple of (new_happiness, new_support, new_income) with clamped values.
    """
    new_happiness = np.clip(prev_state.happiness + deltas[0], 0.0, 1.0)
    new_support = np.clip(prev_state.policy_support + deltas[1], -1.0, 1.0)
    new_income = max(0.0, prev_state.income + deltas[2])
    
    return float(new_happiness), float(new_support), float(new_income)


def get_policy_presets() -> List[Dict[str, str]]:
    """
    Get predefined policy presets for quick demos.
    
    Returns:
        List of policy dicts with title, description, and domain.
    """
    return [
        {
            "title": "Fuel Subsidy Removal",
            "description": "Remove government fuel subsidies to reduce budget deficit. Gas prices will increase by 30%.",
            "domain": "Economy"
        },
        {
            "title": "Student Scholarship Program",
            "description": "Launch a comprehensive scholarship program providing $5000/year to low-income students pursuing higher education.",
            "domain": "Education"
        },
        {
            "title": "AI Tutor for Low-Income Students",
            "description": "Deploy free AI-powered tutoring platform accessible to all students in low-income neighborhoods.",
            "domain": "Education"
        },
        {
            "title": "Startup Pricing Change",
            "description": "A food delivery startup increases delivery fees by 25% and reduces driver commissions by 10%.",
            "domain": "Startup"
        },
        {
            "title": "Universal Basic Income Pilot",
            "description": "Test a universal basic income program providing $500/month to all citizens for 6 months.",
            "domain": "Social"
        }
    ]
