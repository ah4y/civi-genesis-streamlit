"""
Population generation module for CIVI-GENESIS.
Generate synthetic citizen populations with controllable distributions.
"""
import numpy as np
from typing import List, Optional
from data_models import Citizen, CITY_ZONES, INCOME_LEVELS, POLITICAL_VIEWS, GENDERS, EDUCATION_LEVELS


# Profession mappings by income level
PROFESSIONS_BY_INCOME = {
    "low": [
        "Retail Worker", "Food Service Worker", "Janitor", "Security Guard",
        "Factory Worker", "Delivery Driver", "Warehouse Worker", "Cashier"
    ],
    "middle": [
        "Teacher", "Nurse", "Accountant", "Engineer", "Sales Manager",
        "IT Specialist", "Government Employee", "Technician", "Project Manager"
    ],
    "high": [
        "Doctor", "Lawyer", "CEO", "Senior Engineer", "Investment Banker",
        "Business Owner", "Consultant", "Architect", "Executive"
    ]
}


def generate_population(
    size: int,
    seed: Optional[int] = None,
    low_share: float = 0.4,
    middle_share: float = 0.4,
    high_share: float = 0.2
) -> List[Citizen]:
    """
    Generate a synthetic population of citizens.
    
    Args:
        size: Number of citizens to generate (up to 50,000).
        seed: Random seed for reproducibility.
        low_share: Proportion of low-income citizens (default 0.4).
        middle_share: Proportion of middle-income citizens (default 0.4).
        high_share: Proportion of high-income citizens (default 0.2).
    
    Returns:
        List of Citizen objects with randomized but realistic attributes.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Validate shares sum to approximately 1.0
    total_share = low_share + middle_share + high_share
    if not (0.99 <= total_share <= 1.01):
        raise ValueError(f"Income shares must sum to 1.0, got {total_share}")
    
    # Calculate counts for each income level
    low_count = int(size * low_share)
    middle_count = int(size * middle_share)
    high_count = size - low_count - middle_count  # Ensure exact total
    
    # Create income level assignments
    income_levels = (
        ["low"] * low_count +
        ["middle"] * middle_count +
        ["high"] * high_count
    )
    rng.shuffle(income_levels)
    
    citizens = []
    
    for i in range(size):
        income_level = income_levels[i]
        
        # Age: 18 to 70
        age = int(rng.integers(18, 71))
        
        # Gender
        gender = rng.choice(GENDERS)
        
        # City zone
        city_zone = rng.choice(CITY_ZONES)
        
        # Education (correlated with income)
        if income_level == "low":
            education = rng.choice(EDUCATION_LEVELS[:2], p=[0.7, 0.3])
        elif income_level == "middle":
            education = rng.choice(EDUCATION_LEVELS[1:4], p=[0.2, 0.6, 0.2])
        else:  # high
            education = rng.choice(EDUCATION_LEVELS[2:], p=[0.1, 0.4, 0.5])
        
        # Profession based on income level
        profession = rng.choice(PROFESSIONS_BY_INCOME[income_level])
        
        # Family size: 1 to 6
        family_size = int(rng.integers(1, 7))
        
        # Political view
        political_view = rng.choice(POLITICAL_VIEWS)
        
        # Personality traits
        risk_tolerance = float(rng.uniform(0.0, 1.0))
        openness_to_change = float(rng.uniform(0.0, 1.0))
        
        # Base happiness: 0.3 to 0.8
        base_happiness = float(rng.uniform(0.3, 0.8))
        
        # Base income depends on income level
        if income_level == "low":
            base_income = float(rng.uniform(300, 700))
        elif income_level == "middle":
            base_income = float(rng.uniform(800, 2000))
        else:  # high
            base_income = float(rng.uniform(2000, 8000))
        
        citizen = Citizen(
            id=i,
            age=age,
            gender=gender,
            city_zone=city_zone,
            income_level=income_level,
            education=education,
            profession=profession,
            family_size=family_size,
            political_view=political_view,
            risk_tolerance=risk_tolerance,
            openness_to_change=openness_to_change,
            base_happiness=base_happiness,
            base_income=base_income
        )
        
        citizens.append(citizen)
    
    return citizens
