"""
Core data models and domain structures for CIVI-GENESIS.
Defines citizens, states, policies, and simulation results.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Citizen:
    """Represents a synthetic citizen with demographic and personality attributes."""
    
    id: int
    age: int
    gender: str
    city_zone: str  # downtown, industrial, suburban, rural
    income_level: str  # low, middle, high
    education: str
    profession: str
    family_size: int
    political_view: str  # conservative, neutral, progressive
    risk_tolerance: float  # 0-1
    openness_to_change: float  # 0-1
    base_happiness: float  # 0-1
    base_income: float  # >= 0


@dataclass
class CitizenState:
    """Represents the state of a citizen at a specific simulation step."""
    
    citizen_id: int
    step: int
    happiness: float  # 0-1
    policy_support: float  # -1 to 1
    income: float  # >= 0
    diary_entry: Optional[str] = None
    llm_updated: bool = False  # True if updated via LLM, False if rule-based or NN


@dataclass
class PolicyInput:
    """Represents a policy or business idea to be simulated."""
    
    title: str
    description: str
    domain: str  # Economy, Education, Social, Startup


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    
    name: str
    population_size: int
    steps: int
    policy: PolicyInput
    random_seed: Optional[int] = None
    mode: str = "LLM_ONLY"  # LLM_ONLY, HYBRID, NN_ONLY


@dataclass
class StepStats:
    """Aggregated statistics for a simulation step."""
    
    step: int
    avg_happiness: float
    avg_support: float
    avg_income: float
    by_income: List[Dict[str, Any]] = field(default_factory=list)
    # Each dict: {income_level: str, avg_happiness: float, avg_support: float}
    by_zone: List[Dict[str, Any]] = field(default_factory=list)
    # Each dict: {city_zone: str, avg_happiness: float, avg_support: float}
    inequality_gap_happiness: float = 0.0  # high minus low income avg happiness
    inequality_gap_support: float = 0.0  # high minus low income avg support


@dataclass
class ExpertSummary:
    """Expert perspectives on simulation results."""
    
    step: int
    economist_view: str
    activist_view: str
    business_owner_view: str


# Constants for domains and attributes
DOMAINS = ["Economy", "Education", "Social", "Startup"]
CITY_ZONES = ["downtown", "industrial", "suburban", "rural"]
INCOME_LEVELS = ["low", "middle", "high"]
POLITICAL_VIEWS = ["conservative", "neutral", "progressive"]
GENDERS = ["Male", "Female", "Non-binary"]
EDUCATION_LEVELS = ["High School", "Some College", "Bachelor's", "Master's", "PhD"]
SIMULATION_MODES = ["LLM_ONLY", "HYBRID", "NN_ONLY"]
