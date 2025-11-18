"""
Statistics computation module for CIVI-GENESIS.
Aggregate and analyze simulation step data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from data_models import Citizen, CitizenState, StepStats


def compute_step_stats(
    citizens: List[Citizen],
    states_for_step: List[CitizenState]
) -> StepStats:
    """
    Compute aggregated statistics for a single simulation step.
    
    Args:
        citizens: List of all citizens in the population.
        states_for_step: List of CitizenState objects for this step.
    
    Returns:
        StepStats object with aggregated metrics.
    """
    if not states_for_step:
        # Return empty stats if no states
        return StepStats(
            step=0,
            avg_happiness=0.0,
            avg_support=0.0,
            avg_income=0.0,
            by_income=[],
            by_zone=[],
            inequality_gap_happiness=0.0,
            inequality_gap_support=0.0
        )
    
    step = states_for_step[0].step
    
    # Create a map from citizen_id to citizen
    citizen_map = {c.id: c for c in citizens}
    
    # Overall averages
    avg_happiness = np.mean([s.happiness for s in states_for_step])
    avg_support = np.mean([s.policy_support for s in states_for_step])
    avg_income = np.mean([s.income for s in states_for_step])
    
    # Group by income level
    by_income_dict = {}
    for state in states_for_step:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            income_level = citizen.income_level
            if income_level not in by_income_dict:
                by_income_dict[income_level] = {"happiness": [], "support": []}
            by_income_dict[income_level]["happiness"].append(state.happiness)
            by_income_dict[income_level]["support"].append(state.policy_support)
    
    by_income = []
    for income_level in ["low", "middle", "high"]:
        if income_level in by_income_dict:
            data = by_income_dict[income_level]
            by_income.append({
                "income_level": income_level,
                "avg_happiness": float(np.mean(data["happiness"])),
                "avg_support": float(np.mean(data["support"]))
            })
    
    # Group by city zone
    by_zone_dict = {}
    for state in states_for_step:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            zone = citizen.city_zone
            if zone not in by_zone_dict:
                by_zone_dict[zone] = {"happiness": [], "support": []}
            by_zone_dict[zone]["happiness"].append(state.happiness)
            by_zone_dict[zone]["support"].append(state.policy_support)
    
    by_zone = []
    for zone, data in by_zone_dict.items():
        by_zone.append({
            "city_zone": zone,
            "avg_happiness": float(np.mean(data["happiness"])),
            "avg_support": float(np.mean(data["support"]))
        })
    
    # Compute inequality gaps
    high_income_stats = next((x for x in by_income if x["income_level"] == "high"), None)
    low_income_stats = next((x for x in by_income if x["income_level"] == "low"), None)
    
    inequality_gap_happiness = 0.0
    inequality_gap_support = 0.0
    
    if high_income_stats and low_income_stats:
        inequality_gap_happiness = high_income_stats["avg_happiness"] - low_income_stats["avg_happiness"]
        inequality_gap_support = high_income_stats["avg_support"] - low_income_stats["avg_support"]
    
    return StepStats(
        step=step,
        avg_happiness=float(avg_happiness),
        avg_support=float(avg_support),
        avg_income=float(avg_income),
        by_income=by_income,
        by_zone=by_zone,
        inequality_gap_happiness=float(inequality_gap_happiness),
        inequality_gap_support=float(inequality_gap_support)
    )


def build_stats_dataframe(
    all_states: List[CitizenState],
    citizens: List[Citizen]
) -> pd.DataFrame:
    """
    Build a pandas DataFrame with all states for analysis.
    
    Args:
        all_states: List of all CitizenState objects across all steps.
        citizens: List of all citizens.
    
    Returns:
        DataFrame with columns: step, citizen_id, happiness, policy_support, income,
                                income_level, city_zone, llm_updated, etc.
    """
    citizen_map = {c.id: c for c in citizens}
    
    rows = []
    for state in all_states:
        citizen = citizen_map.get(state.citizen_id)
        if citizen:
            rows.append({
                "step": state.step,
                "citizen_id": state.citizen_id,
                "happiness": state.happiness,
                "policy_support": state.policy_support,
                "income": state.income,
                "income_level": citizen.income_level,
                "city_zone": citizen.city_zone,
                "age": citizen.age,
                "profession": citizen.profession,
                "political_view": citizen.political_view,
                "llm_updated": state.llm_updated,
                "has_diary": state.diary_entry is not None
            })
    
    return pd.DataFrame(rows)


def compute_time_series_stats(
    all_states: List[CitizenState],
    citizens: List[Citizen],
    max_step: int
) -> List[StepStats]:
    """
    Compute statistics for all steps in the simulation.
    
    Args:
        all_states: List of all CitizenState objects.
        citizens: List of all citizens.
        max_step: Maximum step number.
    
    Returns:
        List of StepStats objects, one per step.
    """
    # Group states by step
    states_by_step = {}
    for state in all_states:
        if state.step not in states_by_step:
            states_by_step[state.step] = []
        states_by_step[state.step].append(state)
    
    # Compute stats for each step
    time_series = []
    for step in range(max_step + 1):
        states = states_by_step.get(step, [])
        stats = compute_step_stats(citizens, states)
        time_series.append(stats)
    
    return time_series
