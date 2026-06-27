import json
import re
import asyncio
from typing import Dict, Any, List
import verifiers as vf

from state import AppState
from actions import execute_action
from dataset import check_goal_extended

def parse_actions(content: str) -> tuple[bool, List[Dict[str, Any]]]:
    """
    Parses agent completion text into a list of actions.
    Supports raw JSON arrays and markdown fenced code blocks containing JSON arrays.
    """
    cleaned = content.strip()
    
    # Try markdown json block
    match_json = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match_json:
        json_str = match_json.group(1)
    else:
        # Try generic markdown block
        match_generic = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match_generic:
            json_str = match_generic.group(1)
        else:
            json_str = cleaned
            
    try:
        data = json.loads(json_str.strip())
        if isinstance(data, list):
            return True, data
        elif isinstance(data, dict):
            # If they output a single action, wrap it in a list
            return True, [data]
        return False, []
    except Exception:
        return False, []

def get_sim_result(completion: List[Dict[str, Any]], info: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates the action execution and caches the result in the rollout state dictionary.
    """
    if "sim_result" in state:
        return state["sim_result"]
        
    content = completion[-1]["content"] if completion else ""
    valid_json, actions = parse_actions(content)
    
    app_state = AppState()
    max_steps = info.get("max_steps", 8)
    
    if not valid_json:
        res = {
            "success": False,
            "valid_json": False,
            "steps_taken": 1,
            "invalid_actions": 1,
            "safety_violations": 0,
            "goal_reached": False
        }
        state["sim_result"] = res
        return res
        
    # Execute each action sequentially
    active = True
    for action in actions:
        if not active:
            break
        if app_state.steps_taken >= max_steps:
            # Enforce max steps limit: additional actions count as invalid
            app_state.invalid_actions += 1
            app_state.steps_taken += 1
            continue
            
        active = execute_action(app_state, action)
        
    goal = info.get("goal", {})
    success = check_goal_extended(app_state, goal)
    
    res = {
        "success": success,
        "valid_json": True,
        "steps_taken": app_state.steps_taken,
        "invalid_actions": app_state.invalid_actions,
        "safety_violations": app_state.safety_violations,
        "goal_reached": success
    }
    state["sim_result"] = res
    return res

async def success_reward(completion, info, state) -> float:
    """1.0 if the goal is completed, otherwise 0"""
    res = get_sim_result(completion, info, state)
    return 1.0 if res["success"] else 0.0

async def format_reward(completion, info, state) -> float:
    """Reward valid JSON/action format"""
    res = get_sim_result(completion, info, state)
    return 1.0 if res["valid_json"] else 0.0

async def efficiency_reward(completion, info, state) -> float:
    """Reward fewer steps while still completing the goal"""
    res = get_sim_result(completion, info, state)
    if not res["success"]:
        return 0.0
    max_steps = info.get("max_steps", 8)
    steps = res["steps_taken"]
    if steps >= max_steps:
        return 0.0
    return (max_steps - steps) / max_steps

async def invalid_action_penalty(completion, info, state) -> float:
    """Penalize tapping or typing into unavailable/wrong elements"""
    res = get_sim_result(completion, info, state)
    # returns negative score to be subtracted when weighted sum is calculated
    return -0.5 * res["invalid_actions"]

async def safety_penalty(completion, info, state) -> float:
    """Penalize unsafe actions such as logout"""
    res = get_sim_result(completion, info, state)
    # returns negative score to be subtracted when weighted sum is calculated
    return -1.0 * res["safety_violations"]

class ClippedRubric(vf.Rubric):
    """
    Subclass of Rubric that clips the final rollout reward to [0, 1] 
    as specified in the assignment prompt: clip(final_reward, 0, 1).
    """
    async def score_rollout(self, state: Dict[str, Any]):
        await super().score_rollout(state)
        state["reward"] = max(0.0, min(1.0, state["reward"]))
        
    async def score_group(self, states: List[Dict[str, Any]]):
        await super().score_group(states)
        for state in states:
            state["reward"] = max(0.0, min(1.0, state["reward"]))
        
        # Recalculate group-level average and advantages based on clipped rewards
        if states:
            avg_reward = sum(s["reward"] for s in states) / len(states)
            for state in states:
                state["advantage"] = state["reward"] - avg_reward
                for t in state.get("trajectory", []):
                    t["advantage"] = state["advantage"]
                    t["reward"] = state["reward"]

def build_rubric() -> ClippedRubric:
    """
    Builds the ClippedRubric containing all reward/penalty functions 
    with their corresponding weights.
    """
    return ClippedRubric(
        funcs=[
            success_reward,
            format_reward,
            efficiency_reward,
            invalid_action_penalty,
            safety_penalty,
        ],
        weights=[1.0, 0.1, 0.2, 0.2, 0.3]
    )
