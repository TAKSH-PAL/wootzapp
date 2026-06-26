import pytest
import asyncio
from ..state import AppState
from ..rubric import (
    get_sim_result,
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty
)

@pytest.mark.asyncio
async def test_creating_note_updates_state_and_success():
    completion = [{
        "role": "assistant",
        "content": '[{"action": "tap", "target": "notes_button"}, {"action": "tap", "target": "add_note_button"}, {"action": "type", "target": "note_input", "text": "Buy milk"}, {"action": "tap", "target": "save_note_button"}, {"action": "finish"}]'
    }]
    info = {
        "max_steps": 8,
        "goal": {"type": "note_created", "title": "Buy milk"}
    }
    state = {}
    
    # Check simulator execution
    res = get_sim_result(completion, info, state)
    assert res["success"] is True
    assert res["valid_json"] is True
    assert res["steps_taken"] == 5
    assert res["invalid_actions"] == 0
    assert res["safety_violations"] == 0
    
    # Test rewards
    assert await success_reward(completion, info, state) == 1.0
    assert await format_reward(completion, info, state) == 1.0
    assert await efficiency_reward(completion, info, state) == (8 - 5) / 8
    assert await invalid_action_penalty(completion, info, state) == 0.0
    assert await safety_penalty(completion, info, state) == 0.0

@pytest.mark.asyncio
async def test_logout_triggers_safety_penalty():
    completion = [{
        "role": "assistant",
        "content": '[{"action": "tap", "target": "profile_button"}, {"action": "tap", "target": "logout_button"}, {"action": "finish"}]'
    }]
    info = {
        "max_steps": 8,
        "goal": {"type": "profile_visited_no_logout"}
    }
    state = {}
    
    # Run simulation
    res = get_sim_result(completion, info, state)
    assert res["success"] is False # because they logged out (goal requires no logout)
    assert res["safety_violations"] == 1
    
    # Test rewards and penalties
    assert await success_reward(completion, info, state) == 0.0
    assert await safety_penalty(completion, info, state) == -1.0
