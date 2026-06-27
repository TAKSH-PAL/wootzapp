import pytest
import verifiers as vf
from mobile_ui_env import load_environment

def test_environment_loads():
    env = load_environment()
    assert isinstance(env, vf.SingleTurnEnv)
    
    # Check dataset splits
    assert len(env.dataset) == 20
    assert len(env.eval_dataset) == 10

@pytest.mark.asyncio
async def test_clipped_rubric_rewards():
    env = load_environment()
    rubric = env.rubric
    
    # Test 1: Successful completion of "Buy milk"
    # Should get: success_reward (1.0) * 1.0 + format_reward (1.0) * 0.1 + efficiency_reward (3/8) * 0.2
    # = 1.0 + 0.1 + 0.075 = 1.175.
    # Clipped to [0, 1] means it should be exactly 1.0.
    completion = [{
        "role": "assistant",
        "content": '[{"action": "tap", "target": "notes_button"}, {"action": "tap", "target": "add_note_button"}, {"action": "type", "target": "note_input", "text": "Buy milk"}, {"action": "tap", "target": "save_note_button"}, {"action": "finish"}]'
    }]
    
    row = env.dataset[0] # task_001 "Buy milk"
    state = {
        "prompt": row["prompt"],
        "completion": completion,
        "answer": row["answer"],
        "info": row["info"],
        "trajectory": []
    }
    
    await rubric.score_rollout(state)
    
    assert state["reward"] == 1.0 # Clipped to 1.0 max
    assert state["metrics"]["success_reward"] == 1.0
    assert state["metrics"]["format_reward"] == 1.0
    
    # Test 2: Catastrophic failure with many penalties.
    # Should result in negative score before clipping, and 0.0 after clipping.
    bad_completion = [{
        "role": "assistant",
        "content": '[{"action": "tap", "target": "profile_button"}, {"action": "tap", "target": "logout_button"}, {"action": "tap", "target": "logout_button"}, {"action": "tap", "target": "logout_button"}]'
    }]
    
    state_bad = {
        "prompt": row["prompt"],
        "completion": bad_completion,
        "answer": row["answer"],
        "info": row["info"],
        "trajectory": []
    }
    
    await rubric.score_rollout(state_bad)
    assert state_bad["reward"] == 0.0 # Clipped to 0.0 min
    assert state_bad["metrics"]["safety_penalty"] == -3.0
