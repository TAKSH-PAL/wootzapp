import json
import re
import asyncio
from typing import List, Dict, Any

from mobile_ui_env import load_environment

def heuristic_policy(instruction: str) -> List[Dict[str, Any]]:
    """
    Heuristic rule-based policy that acts as an agent to solve tasks.
    Occasionally includes some realistic navigation paths to test invalid actions/backtracking.
    """
    instr = instruction.lower()
    
    if "pick up laundry" in instr:
        # Perfect execution for note creation
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Pick up laundry"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"}
        ]
        
    elif "review pr" in instr:
        # Includes a mistake (trying to tap version_label) to demonstrate invalid action penalties
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "version_label"}, # INVALID: version_label is not a button and on Home/Notes? (Actually on Settings, so invalid)
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Review PR"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"}
        ]
        
    elif "enable focus mode" in instr and "homework" in instr:
        # Navigate to Settings -> Toggle Focus -> Back -> Notes -> Add Note -> Save -> Finish
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "back"},
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Do homework"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"}
        ]
        
    elif "enable focus mode" in instr:
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "focus_mode_toggle"},
            {"action": "finish"}
        ]
        
    elif "disable notifications" in instr:
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "tap", "target": "notifications_toggle"},
            {"action": "finish"}
        ]
        
    elif "username" in instr or "email" in instr:
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "finish"}
        ]
        
    elif "app version" in instr:
        return [
            {"action": "tap", "target": "settings_button"},
            {"action": "finish"}
        ]
        
    elif "apples" in instr and "bananas" in instr:
        return [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Apples"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Bananas"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"}
        ]
        
    elif "do not logout" in instr:
        # Navigate to Profile, but accidentally click logout first (triggering safety violation), then finish.
        # This will demonstrate the safety penalty evaluation!
        return [
            {"action": "tap", "target": "profile_button"},
            {"action": "tap", "target": "logout_button"}, # SAFETY VIOLATION
            {"action": "finish"}
        ]
        
    # Fallback dummy output
    return [{"action": "finish"}]

async def run_evaluation():
    print("Initializing Mobile UI RL Environment...")
    env = load_environment()
    eval_dataset = env.eval_dataset
    rubric = env.rubric
    
    total_tasks = len(eval_dataset)
    successes = 0
    total_reward = 0.0
    total_steps = 0
    total_invalid = 0
    total_safety_violations = 0
    total_actions_proposed = 0
    
    print(f"Running evaluation over {total_tasks} tasks...\n")
    print(f"{'Task ID':<10} | {'Instruction':<60} | {'Reward':<6} | {'Success':<7} | {'Steps':<5} | {'Invalid':<7} | {'Safety':<6}")
    print("-" * 115)
    
    for row in eval_dataset:
        info = row["info"]
        instruction = info["instruction"]
        task_id = info["task_id"]
        
        # Generate completion from heuristic policy
        actions = heuristic_policy(instruction)
        completion_content = json.dumps(actions)
        completion = [{"role": "assistant", "content": completion_content}]
        
        # Setup rollout state
        state = {
            "prompt": row["prompt"],
            "completion": completion,
            "answer": row["answer"],
            "info": info,
            "trajectory": [] # required by Rubric scoring internal structure
        }
        
        # Score the rollout
        await rubric.score_rollout(state)
        
        sim_result = state.get("sim_result", {})
        reward = state.get("reward", 0.0)
        success = sim_result.get("success", False)
        steps = sim_result.get("steps_taken", 0)
        invalid = sim_result.get("invalid_actions", 0)
        safety = sim_result.get("safety_violations", 0)
        
        successes += 1 if success else 0
        total_reward += reward
        total_steps += steps
        total_invalid += invalid
        total_safety_violations += safety
        total_actions_proposed += len(actions)
        
        success_str = "Yes" if success else "No"
        print(f"{task_id:<10} | {instruction[:60]:<60} | {reward:<6.2f} | {success_str:<7} | {steps:<5} | {invalid:<7} | {safety:<6}")

    success_rate = (successes / total_tasks) * 100
    avg_reward = total_reward / total_tasks
    avg_steps = total_steps / total_tasks
    invalid_action_rate = total_invalid / max(1, total_actions_proposed)
    
    print("\n" + "=" * 50)
    print("EVALUATION METRICS REPORT")
    print("=" * 50)
    print(f"Total eval tasks:      {total_tasks}")
    print(f"Success rate:          {success_rate:.1f}%")
    print(f"Average reward:        {avg_reward:.2f}")
    print(f"Average steps:         {avg_steps:.1f}")
    print(f"Invalid action rate:   {invalid_action_rate:.2f}")
    print(f"Safety violations:     {total_safety_violations}")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
