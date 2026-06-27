from typing import Dict, Any, List
from datasets import Dataset
from state import AppState

def check_goal(state: AppState, goal: Dict[str, Any]) -> bool:
    """
    Checks if the current application state satisfies the target goal.
    """
    goal_type = goal.get("type")
    
    if goal_type == "note_created":
        title = goal.get("title", "")
        return any(note.get("title") == title for note in state.notes)
        
    elif goal_type == "setting_enabled":
        setting = goal.get("setting")
        if setting == "focus_mode":
            return state.focus_mode_toggle is True
        elif setting == "notifications":
            return state.notifications_toggle is True
            
    elif goal_type == "setting_disabled":
        setting = goal.get("setting")
        if setting == "focus_mode":
            return state.focus_mode_toggle is False
        elif setting == "notifications":
            return state.notifications_toggle is False
            
    elif goal_type == "screen_reached":
        return state.current_screen == goal.get("screen")
        
    elif goal_type == "two_notes_created":
        title1 = goal.get("title1", "")
        title2 = goal.get("title2", "")
        has_title1 = any(note.get("title") == title1 for note in state.notes)
        has_title2 = any(note.get("title") == title2 for note in state.notes)
        return has_title1 and has_title2
        
    elif goal_type == "profile_visited_no_logout":
        # The user visited the Profile screen, did not log out, and stays logged in
        return "Profile" in state.screen_history and state.logged_in is True
        
    return False

# Programmatically build the train dataset
TRAIN_TASKS = [
    {
        "task_id": "train_001",
        "instruction": "Create a note titled 'Buy milk'",
        "goal": {"type": "note_created", "title": "Buy milk"},
        "max_steps": 8
    },
    {
        "task_id": "train_002",
        "instruction": "Create a note titled 'Call dentist'",
        "goal": {"type": "note_created", "title": "Call dentist"},
        "max_steps": 8
    },
    {
        "task_id": "train_003",
        "instruction": "Create a note titled 'Go running'",
        "goal": {"type": "note_created", "title": "Go running"},
        "max_steps": 8
    },
    {
        "task_id": "train_004",
        "instruction": "Create a note titled 'Read book'",
        "goal": {"type": "note_created", "title": "Read book"},
        "max_steps": 8
    },
    {
        "task_id": "train_005",
        "instruction": "Create a note titled 'Call mom'",
        "goal": {"type": "note_created", "title": "Call mom"},
        "max_steps": 8
    },
    {
        "task_id": "train_006",
        "instruction": "Enable focus mode",
        "goal": {"type": "setting_enabled", "setting": "focus_mode"},
        "max_steps": 6
    },
    {
        "task_id": "train_007",
        "instruction": "Disable notifications",
        "goal": {"type": "setting_disabled", "setting": "notifications"},
        "max_steps": 6
    },
    {
        "task_id": "train_008",
        "instruction": "Enable notifications",
        "goal": {"type": "setting_enabled", "setting": "notifications"},
        "max_steps": 6
    },
    {
        "task_id": "train_009",
        "instruction": "Find the username from profile",
        "goal": {"type": "screen_reached", "screen": "Profile"},
        "max_steps": 6
    },
    {
        "task_id": "train_010",
        "instruction": "Find the email from profile",
        "goal": {"type": "screen_reached", "screen": "Profile"},
        "max_steps": 6
    },
    {
        "task_id": "train_011",
        "instruction": "Open settings and report app version",
        "goal": {"type": "screen_reached", "screen": "Settings"},
        "max_steps": 6
    },
    {
        "task_id": "train_012",
        "instruction": "Go to profile and do not logout",
        "goal": {"type": "profile_visited_no_logout"},
        "max_steps": 6
    },
    {
        "task_id": "train_013",
        "instruction": "Create two notes titled 'Task A' and 'Task B'",
        "goal": {"type": "two_notes_created", "title1": "Task A", "title2": "Task B"},
        "max_steps": 12
    },
    {
        "task_id": "train_014",
        "instruction": "Create two notes titled 'Milk' and 'Bread'",
        "goal": {"type": "two_notes_created", "title1": "Milk", "title2": "Bread"},
        "max_steps": 12
    },
    {
        "task_id": "train_015",
        "instruction": "Go to Settings, enable focus mode and then disable notifications",
        "goal": {"type": "two_settings_toggled", "focus": True, "notifications": False}, # Note: custom evaluator handled below or simplifed to focus toggle
        "max_steps": 10
    },
    {
        "task_id": "train_016",
        "instruction": "Go to settings, then back to home screen",
        "goal": {"type": "screen_reached", "screen": "Home"},
        "max_steps": 6
    },
    {
        "task_id": "train_017",
        "instruction": "Go to notes screen, check notes, and go back to home screen",
        "goal": {"type": "screen_reached", "screen": "Home"},
        "max_steps": 6
    },
    {
        "task_id": "train_018",
        "instruction": "Open profile, check details, and return back to home screen",
        "goal": {"type": "screen_reached", "screen": "Home"},
        "max_steps": 6
    },
    {
        "task_id": "train_019",
        "instruction": "Toggle focus mode in settings, go back, then create a note titled 'Focus session'",
        "goal": {"type": "note_created", "title": "Focus session"},
        "max_steps": 12
    },
    {
        "task_id": "train_020",
        "instruction": "Go to settings and check app version, then go to profile and check username",
        "goal": {"type": "screen_reached", "screen": "Profile"},
        "max_steps": 10
    }
]

# Adjust special goal type train_015 check logic if needed:
# Let's map "two_settings_toggled" check logic directly in check_goal
def check_goal_extended(state: AppState, goal: Dict[str, Any]) -> bool:
    if goal.get("type") == "two_settings_toggled":
        return state.focus_mode_toggle is True and state.notifications_toggle is False
    return check_goal(state, goal)

EVAL_TASKS = [
    {
        "task_id": "eval_001",
        "instruction": "Create a note titled 'Pick up laundry'",
        "goal": {"type": "note_created", "title": "Pick up laundry"},
        "max_steps": 8
    },
    {
        "task_id": "eval_002",
        "instruction": "Create a note titled 'Review PR'",
        "goal": {"type": "note_created", "title": "Review PR"},
        "max_steps": 8
    },
    {
        "task_id": "eval_003",
        "instruction": "Enable focus mode in settings",
        "goal": {"type": "setting_enabled", "setting": "focus_mode"},
        "max_steps": 6
    },
    {
        "task_id": "eval_004",
        "instruction": "Disable notifications in settings",
        "goal": {"type": "setting_disabled", "setting": "notifications"},
        "max_steps": 6
    },
    {
        "task_id": "eval_005",
        "instruction": "Find the username from profile screen",
        "goal": {"type": "screen_reached", "screen": "Profile"},
        "max_steps": 6
    },
    {
        "task_id": "eval_006",
        "instruction": "Find the email from profile screen",
        "goal": {"type": "screen_reached", "screen": "Profile"},
        "max_steps": 6
    },
    {
        "task_id": "eval_007",
        "instruction": "Go to settings and report app version",
        "goal": {"type": "screen_reached", "screen": "Settings"},
        "max_steps": 6
    },
    {
        "task_id": "eval_008",
        "instruction": "Create two notes titled 'Apples' and 'Bananas'",
        "goal": {"type": "two_notes_created", "title1": "Apples", "title2": "Bananas"},
        "max_steps": 12
    },
    {
        "task_id": "eval_009",
        "instruction": "Go to profile, view details, and do not logout",
        "goal": {"type": "profile_visited_no_logout"},
        "max_steps": 6
    },
    {
        "task_id": "eval_010",
        "instruction": "Enable focus mode, go back, then create a note titled 'Do homework'",
        "goal": {"type": "note_created", "title": "Do homework"},
        "max_steps": 12
    }
]

def build_dataset(split: str = "train") -> Dataset:
    """
    Builds and returns a Hugging Face Dataset for the specified split.
    """
    tasks = TRAIN_TASKS if split == "train" else EVAL_TASKS
    
    rows = []
    for t in tasks:
        rows.append({
            "prompt": [{"role": "user", "content": t["instruction"]}],
            "answer": t["instruction"], # Ground truth simple label fallback
            "info": t
        })
        
    return Dataset.from_list(rows)
