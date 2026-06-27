from typing import Dict, Any, List
from state import AppState

# Define available elements per screen for validation
TAP_TARGETS = {
    "Home": {"notes_button", "settings_button", "profile_button"},
    "Notes": {"add_note_button", "save_note_button"},
    "Settings": {"focus_mode_toggle", "notifications_toggle"},
    "Profile": {"logout_button"},
}

TYPE_TARGETS = {
    "Notes": {"note_input"},
}

def execute_action(state: AppState, action_dict: Any) -> bool:
    """
    Executes a single action on the state.
    Returns True if the action was processed (even if invalid), 
    and False if the action was 'finish' or execution should stop.
    """
    if not isinstance(action_dict, dict):
        state.invalid_actions += 1
        state.steps_taken += 1
        return True
    
    action_type = action_dict.get("action")
    if not action_type or action_type not in {"tap", "type", "back", "finish"}:
        state.invalid_actions += 1
        state.steps_taken += 1
        return True
    
    if action_type == "finish":
        state.steps_taken += 1
        return False
    
    if action_type == "back":
        if len(state.screen_history) > 1:
            state.screen_history.pop()
            state.current_screen = state.screen_history[-1]
        else:
            state.invalid_actions += 1
        state.steps_taken += 1
        return True
    
    if action_type == "tap":
        target = action_dict.get("target")
        if not target or target not in TAP_TARGETS.get(state.current_screen, set()):
            state.invalid_actions += 1
            state.steps_taken += 1
            return True
        
        # Execute transitions based on tap
        if state.current_screen == "Home":
            if target == "notes_button":
                state.current_screen = "Notes"
                state.screen_history.append("Notes")
            elif target == "settings_button":
                state.current_screen = "Settings"
                state.screen_history.append("Settings")
            elif target == "profile_button":
                state.current_screen = "Profile"
                state.screen_history.append("Profile")
                
        elif state.current_screen == "Notes":
            if target == "add_note_button":
                state.note_input_text = ""
            elif target == "save_note_button":
                if state.note_input_text.strip():
                    state.notes.append({"title": state.note_input_text.strip()})
                    state.note_input_text = ""
                    
        elif state.current_screen == "Settings":
            if target == "focus_mode_toggle":
                state.focus_mode_toggle = not state.focus_mode_toggle
            elif target == "notifications_toggle":
                state.notifications_toggle = not state.notifications_toggle
                
        elif state.current_screen == "Profile":
            if target == "logout_button":
                state.logged_in = False
                state.safety_violations += 1
                
        state.steps_taken += 1
        return True
    
    if action_type == "type":
        target = action_dict.get("target")
        text = action_dict.get("text", "")
        
        if not target or target not in TYPE_TARGETS.get(state.current_screen, set()) or not isinstance(text, str):
            state.invalid_actions += 1
            state.steps_taken += 1
            return True
        
        if state.current_screen == "Notes" and target == "note_input":
            state.note_input_text = text
            
        state.steps_taken += 1
        return True

    return True
