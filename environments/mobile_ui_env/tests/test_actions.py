import pytest
from ..state import AppState
from ..actions import execute_action

def test_valid_tap_changes_screen():
    state = AppState()
    assert state.current_screen == "Home"
    assert state.screen_history == ["Home"]
    
    # Tap notes button
    active = execute_action(state, {"action": "tap", "target": "notes_button"})
    assert active is True
    assert state.current_screen == "Notes"
    assert state.screen_history == ["Home", "Notes"]
    assert state.steps_taken == 1
    assert state.invalid_actions == 0

def test_invalid_tap_does_not_crash():
    state = AppState()
    assert state.current_screen == "Home"
    
    # Tapping add_note_button on Home screen is invalid but shouldn't crash
    active = execute_action(state, {"action": "tap", "target": "add_note_button"})
    assert active is True
    assert state.current_screen == "Home"
    assert state.invalid_actions == 1
    assert state.steps_taken == 1

def test_type_updates_input_text():
    state = AppState()
    # Go to notes screen first
    execute_action(state, {"action": "tap", "target": "notes_button"})
    
    # Type text
    active = execute_action(state, {"action": "type", "target": "note_input", "text": "Buy milk"})
    assert active is True
    assert state.note_input_text == "Buy milk"
    assert state.invalid_actions == 0
    assert state.steps_taken == 2

def test_back_navigation():
    state = AppState()
    execute_action(state, {"action": "tap", "target": "notes_button"}) # Go to Notes
    execute_action(state, {"action": "tap", "target": "add_note_button"}) # Reset input
    
    # Execute back
    active = execute_action(state, {"action": "back"})
    assert active is True
    assert state.current_screen == "Home"
    assert state.screen_history == ["Home"]
    assert state.steps_taken == 3
