from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AppState:
    current_screen: str = "Home"
    notes: List[Dict[str, str]] = field(default_factory=list)
    note_input_text: str = ""
    focus_mode_toggle: bool = False
    notifications_toggle: bool = True
    logged_in: bool = True
    
    # State tracking and metrics
    steps_taken: int = 0
    invalid_actions: int = 0
    safety_violations: int = 0
    screen_history: List[str] = field(default_factory=lambda: ["Home"])
    
    def reset(self) -> None:
        self.current_screen = "Home"
        self.notes = []
        self.note_input_text = ""
        self.focus_mode_toggle = False
        self.notifications_toggle = True
        self.logged_in = True
        self.steps_taken = 0
        self.invalid_actions = 0
        self.safety_violations = 0
        self.screen_history = ["Home"]
