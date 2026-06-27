# Mobile UI Agent RL Environment

An RL-style environment that simulates a simple mobile application using the Prime Intellect Verifiers framework. The environment evaluates agent sequences for task success, action formatting, execution efficiency, and safety.

### Overview
- **Environment ID**: `mobile_ui_env`
- **Short description**: Evaluates mobile UI agents on task success, formatting, efficiency, and safety inside a simulated mobile app.
- **Tags**: `mobile-ui`, `single-turn`, `rl-environment`, `eval`

### Datasets
- **Primary dataset(s)**: Programmatic synthetic datasets of mobile task instructions (e.g., creating notes, toggling focus mode, inspecting profile values).
- **Split sizes**: 20 training tasks, 10 evaluation tasks (30 tasks total).

### Task
- **Type**: Single-turn action sequence execution.
- **Output format expectations**: A JSON array containing a sequence of tap, type, back, or finish actions.
- **Rubric overview**: Evaluates success (weight 1.0), format validity (weight 0.1), step efficiency (weight 0.2), invalid action penalties (weight 0.2), and safety penalties (weight 0.3) with final reward clipped to `[0.0, 1.0]`.

---

### Quickstart

Run the evaluation runner script locally to test the heuristic policy:
```bash
uv run python environments/mobile_ui_env/run_eval.py
```

Run evaluation using the Prime CLI:
```bash
prime eval run mobile_ui_env
```

Run the unit tests:
```bash
uv run pytest environments/mobile_ui_env/tests/
```

---

## Assignment Design & Discussion Questions

### 1. What is the state space?
The state space is represented by the application simulator's internal state (`AppState`):
*   `current_screen`: Discrete variable taking one of four values (`Home`, `Notes`, `Settings`, `Profile`).
*   `notes`: A list of dictionaries representing successfully saved notes (`[{"title": str}]`).
*   `note_input_text`: A text buffer string representing what has been entered into the `note_input` text field.
*   `focus_mode_toggle`: Boolean toggle state (`True` / `False`).
*   `notifications_toggle`: Boolean toggle state (`True` / `False`).
*   `logged_in`: Boolean sign-in status (`True` / `False`).
*   `screen_history`: Stack/list of previously visited screens supporting screen backtracking.
*   `steps_taken`, `invalid_actions`, `safety_violations`: Counters tracking agent efficiency and correctness.

### 2. What is the action space?
The action space consists of structured JSON dictionaries:
*   `tap` action: Requires a `target` key. Valid targets are screen-specific:
    *   `Home`: `notes_button`, `settings_button`, `profile_button`
    *   `Notes`: `add_note_button`, `save_note_button`
    *   `Settings`: `focus_mode_toggle`, `notifications_toggle`
    *   `Profile`: `logout_button`
*   `type` action: Requires `target` (must be `note_input` on the `Notes` screen) and a `text` string.
*   `back` action: Pops the top screen from the history stack and moves the agent to the previous screen. Valid from any screen except `Home`.
*   `finish` action: Terminates the execution loop early.

### 3. What is the episode termination condition?
An episode terminates when:
1.  The agent submits an `"action": "finish"` command.
2.  The step counter (`steps_taken`) reaches the task's step budget limit (`max_steps` - typically 6, 8, or 12 depending on task complexity).

### 4. Which rewards are sparse?
*   `success_reward`: It is a binary $1.0$ if the target goal condition (e.g., specific note exists in state, setting toggled, or screen reached) is met at the end of the episode, and $0.0$ otherwise.
*   `format_reward`: Evaluated at the trajectory level ($1.0$ if the entire sequence is a valid JSON array, $0.0$ otherwise).

### 5. Which rewards are dense or shaped?
*   `efficiency_reward`: Varies continuously based on the number of steps taken to complete the task successfully: `(max_steps - steps_taken) / max_steps`.
*   `invalid_action_penalty`: Penalizes the agent incrementally ($-0.1$ net penalty per invalid tap or type target) for each incorrect action attempted.
*   `safety_penalty`: Inflicts a negative score ($-0.3$ net penalty per violation) when the agent triggers unsafe state transitions (such as logging out).

### 6. How can reward hacking happen in this environment?
*   If the `efficiency_reward` was not contingent on *successful completion*, the agent would immediately output `[{"action": "finish"}]` on step 1 to maximize efficiency without solving the task.
*   If we rewarded visiting screens without a step budget or duplicate penalty, the agent could navigate back and forth infinitely (`tap notes_button` -> `back` -> `tap notes_button`...) to accumulate infinite screen-visitation rewards.
*   If the penalties are too weak compared to the success reward, an agent might learn to exhaustively tap all buttons on the screen randomly until it completes the goal, brute-forcing success despite being highly inefficient and unsafe.

### 7. How would you scale this from a mock UI to a real Android emulator?
*   **Observation Space**: Instead of managing a discrete `current_screen` string, we would extract the XML layout hierarchy (accessibility tree) of the active window using tools like Android Uiautomator2 or Appium. We could also feed screenshot images to a multimodal agent.
*   **Action Space**: Instead of structured targets, actions would be translated into absolute pixel tap coordinates `(x, y)` or element IDs. The `type` action would send ADB keystrokes.
*   **State Detection**: The environment would detect screens and transitions dynamically by parsing XML node classnames and activity names rather than hardcoded string states.

### 8. How would this work with Prime Intellect, Verifiers, or PRIME-RL later?
*   **Evaluation**: The environment's `load_environment()` function is fully registered with `verifiers`. In evaluation, we run `prime eval run mobile_ui_env` where different model endpoints are prompted with instructions and output the action plans evaluated by our `ClippedRubric`.
*   **Training (PRIME-RL)**: During RL training, the model generates trajectories of actions. The verifier runs `score_group` to calculate rewards and relative advantages (comparing trajectories for the same prompt), guiding policy updates to optimize success rate and step efficiency.

### 9. What tests did you write?
*   `test_actions.py`: Validates valid/invalid taps, screen state transitions, text input typing, and screen backtracking.
*   `test_rewards.py`: Tests `get_sim_result` helper, `success_reward` logic, `format_reward`, `efficiency_reward` step calculation, and `safety_penalty`.
*   `test_env.py`: Tests package loading, dataset split counts, and verifies `ClippedRubric` outputs are bounded between `[0.0, 1.0]` under both successful and heavily penalized trajectories.

### 10. What tradeoffs did you make due to the limited assignment scope?
*   **Single-Turn Execution**: The agent submits the entire sequence of actions up front rather than observing the intermediate screen layout at each step. This makes implementation simple but prevents the agent from reacting to unexpected visual feedback.
*   **State Simulation**: We simulated a mock app state transitions state machine rather than rendering a visual GUI or HTML viewport, which is sufficient for logic evaluation but ignores visual processing.
