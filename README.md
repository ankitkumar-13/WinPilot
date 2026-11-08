# WinPilot
Windows Desktop Automation System

## Command interpretation

WinPilot tries command interpretation in this order:

1. The deterministic parser handles clear, supported command patterns.
2. If the parser cannot interpret a command, the optional local intent and
   entity models in `nlp/` try it. Predictions below the confidence thresholds
   or missing required entities are treated as uncertain.
3. Gemini handles requests the local stages cannot interpret.

Install the optional local-model dependencies with `uv sync --extra nlp`.
Without these dependencies, WinPilot skips local NLP and uses Gemini when the
deterministic parser cannot interpret a command. The local intent threshold
(0.80) and per-entity token threshold (0.70) are conservative starting values;
they should be calibrated against held-out commands before being treated as
reliability guarantees. Deletions continue to go through the MCP approval
flow.

## File operation context

Relative file and folder operations use the foreground File Explorer folder,
then the foremost visible, non-minimized Explorer folder. If neither is
available, WinPilot uses the user's Desktop. It scans every open Explorer
window, including minimized windows, for matching source names. When the same
name exists in more than one open folder, the default `ask` policy presents
numbered choices with absolute paths and marks the current topmost match. Choose
by number or paste a listed path. Set `WINPILOT_AMBIGUOUS_TARGET_POLICY=topmost`
to automatically use the matching item in the active workspace when duplicates
exist; `ask` is the default. If a match exists only in another open folder, or
the requested item is missing, WinPilot asks for clarification. Deletes still
ask for approval, then request Recycle Bin handling from Windows. Windows
preserves undo information when the target volume supports it.

The `nlp/` directory retains the earlier project's training scripts, datasets,
evaluation results, and model artifacts. `nlp/adapter.py` is the runtime bridge
used by the new architecture.
