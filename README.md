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

The `nlp/` directory retains the earlier project's training scripts, datasets,
evaluation results, and model artifacts. `nlp/adapter.py` is the runtime bridge
used by the new architecture.
