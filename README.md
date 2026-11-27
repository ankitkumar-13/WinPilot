# WinPilot

WinPilot is an AI-assisted Windows desktop automation tool. Give it a supported request in plain language to launch applications, manage files and folders, type text, or work with LibreOffice documents.

WinPilot tries a deterministic command parser first, then local NLP models when available, and finally Gemini when the local stages cannot interpret a request.

## Features

- Launch installed Windows applications by name or common alias.
- Create, copy, move, rename, and delete files and folders.
- Type text into the active window or a specified application.
- Create and manage LibreOffice documents and spreadsheets, and export supported documents as PDF.
- Use open File Explorer windows to resolve file-operation locations.
- Require confirmation before deleting; approved deletions are sent to the Windows Recycle Bin.

## Requirements

- Windows
- Python 3.14 or later
- A Gemini API key
- LibreOffice installed for LibreOffice features

## Installation

Install [uv](https://docs.astral.sh/uv/) if needed, then clone the repository and install its dependencies:

```powershell
git clone https://github.com/ankitkumar-13/WinPilot.git
cd WinPilot
uv sync
```

Create a `.env` file in the project root and add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Keep your API key private. Do not commit your `.env` file.

## Run

Start WinPilot from the project directory:

```powershell
uv run python -m app.main
```

WinPilot opens an interactive command prompt. Type `exit` or `quit` to close it.

## Example requests

- `Open Notepad`
- `Create a folder named Projects`
- `Move report.docx to Archive`
- `Rename the file notes.txt to meeting-notes.txt`
- `Type Hello there into Notepad`
- `Create a new Writer document`
- `Save this document as report.odt`
- `Export as PDF`

Supported requests depend on the command parser and the tools exposed by the MCP servers. Deleting a file or folder requires confirmation.

## File and folder operations

Relative paths use the foreground File Explorer folder when available, otherwise the foremost visible, non-minimized Explorer folder. If neither is available, WinPilot uses the Desktop.

If a source name matches items in multiple open Explorer folders, WinPilot asks you to choose from numbered paths by default. To automatically use the matching item in the active workspace instead, set:

```text
WINPILOT_AMBIGUOUS_TARGET_POLICY=topmost
```

If an item is found only in another open folder, or the requested item cannot be found, WinPilot asks for clarification.

## Command interpretation

WinPilot processes requests in this order:

1. The deterministic parser handles supported command patterns.
2. Local intent and entity models try requests the parser does not recognize, when the models are available. Low-confidence predictions are rejected.
3. Gemini handles requests the local stages cannot interpret.

The local NLP confidence thresholds are starting values, not reliability guarantees. Deletions still require approval.

## Project layout

| Directory | Purpose |
| --- | --- |
| `app/` | CLI entry point, orchestration, and MCP client |
| `core/` | Command parsing, execution, confirmations, and result types |
| `plugins/` | Windows Explorer, file/folder, and LibreOffice context helpers |
| `mcp_servers/` | Windows utility and LibreOffice MCP tools |
| `llm/` | Gemini provider and LLM routing |
| `nlp/` | Local NLP adapter, training resources, evaluation results, and model artifacts |
| `data/` | Application names and aliases |

## Troubleshooting

- **Missing API key:** Add `GEMINI_API_KEY` to the root `.env` file.
- **Local NLP unavailable:** WinPilot can continue to Gemini when local models or their dependencies are unavailable.
- **LibreOffice features unavailable:** Install LibreOffice and ensure it can be launched on your system.
