# Stream-to-Shorts MAS

Autonomous multi-agent system that monitors live streams, detects viral moments, and automatically edits and publishes 9:16 short-form clips.

## Quick Start (Phase 1 Setup)

```powershell
.\setup.ps1
```

The script will:
1. Create a Python virtual environment at `.venv`
2. Install all dependencies (CPU-only PyTorch to keep RAM < 5 GB)
3. Check for FFmpeg on PATH (install via `winget install Gyan.FFmpeg` if missing)
4. Pull required Ollama models (`qwen2.5:7b-instruct-q4_k_m`, `llama3.2:3b`)
5. Register the `shorts-llm` model with `num_ctx 8192`
6. Run `verify_setup.py` to confirm everything is working

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- [FFmpeg](https://ffmpeg.org) on PATH
- 16 GB RAM (local CPU inference)

## Verification

```powershell
.\.venv\Scripts\python.exe verify_setup.py
```

## Project Structure

```
.
├── config/
│   └── Modelfile.qwen25   # Ollama Modelfile (num_ctx 8192)
├── requirements.txt
├── setup.ps1              # Phase 1 one-shot setup
├── verify_setup.py        # Environment verification
└── Roadmap.md
```

See [Roadmap.md](Roadmap.md) for the full build plan.
