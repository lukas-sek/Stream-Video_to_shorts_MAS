"""
Phase 1 verification script.

Checks:
  1. All required packages import without error
  2. FFmpeg binary is available on PATH
  3. Ollama is reachable and the required models are listed
  4. Sends a 1-sentence prompt to 'shorts-llm' and prints the response + RAM usage
"""

import sys
import shutil
import subprocess
import importlib
import psutil
import os

SEP = "-" * 60
OK   = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check_imports() -> bool:
    section("1 / Package imports")
    packages = {
        "yt_dlp":        "yt-dlp",
        "streamlink":    "streamlink",
        "faster_whisper":"faster-whisper",
        "torch":         "torch",
        "ffmpeg":        "ffmpeg-python",
        "pydantic":      "pydantic",
        "openai":        "openai",
        "psutil":        "psutil",
    }
    all_ok = True
    for module, pip_name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "n/a")
            print(f"  {OK}{pip_name:<20} {version}")
        except ImportError as exc:
            print(f"  {FAIL}{pip_name:<20} {exc}")
            all_ok = False
    return all_ok


def check_ffmpeg() -> bool:
    section("2 / FFmpeg binary")
    path = shutil.which("ffmpeg")
    if path is None:
        print(f"  {WARN}ffmpeg not found on PATH.")
        print("       Install with:  winget install Gyan.FFmpeg")
        return False
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
    print(f"  {OK}{first_line}")
    return True


def check_ollama_models() -> bool:
    section("3 / Ollama models")
    # Match on prefix so "shorts-llm:latest" satisfies "shorts-llm"
    required = {"qwen2.5:7b-instruct-q4_k_m", "llama3.2:3b", "shorts-llm"}
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"  {FAIL}ollama list failed: {result.stderr.strip()}")
            return False
        lines = result.stdout.strip().splitlines()
        found = set()
        for line in lines[1:]:  # skip header
            name = line.split()[0] if line.split() else ""
            found.add(name)
            print(f"  {OK}{line}")
        # A required entry matches if any found name starts with it (handles :latest suffix)
        missing = {r for r in required if not any(f.startswith(r) for f in found)}
        if missing:
            for m in sorted(missing):
                print(f"  {WARN}Model not found: {m}  (run setup.ps1 to pull)")
            return False
        return True
    except FileNotFoundError:
        print(f"  {FAIL}ollama binary not found. Install Ollama and re-run.")
        return False


def check_ollama_endpoint() -> bool:
    section("4 / Ollama OpenAI-compatible endpoint")
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"  {OK}Endpoint reachable — {len(model_ids)} model(s) listed")
        return True
    except Exception as exc:
        print(f"  {WARN}Endpoint not reachable: {exc}")
        print("       Start Ollama (ollama serve) and re-run to test the endpoint.")
        return False


def smoke_test_llm() -> bool:
    section("5 / shorts-llm smoke test")
    try:
        from openai import OpenAI
        import psutil, os

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        prompt = (
            "You are a viral-clip hook writer. "
            "Write one punchy 6-word hook for a gaming clip where the player gets a 1-vs-5 ace."
        )
        print("  Sending prompt to shorts-llm...")
        response = client.chat.completions.create(
            model="shorts-llm",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )
        hook = response.choices[0].message.content.strip()
        print(f"  {OK}Response: {hook}")

        ram = psutil.virtual_memory()
        used_gb  = (ram.total - ram.available) / 1024**3
        total_gb = ram.total / 1024**3
        print(f"  {OK}RAM usage after inference: {used_gb:.1f} GB / {total_gb:.1f} GB ({ram.percent}%)")
        return True
    except Exception as exc:
        print(f"  {WARN}LLM smoke test skipped: {exc}")
        print("       Ensure Ollama is running and 'shorts-llm' model is registered.")
        return False


def main() -> None:
    print("\nStream-to-Shorts MAS — Phase 1 Environment Verification")
    print(f"Python {sys.version}")

    results = {
        "Package imports":     check_imports(),
        "FFmpeg binary":       check_ffmpeg(),
        "Ollama models":       check_ollama_models(),
        "Ollama endpoint":     check_ollama_endpoint(),
        "LLM smoke test":      smoke_test_llm(),
    }

    section("Summary")
    all_passed = True
    for name, passed in results.items():
        status = OK if passed else WARN
        print(f"  {status}{name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("Phase 1 verification PASSED. Ready for Phase 2.")
        sys.exit(0)
    else:
        print("Some checks did not pass (see warnings above).")
        print("FFmpeg and Ollama warnings are non-fatal; fix them before Phase 2.")
        sys.exit(0)


if __name__ == "__main__":
    main()
