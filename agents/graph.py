"""
Phase 2 LangGraph StateGraph — Audio Detection Pipeline.

Nodes (in order):
  1. download_segment  — streamlink | ffmpeg pipe, saves raw .ts clip
  2. extract_audio     — ffmpeg-python converts .ts → 16kHz mono WAV
  3. run_vad           — silero-vad speech timestamp detection
  4. analyze_rms       — numpy 500ms sliding-window RMS peak analysis
  5. evaluate_signal   — conditional edge: emit_candidate OR discard_segment

Invoked once per ChatSpike by pipeline.py.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated

import ffmpeg  # type: ignore[import-untyped]
import numpy as np
from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
from typing_extensions import TypedDict

from models.signals import AudioAnalysis, ChatSpike

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEGMENT_DURATION: int = int(os.getenv("SEGMENT_DURATION", "75"))
RMS_THRESHOLD: float = float(os.getenv("RMS_THRESHOLD", "0.05"))
RMS_WINDOW_MS: int = 500
OUTPUT_DIR = Path("output/segments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class Phase2State(TypedDict):
    spike: ChatSpike
    segment_path: str
    wav_path: str
    vad_segments: Annotated[list[dict], lambda a, b: b]  # always replace
    rms_peak: float
    rms_mean: float
    has_audio_event: bool
    candidate: AudioAnalysis | None


# ---------------------------------------------------------------------------
# Node 1: download_segment
# ---------------------------------------------------------------------------
async def download_segment(state: Phase2State) -> dict:
    """
    Stream 75 s of live Twitch video via:
        streamlink --stdout twitch.tv/<channel> best | ffmpeg -i pipe:0 -t N <out.ts>
    """
    spike: ChatSpike = state["spike"]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    segment_path = str(OUTPUT_DIR / f"{spike.channel}_{ts}.ts")

    stream_url = f"https://twitch.tv/{spike.channel}"
    logger.info("Downloading %ds segment from %s → %s", SEGMENT_DURATION, stream_url, segment_path)

    # streamlink pipes raw stream bytes; ffmpeg demuxes and cuts to SEGMENT_DURATION
    cmd_streamlink = [
        "streamlink", "--stdout", "--quiet",
        stream_url, "best",
    ]
    cmd_ffmpeg = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-t", str(SEGMENT_DURATION),
        "-c", "copy",
        segment_path,
    ]

    loop = asyncio.get_event_loop()

    def _run() -> None:
        with subprocess.Popen(cmd_streamlink, stdout=subprocess.PIPE) as sl_proc:
            with subprocess.Popen(
                cmd_ffmpeg, stdin=sl_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ) as ff_proc:
                if sl_proc.stdout:
                    sl_proc.stdout.close()
                ff_proc.wait(timeout=SEGMENT_DURATION + 30)
                sl_proc.terminate()

    await loop.run_in_executor(None, _run)

    if not Path(segment_path).exists():
        raise RuntimeError(f"Segment download failed: {segment_path} not created")

    logger.info("Segment saved: %s (%.1f MB)", segment_path, Path(segment_path).stat().st_size / 1e6)
    return {"segment_path": segment_path}


# ---------------------------------------------------------------------------
# Node 2: extract_audio
# ---------------------------------------------------------------------------
async def extract_audio(state: Phase2State) -> dict:
    """Convert the raw .ts segment to a 16 kHz mono PCM WAV using ffmpeg-python."""
    segment_path = state["segment_path"]
    wav_path = segment_path.replace(".ts", ".wav")

    logger.info("Extracting audio: %s → %s", segment_path, wav_path)

    loop = asyncio.get_event_loop()

    def _run() -> None:
        (
            ffmpeg
            .input(segment_path)
            .output(wav_path, ar=16000, ac=1, acodec="pcm_s16le", vn=None)
            .overwrite_output()
            .run(quiet=True)
        )

    await loop.run_in_executor(None, _run)

    if not Path(wav_path).exists():
        raise RuntimeError(f"Audio extraction failed: {wav_path} not created")

    logger.info("WAV saved: %s", wav_path)
    return {"wav_path": wav_path}


# ---------------------------------------------------------------------------
# Node 3: run_vad
# ---------------------------------------------------------------------------
async def run_vad(state: Phase2State) -> dict:
    """Run silero-vad on the WAV and return speech segment timestamps."""
    wav_path = state["wav_path"]
    logger.info("Running silero-vad on %s", wav_path)

    loop = asyncio.get_event_loop()

    def _run() -> list[dict]:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # lazy import

        model = load_silero_vad()
        wav = read_audio(wav_path, sampling_rate=16000)
        raw_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        # Ensure JSON-serialisable plain dicts
        segments = [{"start": float(t["start"]), "end": float(t["end"])} for t in raw_timestamps]
        del model, wav
        gc.collect()
        return segments

    vad_segments = await loop.run_in_executor(None, _run)
    speech_duration = sum(s["end"] - s["start"] for s in vad_segments)
    logger.info("VAD found %d speech segment(s), %.1fs total", len(vad_segments), speech_duration)
    return {"vad_segments": vad_segments}


# ---------------------------------------------------------------------------
# Node 4: analyze_rms
# ---------------------------------------------------------------------------
async def analyze_rms(state: Phase2State) -> dict:
    """
    Compute RMS energy in 500 ms sliding windows across the full WAV.
    Returns rms_peak and rms_mean.
    """
    wav_path = state["wav_path"]
    logger.info("Analyzing RMS energy: %s", wav_path)

    loop = asyncio.get_event_loop()

    def _run() -> tuple[float, float]:
        import wave

        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        dtype = np.int16 if sampwidth == 2 else np.int32
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            samples = samples[::n_channels]  # take left channel

        # Normalize to [-1, 1]
        max_val = float(np.iinfo(dtype).max)
        samples /= max_val

        window_size = int(sample_rate * RMS_WINDOW_MS / 1000)
        if window_size < 1 or len(samples) < window_size:
            return 0.0, 0.0

        n_windows = len(samples) // window_size
        windows = samples[: n_windows * window_size].reshape(n_windows, window_size)
        rms_per_window = np.sqrt(np.mean(windows ** 2, axis=1))

        return float(rms_per_window.max()), float(rms_per_window.mean())

    rms_peak, rms_mean = await loop.run_in_executor(None, _run)
    logger.info("RMS peak=%.4f mean=%.4f (threshold=%.4f)", rms_peak, rms_mean, RMS_THRESHOLD)
    return {"rms_peak": rms_peak, "rms_mean": rms_mean}


# ---------------------------------------------------------------------------
# Node 5a: emit_candidate
# ---------------------------------------------------------------------------
async def emit_candidate(state: Phase2State) -> dict:
    """Package the analysis result as an AudioAnalysis and store in state."""
    candidate = AudioAnalysis(
        spike=state["spike"],
        segment_path=state["segment_path"],
        wav_path=state["wav_path"],
        vad_segments=state["vad_segments"],
        rms_peak=state["rms_peak"],
        rms_mean=state["rms_mean"],
        has_audio_event=True,
    )
    logger.info(
        "Candidate accepted: channel=%s rms_peak=%.4f vad_segs=%d",
        state["spike"].channel,
        state["rms_peak"],
        len(state["vad_segments"]),
    )
    return {"has_audio_event": True, "candidate": candidate}


# ---------------------------------------------------------------------------
# Node 5b: discard_segment
# ---------------------------------------------------------------------------
async def discard_segment(state: Phase2State) -> dict:
    """RMS below threshold — delete segment files and mark as discarded."""
    for path_key in ("segment_path", "wav_path"):
        p = Path(state.get(path_key, ""))
        if p.exists():
            p.unlink()
            logger.debug("Deleted %s", p)

    logger.info(
        "Segment discarded (rms_peak=%.4f < threshold=%.4f)",
        state["rms_peak"],
        RMS_THRESHOLD,
    )
    return {"has_audio_event": False, "candidate": None}


# ---------------------------------------------------------------------------
# Conditional edge: evaluate_signal
# ---------------------------------------------------------------------------
def evaluate_signal(state: Phase2State) -> str:
    """Route to emit_candidate if RMS peak crosses threshold, else discard."""
    return "emit_candidate" if state["rms_peak"] >= RMS_THRESHOLD else "discard_segment"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None) -> object:
    """Compile and return the Phase 2 detection StateGraph."""
    builder = StateGraph(Phase2State)

    builder.add_node("download_segment", download_segment, timeout=SEGMENT_DURATION + 30)
    builder.add_node("extract_audio", extract_audio, timeout=60)
    builder.add_node("run_vad", run_vad, timeout=120)
    builder.add_node("analyze_rms", analyze_rms, timeout=30)
    builder.add_node("emit_candidate", emit_candidate)
    builder.add_node("discard_segment", discard_segment)

    builder.add_edge(START, "download_segment")
    builder.add_edge("download_segment", "extract_audio")
    builder.add_edge("extract_audio", "run_vad")
    builder.add_edge("run_vad", "analyze_rms")
    builder.add_conditional_edges(
        "analyze_rms",
        evaluate_signal,
        {"emit_candidate": "emit_candidate", "discard_segment": "discard_segment"},
    )
    builder.add_edge("emit_candidate", END)
    builder.add_edge("discard_segment", END)

    return builder.compile(checkpointer=checkpointer)
