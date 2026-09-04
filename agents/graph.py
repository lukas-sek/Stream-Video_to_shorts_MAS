"""
Phase 2 + 3 + 4 LangGraph StateGraph — Detection, Packaging & Editing Pipeline.

Phase 2 nodes:
  1. download_segment  — streamlink | ffmpeg pipe, saves raw .ts clip
  2. extract_audio     — ffmpeg-python converts .ts → 16kHz mono WAV
  3. run_vad           — silero-vad speech timestamp detection
  4. analyze_rms       — numpy 500ms sliding-window RMS peak analysis
  5. evaluate_signal   — conditional: emit_candidate OR discard_segment

Phase 3 nodes (chained after emit_candidate):
  6. transcribe_audio  — faster-whisper base.en int8, word-level timestamps
  7. package_clip      — Qwen 2.5 7B via Ollama → ClipMetadata (title/hook/score/tags)
  8. evaluate_virality — conditional: finalize_clip OR discard_transcript
  9. finalize_clip     — write to output/clips.jsonl (passing clips)
 10. discard_transcript— delete .wav, log discard reason

Phase 4 nodes (chained after finalize_clip):
 11. prepare_edit      — compute cut_points + generate .ass subtitle file
 12. render_video      — FFmpeg: dead-air trim + 9:16 reframe + subtitle burn-in
 13. store_edit        — assemble EditedClip, write to output/edits.jsonl

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

from agents.prompts import build_messages
from agents.subtitles import escape_ass_path, generate_ass
from models.signals import AudioAnalysis, ChatSpike, ClipMetadata, EditedClip, _LLMPackage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEGMENT_DURATION: int = int(os.getenv("SEGMENT_DURATION", "75"))
RMS_THRESHOLD: float = float(os.getenv("RMS_THRESHOLD", "0.05"))
RMS_WINDOW_MS: int = 500
VIRALITY_THRESHOLD: float = float(os.getenv("VIRALITY_THRESHOLD", "6.0"))
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base.en")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "shorts-llm")

CUT_GAP_MS: int = int(os.getenv("CUT_GAP_MS", "400"))
VIDEO_LAYOUT: str = os.getenv("VIDEO_LAYOUT", "center_crop")
FACECAM_ROI: str = os.getenv("FACECAM_ROI", "1280,720,640,360")
VIDEO_CRF: int = int(os.getenv("VIDEO_CRF", "23"))
VIDEO_PRESET: str = os.getenv("VIDEO_PRESET", "fast")

OUTPUT_DIR = Path("output/segments")
FINAL_DIR = Path("output/final")
CLIPS_LOG = Path("output/clips.jsonl")
EDITS_LOG = Path("output/edits.jsonl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_LOG.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class Phase2State(TypedDict):
    # Phase 2
    spike: ChatSpike
    segment_path: str
    wav_path: str
    vad_segments: Annotated[list[dict], lambda a, b: b]  # always replace
    rms_peak: float
    rms_mean: float
    has_audio_event: bool
    candidate: AudioAnalysis | None
    # Phase 3
    transcript_text: str
    word_timestamps: Annotated[list[dict], lambda a, b: b]  # always replace
    language: str
    clip_metadata: ClipMetadata | None
    # Phase 4
    cut_points: Annotated[list[dict], lambda a, b: b]   # speech windows on trimmed timeline
    ass_path: str
    output_path: str


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
# Phase 3 Node 6: transcribe_audio
# ---------------------------------------------------------------------------
async def transcribe_audio(state: Phase2State) -> dict:
    """
    Run faster-whisper (base.en, int8) on the WAV file.
    Exports full transcript text and word-level timestamps.
    Explicit gc.collect() after to free model RAM.
    """
    wav_path = state["wav_path"]
    logger.info("Transcribing %s with faster-whisper (%s, int8)", wav_path, WHISPER_MODEL)

    loop = asyncio.get_event_loop()

    def _run() -> tuple[str, list[dict], str]:
        from faster_whisper import WhisperModel  # lazy import

        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav_path, word_timestamps=True, beam_size=5)

        transcript_parts: list[str] = []
        word_timestamps: list[dict] = []

        for seg in segments:
            transcript_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    word_timestamps.append({
                        "word": w.word,
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                        "probability": round(float(w.probability), 4),
                    })

        transcript_text = " ".join(transcript_parts)
        language = info.language if hasattr(info, "language") else "en"

        del model
        gc.collect()
        return transcript_text, word_timestamps, language

    transcript_text, word_timestamps, language = await loop.run_in_executor(None, _run)

    logger.info(
        "Transcription done: lang=%s words=%d text_preview=%r",
        language,
        len(word_timestamps),
        transcript_text[:80],
    )
    return {
        "transcript_text": transcript_text,
        "word_timestamps": word_timestamps,
        "language": language,
    }


# ---------------------------------------------------------------------------
# Phase 3 Node 7: package_clip
# ---------------------------------------------------------------------------
async def package_clip(state: Phase2State) -> dict:
    """
    Send the transcript to Qwen 2.5 7B via Ollama's OpenAI-compatible endpoint.
    Parse structured JSON -> ClipMetadata (without passed_threshold set yet).
    """
    transcript = state["transcript_text"]
    logger.info(
        "Packaging clip via LLM (model=%s, transcript_len=%d)", OLLAMA_MODEL, len(transcript)
    )

    loop = asyncio.get_event_loop()

    def _run() -> ClipMetadata:
        import json
        from openai import OpenAI  # lazy import

        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        messages = build_messages(transcript)

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        raw_json = response.choices[0].message.content or "{}"
        logger.debug("LLM raw output: %s", raw_json[:300])

        try:
            llm_data = _LLMPackage.model_validate(json.loads(raw_json))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("LLM JSON parse failed (%s) — defaulting to zero score", exc)
            llm_data = _LLMPackage(
                title="Unknown Clip",
                hook_text="Something happened here",
                virality_score=0.0,
                tags=[],
            )

        audio_analysis: AudioAnalysis = state["candidate"]  # type: ignore[assignment]
        return ClipMetadata(
            audio_analysis=audio_analysis,
            transcript_text=state["transcript_text"],
            word_timestamps=state["word_timestamps"],
            language=state["language"],
            title=llm_data.title,
            hook_text=llm_data.hook_text,
            virality_score=llm_data.virality_score,
            tags=llm_data.tags,
            passed_threshold=False,  # set by finalize_clip
        )

    clip_metadata = await loop.run_in_executor(None, _run)
    logger.info(
        "LLM packaged: score=%.1f title=%r", clip_metadata.virality_score, clip_metadata.title
    )
    return {"clip_metadata": clip_metadata}


# ---------------------------------------------------------------------------
# Conditional edge: evaluate_virality  (Phase 3)
# ---------------------------------------------------------------------------
def evaluate_virality(state: Phase2State) -> str:
    """Route to finalize_clip if virality score meets threshold, else discard."""
    meta = state.get("clip_metadata")
    if meta is None:
        return "discard_transcript"
    return "finalize_clip" if meta.virality_score >= VIRALITY_THRESHOLD else "discard_transcript"


# ---------------------------------------------------------------------------
# Phase 3 Node 8a: finalize_clip
# ---------------------------------------------------------------------------
async def finalize_clip(state: Phase2State) -> dict:
    """Stamp passed_threshold=True and append ClipMetadata to output/clips.jsonl."""
    meta: ClipMetadata = state["clip_metadata"]  # type: ignore[assignment]
    meta = meta.model_copy(update={"passed_threshold": True})

    with CLIPS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(meta.model_dump_json() + "\n")

    logger.info(
        "CLIP FINALIZED  title=%r  score=%.1f  tags=%s",
        meta.title,
        meta.virality_score,
        meta.tags,
    )
    return {"clip_metadata": meta}


# ---------------------------------------------------------------------------
# Phase 3 Node 8b: discard_transcript
# ---------------------------------------------------------------------------
async def discard_transcript(state: Phase2State) -> dict:
    """Virality below threshold — delete WAV, mark clip_metadata as not passing."""
    meta = state.get("clip_metadata")
    score = meta.virality_score if meta else 0.0

    wav = Path(state.get("wav_path", ""))
    if wav.exists():
        wav.unlink()
        logger.debug("Deleted WAV %s", wav)

    logger.info(
        "CLIP DISCARDED  score=%.1f < threshold=%.1f  title=%r",
        score,
        VIRALITY_THRESHOLD,
        meta.title if meta else "n/a",
    )
    if meta:
        return {"clip_metadata": meta.model_copy(update={"passed_threshold": False})}
    return {}


# ---------------------------------------------------------------------------
# Phase 4 Node 11: prepare_edit
# ---------------------------------------------------------------------------
async def prepare_edit(state: Phase2State) -> dict:
    """
    Compute cut_points from vad_segments (filter gaps > CUT_GAP_MS),
    then generate the .ass subtitle file with time-adjusted word timestamps.
    """
    vad_segments: list[dict] = state["vad_segments"]
    word_timestamps: list[dict] = state["word_timestamps"]
    segment_path: str = state["segment_path"]

    gap_threshold = CUT_GAP_MS / 1000.0

    # Build cut_points: keep speech segments, drop gaps > threshold
    cut_points: list[dict] = []
    if not vad_segments:
        # No VAD data — use the full clip as one segment
        logger.warning("No VAD segments found; using full clip without trimming")
        cut_points = [{"start": 0.0, "end": float(SEGMENT_DURATION)}]
    else:
        for i, seg in enumerate(vad_segments):
            if i == 0:
                cut_points.append(seg)
                continue
            gap = float(seg["start"]) - float(vad_segments[i - 1]["end"])
            if gap <= gap_threshold:
                # Merge with previous segment (gap too small to cut)
                cut_points[-1] = {"start": cut_points[-1]["start"], "end": seg["end"]}
            else:
                cut_points.append(seg)

    logger.info(
        "Cut list: %d segment(s), gap threshold=%.3fs, total kept=%.1fs",
        len(cut_points),
        gap_threshold,
        sum(float(s["end"]) - float(s["start"]) for s in cut_points),
    )

    # Generate .ass subtitle file (adjusted timestamps)
    ass_path = segment_path.replace(".ts", ".ass")
    loop = asyncio.get_event_loop()
    abs_ass_path = await loop.run_in_executor(
        None, generate_ass, word_timestamps, cut_points, ass_path
    )
    logger.info("ASS subtitle file written: %s", abs_ass_path)

    return {"cut_points": cut_points, "ass_path": abs_ass_path}


# ---------------------------------------------------------------------------
# Phase 4 Node 12: render_video
# ---------------------------------------------------------------------------
async def render_video(state: Phase2State) -> dict:
    """
    Single FFmpeg pass:
      1. Dead-air trimming via select/aselect with cut_points
      2. 9:16 reframing (center_crop or stacked layout)
      3. .ass subtitle burn-in

    Output: output/final/{channel}_{ts}.mp4
    """
    segment_path = state["segment_path"]
    ass_path = state["ass_path"]
    cut_points: list[dict] = state["cut_points"]
    spike: ChatSpike = state["spike"]

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = str(FINAL_DIR / f"{spike.channel}_{ts}.mp4")

    # Build select expression from cut_points
    def _build_select_expr(points: list[dict]) -> str:
        parts = [
            f"between(t\\,{float(s['start']):.3f}\\,{float(s['end']):.3f})"
            for s in points
        ]
        return "+".join(parts) if parts else "1"

    select_expr = _build_select_expr(cut_points)
    ass_escaped = escape_ass_path(ass_path)

    # Build video filter chain depending on layout
    if VIDEO_LAYOUT == "stacked":
        try:
            fx, fy, fw, fh = [int(v) for v in FACECAM_ROI.split(",")]
        except ValueError:
            logger.warning("Invalid FACECAM_ROI '%s'; falling back to center_crop", FACECAM_ROI)
            VIDEO_LAYOUT_EFFECTIVE = "center_crop"
        else:
            VIDEO_LAYOUT_EFFECTIVE = "stacked"
            # gameplay: full frame minus the facecam area, centered
            gx, gy = 0, 0
            gw = fx  # everything to the left of the facecam
            gh = fh
    else:
        VIDEO_LAYOUT_EFFECTIVE = "center_crop"

    if VIDEO_LAYOUT_EFFECTIVE == "stacked":
        vf_reframe = (
            f"[0:v]crop={fw}:{fh}:{fx}:{fy},scale=1080:608[fc];"
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale=1080:1312[gp];"
            "[fc][gp]vstack=inputs=2[stacked];"
            f"[stacked]select='{select_expr}',setpts=N/FRAME_RATE/TB,"
            f"subtitles='{ass_escaped}'[vout]"
        )
    else:
        # center_crop: crop 9:16 center, scale to 1080x1920
        vf_reframe = (
            f"[0:v]select='{select_expr}',setpts=N/FRAME_RATE/TB,"
            "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
            "scale=1080:1920:force_original_aspect_ratio=disable,"
            f"subtitles='{ass_escaped}'[vout]"
        )

    af = f"[0:a]aselect='{select_expr}',asetpts=N/SR/TB[aout]"
    filter_complex = f"{vf_reframe};{af}"

    cmd = [
        "ffmpeg", "-y",
        "-i", segment_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", str(VIDEO_CRF),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info("Rendering video: %s (layout=%s)", output_path, VIDEO_LAYOUT_EFFECTIVE)

    loop = asyncio.get_event_loop()

    def _run() -> None:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:].decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg render failed:\n{stderr_tail}")

    await loop.run_in_executor(None, _run)

    size_mb = Path(output_path).stat().st_size / 1e6
    logger.info("Render complete: %s (%.1f MB)", output_path, size_mb)
    return {"output_path": output_path}


# ---------------------------------------------------------------------------
# Phase 4 Node 13: store_edit
# ---------------------------------------------------------------------------
async def store_edit(state: Phase2State) -> dict:
    """
    Assemble EditedClip, compute final duration, write to output/edits.jsonl.
    """
    output_path = state["output_path"]
    cut_points = state["cut_points"]

    # Compute total duration from cut_points
    duration = sum(float(s["end"]) - float(s["start"]) for s in cut_points)

    clip_metadata: ClipMetadata = state["clip_metadata"]  # type: ignore[assignment]
    edited = EditedClip(
        clip_metadata=clip_metadata,
        cut_points=cut_points,
        ass_path=state["ass_path"],
        output_path=output_path,
        duration=round(duration, 2),
    )

    with EDITS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(edited.model_dump_json() + "\n")

    logger.info(
        "EDIT STORED  output=%s  duration=%.1fs  title=%r",
        output_path,
        duration,
        clip_metadata.title,
    )
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None) -> object:
    """Compile and return the Phase 2+3+4 detection, packaging & editing StateGraph."""
    builder = StateGraph(Phase2State)

    # Phase 2 nodes
    builder.add_node("download_segment", download_segment, timeout=SEGMENT_DURATION + 30)
    builder.add_node("extract_audio", extract_audio, timeout=60)
    builder.add_node("run_vad", run_vad, timeout=120)
    builder.add_node("analyze_rms", analyze_rms, timeout=30)
    builder.add_node("emit_candidate", emit_candidate)
    builder.add_node("discard_segment", discard_segment)

    # Phase 3 nodes
    builder.add_node("transcribe_audio", transcribe_audio, timeout=180)
    builder.add_node("package_clip", package_clip, timeout=60)
    builder.add_node("finalize_clip", finalize_clip)
    builder.add_node("discard_transcript", discard_transcript)

    # Phase 4 nodes
    builder.add_node("prepare_edit", prepare_edit, timeout=30)
    builder.add_node("render_video", render_video, timeout=300)
    builder.add_node("store_edit", store_edit)

    # Phase 2 edges
    builder.add_edge(START, "download_segment")
    builder.add_edge("download_segment", "extract_audio")
    builder.add_edge("extract_audio", "run_vad")
    builder.add_edge("run_vad", "analyze_rms")
    builder.add_conditional_edges(
        "analyze_rms",
        evaluate_signal,
        {"emit_candidate": "emit_candidate", "discard_segment": "discard_segment"},
    )
    builder.add_edge("discard_segment", END)

    # Phase 3 edges (chained after emit_candidate)
    builder.add_edge("emit_candidate", "transcribe_audio")
    builder.add_edge("transcribe_audio", "package_clip")
    builder.add_conditional_edges(
        "package_clip",
        evaluate_virality,
        {"finalize_clip": "finalize_clip", "discard_transcript": "discard_transcript"},
    )
    builder.add_edge("discard_transcript", END)

    # Phase 4 edges (chained after finalize_clip instead of END)
    builder.add_edge("finalize_clip", "prepare_edit")
    builder.add_edge("prepare_edit", "render_video")
    builder.add_edge("render_video", "store_edit")
    builder.add_edge("store_edit", END)

    return builder.compile(checkpointer=checkpointer)
