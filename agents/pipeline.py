"""
Phase 2+3 Pipeline — Top-level coordinator.

Wires the Scout Agent (asyncio IRC listener) to the LangGraph detection
and packaging pipeline. Each ChatSpike triggers one graph.ainvoke() call
that runs the full Phase 2+3 chain:
  IRC spike -> download -> VAD -> RMS -> ASR -> LLM -> clips.jsonl

Usage::

    # From project root (with .venv activated):
    python -m agents.pipeline --channel xqc

    # Or via environment variable:
    TWITCH_CHANNEL=xqc python -m agents.pipeline

Environment variables:
    TWITCH_CHANNEL      Twitch channel to monitor (required if --channel not passed)
    SPIKE_MULTIPLIER    Chat velocity multiplier threshold (default 3.0)
    SEGMENT_DURATION    Clip length in seconds (default 75)
    RMS_THRESHOLD       Minimum RMS peak to pass Phase 2 (default 0.05)
    VIRALITY_THRESHOLD  Minimum virality score to pass Phase 3 (default 6.0)
    WHISPER_MODEL       faster-whisper model name (default base.en)
    BASELINE_WINDOWS    Rolling-average window count (default 30)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-untyped]

from agents.graph import Phase2State, build_graph
from agents.scout_agent import ScoutAgent
from models.signals import ChatSpike, ClipMetadata, EditedClip

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------
async def process_spikes(
    spike_queue: asyncio.Queue[ChatSpike],
    graph,
    max_concurrent: int = 2,
) -> None:
    """
    Drain the spike queue and invoke the LangGraph pipeline for each event.
    Limits concurrency to max_concurrent to protect RAM on a 16 GB machine.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _handle(spike: ChatSpike) -> None:
        async with semaphore:
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            initial_state: Phase2State = {
                "spike": spike,
                "segment_path": "",
                "wav_path": "",
                "vad_segments": [],
                "rms_peak": 0.0,
                "rms_mean": 0.0,
                "has_audio_event": False,
                "candidate": None,
                # Phase 3
                "transcript_text": "",
                "word_timestamps": [],
                "language": "en",
                "clip_metadata": None,
                # Phase 4
                "cut_points": [],
                "ass_path": "",
                "output_path": "",
            }
            logger.info("Invoking pipeline for spike: %s", spike.label)
            try:
                result = await graph.ainvoke(initial_state, config=config)
                output_path: str = result.get("output_path", "")
                clip: ClipMetadata | None = result.get("clip_metadata")
                if output_path and clip:
                    logger.info(
                        "EDIT READY  channel=%-20s score=%.1f  title=%r  file=%s",
                        spike.channel,
                        clip.virality_score,
                        clip.title,
                        output_path,
                    )
                elif clip and clip.passed_threshold:
                    logger.info(
                        "PACKAGED (render skipped)  channel=%-20s score=%.1f  title=%r",
                        spike.channel,
                        clip.virality_score,
                        clip.title,
                    )
                elif clip:
                    logger.info(
                        "LOW_SCORE  channel=%-20s score=%.1f  title=%r",
                        spike.channel,
                        clip.virality_score,
                        clip.title,
                    )
                else:
                    logger.info("SKIP  channel=%s (below RMS threshold)", spike.channel)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception("Graph invocation failed for spike %s: %s", spike.label, exc)

    while True:
        spike = await spike_queue.get()
        asyncio.create_task(_handle(spike))
        spike_queue.task_done()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main(channel: str) -> None:
    logger.info("=== Phase 2+3+4 Pipeline starting for channel: #%s ===", channel)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    spike_queue: asyncio.Queue[ChatSpike] = asyncio.Queue(maxsize=10)
    scout = ScoutAgent(channel=channel, spike_queue=spike_queue)

    scout_task = asyncio.create_task(scout.run(), name="scout")
    processor_task = asyncio.create_task(
        process_spikes(spike_queue, graph), name="processor"
    )

    logger.info("Scout and processor tasks running. Press Ctrl+C to stop.")
    try:
        await asyncio.gather(scout_task, processor_task)
    except asyncio.CancelledError:
        pass
    finally:
        scout_task.cancel()
        processor_task.cancel()
        logger.info("Pipeline shut down.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream-to-Shorts Phase 2+3+4 Pipeline")
    parser.add_argument(
        "--channel",
        default=os.getenv("TWITCH_CHANNEL", ""),
        help="Twitch channel name to monitor (or set TWITCH_CHANNEL env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.channel:
        print("ERROR: provide --channel <name> or set TWITCH_CHANNEL env var", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(args.channel))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
