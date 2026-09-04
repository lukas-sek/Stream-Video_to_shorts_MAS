from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatSpike(BaseModel):
    """Emitted by the Scout Agent when a chat velocity or emote-flood spike is detected."""

    channel: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    msg_count: int = Field(description="Messages counted in the current 5-second window")
    baseline: float = Field(description="Rolling-average messages per window over last 2.5 min")
    ratio: float = Field(description="msg_count / baseline")
    trigger: Literal["velocity", "emote_flood"]

    @property
    def label(self) -> str:
        ts: datetime = self.timestamp  # type: ignore[assignment]  # pylint: disable=no-member
        return f"[{self.trigger}] #{self.channel} ratio={self.ratio:.1f}x @ {ts.isoformat()}"


class AudioAnalysis(BaseModel):
    """Output of the LangGraph detection pipeline for a single ChatSpike event."""

    spike: ChatSpike
    segment_path: str = Field(description="Path to the raw downloaded .ts segment")
    wav_path: str = Field(description="Path to the extracted 16kHz mono WAV file")
    vad_segments: list[dict] = Field(
        default_factory=list,
        description="silero-vad speech windows: [{start: float, end: float}]",
    )
    rms_peak: float = Field(default=0.0, description="Peak RMS value across all 500ms windows")
    rms_mean: float = Field(default=0.0, description="Mean RMS across all 500ms windows")
    has_audio_event: bool = Field(
        default=False,
        description="True when rms_peak exceeds RMS_THRESHOLD — passes to Phase 3",
    )


# ---------------------------------------------------------------------------
# Phase 3 models
# ---------------------------------------------------------------------------

class _LLMPackage(BaseModel):
    """Internal model: structured JSON output from the Qwen 2.5 7B LLM call."""

    title: str = Field(description="5-8 word punchy clip title")
    hook_text: str = Field(description="5-8 word first-caption hook (imperative or question)")
    virality_score: float = Field(ge=0.0, le=10.0, description="Predicted virality 0-10")
    tags: list[str] = Field(description="3-6 relevant content tags")

    @field_validator("virality_score", mode="before")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        return max(0.0, min(10.0, float(v)))

    @field_validator("tags", mode="before")
    @classmethod
    def _ensure_list(cls, v: object) -> list:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return list(v)  # type: ignore[arg-type]


class ClipMetadata(BaseModel):
    """
    Full output of the Phase 3 packaging pipeline for a passing clip.
    Consumed by Phase 4 (video editor) and Phase 5 (publisher).
    """

    audio_analysis: AudioAnalysis = Field(
        description="Phase 2 result (includes wav_path, vad_segments, segment_path)"
    )
    # ASR output
    transcript_text: str = Field(default="", description="Full transcript from faster-whisper")
    word_timestamps: list[dict] = Field(
        default_factory=list,
        description="Word-level timestamps: [{word, start, end, probability}]",
    )
    language: str = Field(default="en", description="Detected language code")
    # LLM-generated fields
    title: str = Field(default="", description="5-8 word clip title")
    hook_text: str = Field(default="", description="5-8 word first-caption hook")
    virality_score: float = Field(default=0.0, description="Predicted virality 0.0-10.0")
    tags: list[str] = Field(default_factory=list)
    passed_threshold: bool = Field(
        default=False,
        description="True when virality_score >= VIRALITY_THRESHOLD",
    )
