from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
