"""
Scout Agent — Tier 1 signal detection.

Connects anonymously to Twitch IRC, maintains a 5-second message buffer,
and emits ChatSpike events when a velocity or emote-flood spike is detected.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from collections import deque

from models.signals import ChatSpike

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IRC config
# ---------------------------------------------------------------------------
IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667

# ---------------------------------------------------------------------------
# Spike detection config (overridable via env vars)
# ---------------------------------------------------------------------------
WINDOW_SECS: int = 5
BASELINE_WINDOWS: int = int(os.getenv("BASELINE_WINDOWS", "30"))   # 2.5-min baseline
SPIKE_MULTIPLIER: float = float(os.getenv("SPIKE_MULTIPLIER", "3.0"))
EMOTE_FLOOD_RATIO: float = 0.6   # fraction of uppercase chars to flag as emote flood
MIN_BASELINE_WINDOWS: int = 3    # don't fire until we have at least 3 windows of data

# Known high-signal Twitch emotes
_EMOTE_RE = re.compile(
    r"\b(KEKW|PogChamp|LUL|OMEGALUL|Pog|PauseChamp|POGGERS|monkaS|EZ|Clap|TriHard|HeyGuys)\b",
    re.IGNORECASE,
)


def _is_emote_flood(message: str) -> bool:
    """Return True if the message looks like an emote spam / caps flood."""
    alpha = [c for c in message if c.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if upper_ratio >= EMOTE_FLOOD_RATIO:
        return True
    return bool(_EMOTE_RE.search(message))


class ScoutAgent:
    """
    Monitors a single Twitch channel's IRC chat for viral-moment signals.

    Usage::

        queue: asyncio.Queue[ChatSpike] = asyncio.Queue()
        scout = ScoutAgent(channel="xqc", spike_queue=queue)
        await scout.run()
    """

    def __init__(self, channel: str, spike_queue: asyncio.Queue[ChatSpike]) -> None:
        self.channel = channel.lower().lstrip("#")
        self.spike_queue = spike_queue
        self._nick = f"justinfan{random.randint(10000, 99999)}"

        # Sliding window state
        self._window_counts: deque[int] = deque(maxlen=BASELINE_WINDOWS)
        self._emote_counts: deque[int] = deque(maxlen=BASELINE_WINDOWS)
        self._current_msgs: int = 0
        self._current_emotes: int = 0
        self._window_start: float = time.monotonic()

        # Cooldown: don't fire two spikes within 30 s
        self._last_spike_at: float = 0.0
        self._spike_cooldown: float = 30.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick_window(self) -> None:
        """Flush the current 5-second window into the rolling buffer."""
        self._window_counts.append(self._current_msgs)
        self._emote_counts.append(self._current_emotes)
        self._current_msgs = 0
        self._current_emotes = 0
        self._window_start = time.monotonic()

    def _baseline(self) -> float:
        if not self._window_counts:
            return 0.0
        return sum(self._window_counts) / len(self._window_counts)

    def _emote_baseline(self) -> float:
        if not self._emote_counts:
            return 0.0
        return sum(self._emote_counts) / len(self._emote_counts)

    def _check_spike(self) -> ChatSpike | None:
        """Return a ChatSpike if current window exceeds threshold, else None."""
        now = time.monotonic()
        if now - self._last_spike_at < self._spike_cooldown:
            return None
        if len(self._window_counts) < MIN_BASELINE_WINDOWS:
            return None

        baseline = self._baseline()
        if baseline < 1.0:
            return None

        ratio = self._current_msgs / baseline

        if ratio >= SPIKE_MULTIPLIER:
            self._last_spike_at = now
            return ChatSpike(
                channel=self.channel,
                msg_count=self._current_msgs,
                baseline=baseline,
                ratio=ratio,
                trigger="velocity",
            )

        # Emote flood check: absolute count >= 5 AND ratio >= 2x emote baseline
        emote_base = self._emote_baseline()
        if self._current_emotes >= 5 and (
            emote_base < 1.0 or self._current_emotes / emote_base >= 2.0
        ):
            self._last_spike_at = now
            return ChatSpike(
                channel=self.channel,
                msg_count=self._current_msgs,
                baseline=baseline,
                ratio=ratio,
                trigger="emote_flood",
            )

        return None

    def _handle_message(self, raw: str) -> None:
        """Process a raw IRC line."""
        if raw.startswith("PING"):
            return  # handled in run()

        # PRIVMSG lines: ":nick!user@host PRIVMSG #channel :text"
        if "PRIVMSG" not in raw:
            return

        try:
            text = raw.split("PRIVMSG", 1)[1].split(":", 1)[1].strip()
        except IndexError:
            return

        self._current_msgs += 1
        if _is_emote_flood(text):
            self._current_emotes += 1

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect to Twitch IRC and stream messages until cancelled."""
        logger.info("ScoutAgent connecting to #%s as %s", self.channel, self._nick)
        reconnect_delay = 5.0

        while True:
            try:
                reader, writer = await asyncio.open_connection(IRC_HOST, IRC_PORT)
                logger.info("ScoutAgent connected to %s:%s", IRC_HOST, IRC_PORT)

                # Twitch IRC handshake (anonymous — no PASS needed)
                writer.write(f"NICK {self._nick}\r\n".encode())
                writer.write(f"USER {self._nick} 0 * :{self._nick}\r\n".encode())
                writer.write(f"JOIN #{self.channel}\r\n".encode())
                await writer.drain()

                while True:
                    # Enforce window boundary
                    elapsed = time.monotonic() - self._window_start
                    if elapsed >= WINDOW_SECS:
                        spike = self._check_spike()
                        self._tick_window()
                        if spike:
                            logger.info("Spike detected: %s", spike.label)
                            await self.spike_queue.put(spike)

                    # Read with timeout so we tick windows even during silence
                    try:
                        line_bytes = await asyncio.wait_for(
                            reader.readline(), timeout=max(0.1, WINDOW_SECS - elapsed)
                        )
                    except asyncio.TimeoutError:
                        continue

                    if not line_bytes:
                        logger.warning("ScoutAgent: connection closed by server")
                        break

                    raw = line_bytes.decode("utf-8", errors="replace").rstrip()

                    if raw.startswith("PING"):
                        pong = raw.replace("PING", "PONG")
                        writer.write(f"{pong}\r\n".encode())
                        await writer.drain()
                        continue

                    self._handle_message(raw)

            except (OSError, ConnectionResetError) as exc:
                logger.warning("ScoutAgent connection error: %s — reconnecting in %ss", exc, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)
            except asyncio.CancelledError:
                logger.info("ScoutAgent cancelled")
                return
