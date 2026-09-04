"""
ASS subtitle generator for Phase 4.

Converts faster-whisper word-level timestamps to an Advanced SubStation Alpha
(.ass) file with:
  - Impact font, size 72, white text, 4px black outline
  - Bottom-center alignment (ASS Alignment=2)
  - Per-word pop-in animation: \\fad(150,50)

Also handles timestamp re-mapping after dead-air trimming:
  words that fall inside removed gaps are discarded;
  words after a gap are shifted back by the cumulative removed duration.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# ASS template parts
# ---------------------------------------------------------------------------

_SCRIPT_INFO = """\
[Script Info]
Title: Stream-to-Shorts Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920
"""

_STYLES = """\
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,20,20,60,1
"""

_EVENTS_HEADER = """\
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Dead-air offset computation
# ---------------------------------------------------------------------------

def compute_adjusted_words(
    word_timestamps: list[dict],
    cut_points: list[dict],
) -> list[dict]:
    """
    Re-map word timestamps to the trimmed timeline.

    Words that fall entirely inside a removed gap are dropped.
    Words that span a gap boundary are truncated to the gap start.
    All words are shifted back by the cumulative duration removed before them.

    Parameters
    ----------
    word_timestamps : list of {word, start, end, probability}
        Original word timestamps from faster-whisper (seconds).
    cut_points : list of {start, end}
        Speech windows KEPT in the trimmed video (already gap-filtered).

    Returns
    -------
    Adjusted word list with same schema, times on new timeline.
    """
    if not cut_points:
        return word_timestamps

    adjusted: list[dict] = []
    for w in word_timestamps:
        w_start = float(w["start"])
        w_end = float(w["end"])

        # Find which cut segment this word belongs to
        offset = 0.0
        in_cut = False
        cumulative_removed = 0.0

        prev_end = 0.0
        for seg in cut_points:
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            # gap removed before this segment
            cumulative_removed += seg_start - prev_end
            prev_end = seg_end

            if w_start >= seg_start and w_start < seg_end:
                in_cut = True
                offset = cumulative_removed
                break

        if not in_cut:
            continue  # word falls in a removed gap — discard

        new_start = w_start - offset
        new_end = min(w_end, prev_end) - offset  # clamp to segment end
        if new_end <= new_start:
            continue

        adjusted.append({
            "word": w["word"],
            "start": round(new_start, 3),
            "end": round(new_end, 3),
            "probability": w.get("probability", 1.0),
        })

    return adjusted


# ---------------------------------------------------------------------------
# ASS file generator
# ---------------------------------------------------------------------------

def generate_ass(
    word_timestamps: list[dict],
    cut_points: list[dict],
    output_path: str,
) -> str:
    """
    Generate an .ass subtitle file from word-level timestamps.

    Parameters
    ----------
    word_timestamps : list of {word, start, end, probability}
        Original word timestamps from faster-whisper.
    cut_points : list of {start, end}
        Speech windows used for the final trimmed video.
        Used to re-map word timestamps to the new timeline.
    output_path : str
        Destination path for the .ass file (e.g. output/segments/clip.ass).

    Returns
    -------
    str : absolute path to the written .ass file.
    """
    adjusted = compute_adjusted_words(word_timestamps, cut_points)

    lines: list[str] = [_SCRIPT_INFO, _STYLES, _EVENTS_HEADER]

    for w in adjusted:
        word = w["word"].strip()
        if not word:
            continue
        t_start = _fmt_ass_time(float(w["start"]))
        t_end = _fmt_ass_time(float(w["end"]))
        # Pop-in animation + uppercase for impact
        text = r"{\fad(150,50)}" + word.upper()
        lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text}")

    content = "\n".join(lines) + "\n"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8-sig")  # utf-8-sig for FFmpeg compat
    return str(Path(output_path).resolve())


# ---------------------------------------------------------------------------
# FFmpeg path escaping
# ---------------------------------------------------------------------------

def escape_ass_path(path: str) -> str:
    """
    Escape an .ass file path for use inside an FFmpeg filtergraph string.

    FFmpeg subtitles filter on Windows requires:
      - Backslashes replaced with forward slashes
      - Colons in drive letter escaped as \\\\:  (e.g. C\\:/path/to/file.ass)
    """
    p = path.replace("\\", "/")
    # Escape drive-letter colon:  C:/...  →  C\\:/...
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\\\:" + p[2:]
    return p
