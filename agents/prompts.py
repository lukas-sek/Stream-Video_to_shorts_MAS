"""
Prompt templates for the Phase 3 Packaging Agent (Qwen 2.5 7B via Ollama).

The LLM receives a transcript and must return a strict JSON object with:
  title         — 5-8 word punchy clip title
  hook_text     — 5-8 word first-caption hook (imperative or question)
  virality_score — float 0.0-10.0 (calibrated by few-shot examples)
  tags          — list of 3-6 relevant content tags

Two few-shot examples anchor the virality scale:
  • High virality (9.2): 1-vs-5 ACE clutch with huge chat reaction
  • Low virality  (2.8): routine play with little emotional weight
"""

SYSTEM_PROMPT = """\
You are a viral-clip analyst for gaming live streams.
Given a transcript from a 75-second Twitch clip, your job is to:
1. Write a punchy 5-8 word TITLE for the clip.
2. Write a punchy 5-8 word HOOK TEXT to use as the first burned-in subtitle \
(imperative mood or question, e.g. "He did the impossible" or "Can he survive?").
3. Score the clip's VIRALITY from 0.0 to 10.0 based on:
   - Emotional intensity (clutch, fail, rage, hype, funny)
   - Chat signal (the transcript may reference crowd reaction)
   - Pacing and density of action
4. Provide 3-6 lowercase content TAGS relevant to the clip.

Respond ONLY with a single JSON object — no markdown, no explanation:
{
  "title": "<5-8 words>",
  "hook_text": "<5-8 words>",
  "virality_score": <float 0.0-10.0>,
  "tags": ["tag1", "tag2", "tag3"]
}
"""

# ---------------------------------------------------------------------------
# Few-shot examples (included as user/assistant turns before the real input)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    # --- Example 1: High-virality clutch ---
    {
        "role": "user",
        "content": (
            "Transcript:\n"
            "\"Oh my god, he's the last one alive. Four enemies left. He's low. He hits the corner. "
            "Headshot! Another one down. He's reloading on the fly — doesn't stop. "
            "THIRD kill! The crowd is INSANE. One more. He holds the angle — "
            "ONE TAP. The whole team is dead. Chat is exploding. OMEGALUL KEKW PogChamp "
            "Pog Pog Pog. That was a 1v4 clutch on 12 HP.\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"title": "Insane 1v4 Clutch on 12 HP",'
            ' "hook_text": "He wiped them on 12 HP",'
            ' "virality_score": 9.2,'
            ' "tags": ["clutch", "1v4", "fps", "gaming", "highlight"]}'
        ),
    },
    # --- Example 2: Low-virality routine play ---
    {
        "role": "user",
        "content": (
            "Transcript:\n"
            "\"Alright, so I'm just farming here. Nothing crazy going on. "
            "Got a kill there — standard. Backing off to base, need to buy items. "
            "Chat is pretty quiet. Just a normal mid-game phase, managing resources.\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"title": "Calm Mid-Game Resource Management",'
            ' "hook_text": "Just a quiet farming session",'
            ' "virality_score": 2.8,'
            ' "tags": ["farming", "moba", "gameplay", "strategy"]}'
        ),
    },
]


def build_messages(transcript: str) -> list[dict]:
    """
    Assemble the full messages list for the Ollama API call:
      [system] + [few-shot user/assistant pairs] + [real user input]
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append(
        {
            "role": "user",
            "content": f"Transcript:\n\"{transcript.strip()}\"",
        }
    )
    return messages
