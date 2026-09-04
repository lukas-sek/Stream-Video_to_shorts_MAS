---
project: Autonomous Stream-to-Shorts Engine
status: planning
tags:
  - ai
  - multiagent
  - python
  - video-automation
hardware_budget: 16GB RAM (Local Setup)
---

## Phase 1: Environment & Tooling Setup
- [x] **Python Environment initialisieren**
	- [x] Virtual Environment mit Python 3.10+ anlegen (`.venv`). — Python 3.13.2, `.venv` erstellt
	- [x] Kern-Libraries installieren: `yt-dlp`, `streamlink`, `faster-whisper`, `torch`, `ffmpeg-python`, `pydantic`, `openai`. — torch 2.14.0+cpu, alle imports OK
- [x] **Lokale LLM-Infrastruktur einrichten**
	- [x] Ollama installieren und konfigurieren. — Ollama 0.32.11
	- [x] `qwen2.5:7b-instruct-q4_k_m` herunterladen und RAM-Auslastung (< 5 GB) prüfen. — 4.7 GB, RAM-Nutzung nach Inferenz: 14.1/15.8 GB
	- [x] Fallback-Modell `llama3.2:3b` für High-Speed-Klassifikation testen. — 2.0 GB, gepullt
	- [x] `num_ctx 8192` in Modelfile/Ollama-Config hinterlegen, um KV-Cache-Explosion zu verhindern. — `shorts-llm` registriert via config/Modelfile.qwen25

---

## Phase 2: Ingestion & Signal Detection (Tier 1 & 2)
- [x] **Twitch IRC Ingestion (Scout Agent)**
	- [x] Verbindung zu Twitch IRC via Websocket/Socket aufbauen (anonym oder Bot-Token). — `agents/scout_agent.py`, anonym via `justinfan<random>`
	- [x] Chat-Message-Buffer implementieren (Nachrichten pro 5-Sekunden-Fenster zählen). — `collections.deque`, 5s-Fenster
	- [x] Rolling-Average-Algorithmus zur Erkennung von Velocity-Spikes und Emote-Floods schreiben. — 30-Fenster-Baseline, 3.0x-Multiplikator
- [x] **Audio-Energy & VAD Filter (Detector Agent)**
	- [x] Live-Segment-Download (60–90 Sekunden) via `streamlink` / `yt-dlp` bei getriggertem Chat-Spike. — `agents/graph.py` Node `download_segment`, 75s via streamlink|ffmpeg
	- [x] Audio-Stream mit `ffmpeg` als WAV (16kHz Mono) extrahieren. — Node `extract_audio`, ffmpeg-python
	- [x] `silero-vad` integrieren, um Sprachpausen und Stillebereiche zu taggen. — Node `run_vad`, silero-vad 6.2.1
	- [x] Audio-Peak-Detection (RMS/Pitch) zur Validierung von Gelächter/Lautstärke-Spitzen aufsetzen. — Node `analyze_rms`, numpy 500ms sliding window

---

## Phase 3: ASR & LLM Packaging Agent (Tier 3)
- [x] **Whisper Speech-to-Text Pipeline**
	- [x] `faster-whisper` mit `compute_type="int8"` und Model `base.en` oder `small.en` integrieren. — Node `transcribe_audio`, WhisperModel base.en int8 CPU
	- [x] Word-Level-Timestamps für das candidate audio exportieren. — `word_timestamps: [{word, start, end, probability}]`
	- [x] Explizites Memory-Cleanup nach Transkription (`torch.cuda.empty_cache()` / GC Trigger). — `del model; gc.collect()` nach Transkription
- [x] **Pydantic Schema & Prompt Engineering (Packaging Agent)**
	- [x] Pydantic-Klasse definieren: `ClipMetadata` (Titel, Hook-Text, Virality-Score, Tags). — `models/signals.py`
	- [x] System-Prompt für Qwen 2.5 7B aufsetzen (Fokus auf prägnante 5–8 Wörter Hooks). — `agents/prompts.py` mit 2 Few-Shot-Beispielen
	- [x] Structured Outputs Parsing via lokaler Ollama-OpenAI-Schnittstelle implementieren. — Node `package_clip`, json_object mode, `_LLMPackage` Pydantic-Parser
	- [x] Threshold-Filter: Verwerfen von Clips mit Virality-Score unter Benchmark. — `evaluate_virality` edge, `VIRALITY_THRESHOLD=6.0`

---

## Phase 4: Autonomous Video Editor Agent (Tier 4)
- [x] **9:16 Reframing & Layout Engine**
	- [x] Facecam-Erkennung via `MediaPipe` oder statische ROI-Presets (Region of Interest) definieren. — Statische ROI-Presets: `center_crop` (Standard) und `stacked` (Facecam oben, Gameplay unten) via `VIDEO_LAYOUT` env var
	- [x] FFmpeg-Filtergraph schreiben: Video aufteilen, Gameplay-Zentrum (unten) und Facecam (oben) im 9:16 Format vertikal stacken. — Node `render_video`, single-pass filter_complex mit select+crop+vstack+subtitles
- [x] **Dead-Air Trimming**
	- [x] Schnittlisten basierend auf VAD-Pausen (> 400ms) generieren. — Node `prepare_edit`, gap_threshold=400ms, merge overlapping segments
	- [x] Video mit `ffmpeg` an Keyframes schneiden, um Pacing zu erhöhen. — `select`/`aselect` filter mit `setpts=N/FRAME_RATE/TB`
- [x] **Dynamic Subtitles (Burn-in)**
	- [x] Word-Level-Timestamps in `.ass`-Format mit Custom-Styles (Schriftart, Border, Pop-In-Animation) konvertieren. — `agents/subtitles.py`: Impact 72px, white, 4px outline, `\fad(150,50)` per word, timestamp re-mapped after dead-air removal
	- [x] Subtitles per FFmpeg `subtitles`-Filter in den finalen MP4-Export rendern. — burn-in via `subtitles='file.ass'` im filter_complex, Output: `output/final/*.mp4`

---

## Phase 5: Publishing & Optimization Engine
- [ ] **Publishing Pipeline**
	- [ ] YouTube Data API v3 Upload-Skript schreiben (OAuth 2.0 Refresh-Token Flow).
	- [ ] TikTok Content Posting API / Instagram Reels API Anbindung aufsetzen.
	- [ ] Upload-Queue und Rate-Limiting-Scheduler einrichten (z. B. max. 2 Uploads pro Tag).
- [ ] **Micro-Metric Feedback Loop (Bandit Agent)**
	- [ ] SQLite-Datenbank für Clip-Metadaten, Prompt-Varianten und Performance-Logs anlegen.
	- [ ] Cronjob/Skript zur Erfassung von 24h-Metriken (APV, Viewed vs. Swiped Away).
	- [ ] Contextual Bandit (LinUCB / Thompson Sampling) implementieren zur automatischen Anpassung von:
		- [ ] Hook-Formulierungen
		- [ ] VAD-Schnitt-Schwellenwerten (Pacing)
		- [ ] Chat-Spike-Trigger-Empfindlichkeit

---

## Phase 6: Week 1 MVP Sprint
- [ ] [ ] Standalone Python-Skript für Twitch IRC Chat Spike Logger erstellen.
- [ ] [ ] 60s Clip-Download via `yt-dlp` automatisieren.
- [ ] [ ] Statisches vertikales 9:16 FFmpeg-Crop-Preset ausführen.
- [ ] [ ] `faster-whisper` + ASS Subtitle Burn-In testen.
- [ ] [ ] Lokales Output-Video manuell reviewen.