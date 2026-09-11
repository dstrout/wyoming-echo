# wyoming-echo

A small async Python orchestrator that turns a [Wyoming](https://github.com/OHF-Voice/wyoming) satellite into a fully-local voice assistant driven by your own LLM — no Home Assistant required.

```
[Wyoming Satellite] ⇄ wyoming-echo ⇄ [Wyoming STT] → [LLM] → [Wyoming TTS]
```

## Problem

The [Home Assistant Voice Assistant](https://www.home-assistant.io/voice_control/) stack is powerful but assumes you're running Home Assistant as the pipeline coordinator. If you already own the pieces — a Wyoming satellite (e.g. a Raspberry Pi + ReSpeaker HAT), a Wyoming STT service like `wyoming-faster-whisper`, a Wyoming TTS service like `wyoming-piper`, and a local LLM behind an OpenAI-compatible endpoint — there is no minimal glue that just chains them together.

Two operational problems also show up the moment you try to write that glue naively:

1. **Race conditions.** A fast LLM can return a response while audio capture is still finishing, corrupting the session.
2. **Feedback loops.** On microphones without hardware acoustic echo cancellation (like the ReSpeaker 2Mic HAT), the mic hears the speaker and the assistant starts talking to itself.

## Solution

`wyoming-echo` is a ~460-line orchestrator that:

- Speaks the Wyoming protocol on both ends (satellite ⇄ STT/TTS) and calls any OpenAI-compatible chat completions endpoint in between.
- Runs a **three-state machine** (`IDLE` → `CAPTURING` → `PROCESSING`) with guards so late audio during processing is dropped rather than corrupting the next turn.
- Prevents feedback loops in software by **pausing the satellite** during TTS playback, computing the exact playback duration from the PCM byte stream (`bytes / (rate × width × channels)`), waiting that duration plus a 500 ms safety margin, then resuming.
- Wraps the whole processing step in a **30-second watchdog** — if any downstream service hangs, state resets to `IDLE` and the satellite is resumed automatically.

Two variants are included:

| Script | Speech-end detection | Extra dependency |
| --- | --- | --- |
| `voice_assistant.py` | VAD in the satellite (via `wyoming-satellite --vad`) | — |
| `voice_assistant_vad.py` | VAD in the orchestrator (Silero) — lower latency, no satellite VAD needed | `pysilero-vad` |

## Install

Requires Python 3.10+ and running Wyoming STT / TTS / satellite services.

```bash
git clone https://github.com/dstrout/wyoming-echo.git
cd wyoming-echo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Wyoming services this talks to (bring your own):

- **Satellite** — [wyoming-satellite](https://github.com/rhasspy/wyoming-satellite) on a Raspberry Pi or similar
- **STT** — [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper)
- **TTS** — [wyoming-piper](https://github.com/rhasspy/wyoming-piper) (or any Wyoming TTS)
- **LLM** — any OpenAI-compatible endpoint: [llama.cpp](https://github.com/ggerganov/llama.cpp) server, [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/) (via its OpenAI-compatible mode), vLLM, etc.

## Usage

All configuration is via environment variables with sensible defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SATELLITE_HOST` | `satellite.local` | Wyoming satellite hostname |
| `SATELLITE_PORT` | `10700` | |
| `STT_HOST` / `STT_PORT` | `localhost` / `10300` | Wyoming STT service |
| `TTS_HOST` / `TTS_PORT` | `localhost` / `10200` | Wyoming TTS service |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3-30b` | Model name to send to the LLM |
| `LLM_API_KEY` | `not-needed` | Only matters for hosted endpoints |
| `SYSTEM_PROMPT` | *(generic assistant prompt)* | System message prepended to each turn |
| `PROCESSING_TIMEOUT` | `30.0` | Watchdog seconds |
| `VAD_THRESHOLD` | `0.5` | (VAD variant only) Silero confidence threshold |
| `VAD_SILENCE_DURATION` | `0.8` | (VAD variant only) Seconds of silence to end an utterance |

Run against a satellite whose VAD is enabled with `--vad`:

```bash
SATELLITE_HOST=my-pi.local \
STT_HOST=my-server.local \
TTS_HOST=my-server.local \
python voice_assistant.py
```

Or run the orchestrator-side VAD variant (leave `--vad` off on the satellite):

```bash
pip install pysilero-vad
python voice_assistant_vad.py
```

You should see:

```
INFO - Connected to satellite
INFO - Sent run-pipeline command (wake->asr stage, restart_on_end=True)
INFO - Voice assistant ready! Speak into the microphone...
INFO - TRANSCRIPTION: what's the weather like
INFO - LLM RESPONSE: I don't have access to real-time data...
INFO - TTS audio stopped (forwarded 47 chunks, 266240 bytes)
INFO - Calculated playback duration: 6.03s (waiting 6.53s with safety margin)
INFO - Satellite resumed - ready for next command
```

## Known limitations

- **No conversation history** — each turn is independent.
- **No wake word** — VAD-only. Pair with [wyoming-openwakeword](https://github.com/rhasspy/wyoming-openwakeword) on the satellite if you want one.
- **No barge-in** — you can't interrupt the assistant mid-response.
- **No streaming LLM tokens** — waits for the full LLM response before starting TTS.

These are all deliberate simplifications, not bugs — the goal was the smallest possible orchestrator that reliably closes the loop.

## License

MIT — see [LICENSE](LICENSE).
