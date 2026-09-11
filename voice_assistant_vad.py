#!/usr/bin/env python3
"""
Wyoming Satellite with LLM and TTS Integration + VAD in orchestrator
Implements VAD locally to detect end of speech quickly
"""

import asyncio
import logging
import os
from typing import Optional
from enum import Enum
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.tts import Synthesize
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_LOGGER = logging.getLogger(__name__)

# Network configuration (override via environment variables)
SATELLITE_HOST = os.environ.get("SATELLITE_HOST", "satellite.local")
SATELLITE_PORT = int(os.environ.get("SATELLITE_PORT", "10700"))
STT_HOST = os.environ.get("STT_HOST", "localhost")
STT_PORT = int(os.environ.get("STT_PORT", "10300"))
TTS_HOST = os.environ.get("TTS_HOST", "localhost")
TTS_PORT = int(os.environ.get("TTS_PORT", "10200"))

# LLM configuration (OpenAI-compatible endpoint)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-30b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful voice assistant. Keep responses concise and conversational.",
)

# VAD tuning
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_DURATION = float(os.environ.get("VAD_SILENCE_DURATION", "0.8"))
PROCESSING_TIMEOUT = float(os.environ.get("PROCESSING_TIMEOUT", "30.0"))

class State(Enum):
    """Handler states"""
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"

class AudioHandler:
    """Handler with built-in VAD."""

    def __init__(self, llm_client: AsyncOpenAI, satellite_client: AsyncClient) -> None:
        self.stt_client: Optional[AsyncClient] = None
        self.audio_format = None
        self.llm_client = llm_client
        self.satellite_client = satellite_client

        # State management
        self.state = State.IDLE
        self.processing_task: Optional[asyncio.Task] = None

        # VAD state
        self.vad = None
        self.silence_chunks = 0
        self.silence_threshold_chunks = 0  # Will be calculated

    def _init_vad(self):
        """Initialize Silero VAD."""
        try:
            from pysilero_vad import SileroVoiceActivityDetector
            self.vad = SileroVoiceActivityDetector()
            _LOGGER.info("VAD initialized")
        except ImportError:
            _LOGGER.warning("pysilero-vad not installed, using timeout-based detection")
            self.vad = None

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming events from satellite."""

        if event.type == "audio-chunk":
            if self.state == State.PROCESSING:
                _LOGGER.debug("Ignoring audio-chunk - currently processing")
                return True

            # Handle first chunk
            if self.state == State.IDLE:
                if self.audio_format is None:
                    self.audio_format = {
                        'rate': event.data.get('rate'),
                        'width': event.data.get('width'),
                        'channels': event.data.get('channels')
                    }
                    _LOGGER.info(f"Detected audio format: {self.audio_format}")

                    # Calculate silence threshold in chunks
                    # Each chunk is typically 1024 samples
                    samples_per_second = self.audio_format['rate']
                    samples_per_chunk = 1024
                    chunks_per_second = samples_per_second / samples_per_chunk
                    self.silence_threshold_chunks = int(VAD_SILENCE_DURATION * chunks_per_second)
                    _LOGGER.info(f"Silence threshold: {self.silence_threshold_chunks} chunks ({VAD_SILENCE_DURATION}s)")

                    # Initialize VAD
                    if self.vad is None:
                        self._init_vad()

            # Check VAD on chunk
            if event.payload and self.vad:
                speech_detected = self.vad(event.payload) >= VAD_THRESHOLD

                if self.state == State.IDLE and speech_detected:
                    # Speech started!
                    _LOGGER.info("VAD: Speech detected, starting capture")
                    self.state = State.CAPTURING
                    self.silence_chunks = 0
                    await self._start_transcription()

                elif self.state == State.CAPTURING:
                    if not speech_detected:
                        self.silence_chunks += 1
                        if self.silence_chunks >= self.silence_threshold_chunks:
                            # Silence detected, stop capture
                            _LOGGER.info(f"VAD: Silence detected ({self.silence_chunks} chunks), stopping capture")
                            self.state = State.PROCESSING
                            self.processing_task = asyncio.create_task(self._process_with_timeout())
                            return True
                    else:
                        # Reset silence counter if speech detected
                        self.silence_chunks = 0

            # Forward chunk to STT if capturing
            if self.state == State.CAPTURING and self.stt_client and event.payload:
                chunk = AudioChunk(
                    audio=event.payload,
                    rate=self.audio_format['rate'],
                    width=self.audio_format['width'],
                    channels=self.audio_format['channels']
                )
                await self.stt_client.write_event(chunk.event())

        return True

    async def _process_with_timeout(self):
        """Process request with timeout."""
        try:
            await asyncio.wait_for(
                self._stop_transcription(),
                timeout=PROCESSING_TIMEOUT
            )
        except asyncio.TimeoutError:
            _LOGGER.error(f"Processing timeout after {PROCESSING_TIMEOUT}s")
            await self._reset_state()
        except Exception as e:
            _LOGGER.error(f"Error in processing: {e}", exc_info=True)
            await self._reset_state()

    async def _reset_state(self):
        """Reset to IDLE state."""
        _LOGGER.info("Resetting state to IDLE")

        if self.stt_client:
            try:
                await self.stt_client.__aexit__(None, None, None)
            except:
                pass
            self.stt_client = None

        try:
            await self._resume_satellite()
        except Exception as e:
            _LOGGER.error(f"Failed to resume satellite: {e}")

        self.state = State.IDLE
        self.silence_chunks = 0

    async def _resume_satellite(self):
        """Resume satellite for next command."""
        _LOGGER.info("Resuming satellite")
        run_satellite_event = Event(type="run-satellite", data={})
        await self.satellite_client.write_event(run_satellite_event)

        run_pipeline_event = Event(
            type="run-pipeline",
            data={
                "start_stage": "wake",
                "end_stage": "asr",
                "restart_on_end": True,
                "snd_format": {
                    "rate": 22050,
                    "width": 2,
                    "channels": 1
                }
            }
        )
        await self.satellite_client.write_event(run_pipeline_event)

    async def _start_transcription(self):
        """Connect to STT and start transcription."""
        try:
            _LOGGER.info(f"Connecting to STT service at {STT_HOST}:{STT_PORT}")
            self.stt_client = AsyncClient.from_uri(f"tcp://{STT_HOST}:{STT_PORT}")
            await self.stt_client.__aenter__()

            transcribe_event = Transcribe().event()
            await self.stt_client.write_event(transcribe_event)

            audio_start = AudioStart(
                rate=self.audio_format['rate'],
                width=self.audio_format['width'],
                channels=self.audio_format['channels']
            )
            await self.stt_client.write_event(audio_start.event())
            _LOGGER.info("Started transcription session")

        except Exception as e:
            _LOGGER.error(f"Error starting transcription: {e}")
            if self.stt_client:
                try:
                    await self.stt_client.__aexit__(None, None, None)
                except:
                    pass
                self.stt_client = None
            self.state = State.IDLE
            raise

    async def _stop_transcription(self):
        """Stop transcription and get result."""
        if not self.stt_client:
            self.state = State.IDLE
            return

        try:
            audio_stop = AudioStop()
            await self.stt_client.write_event(audio_stop.event())

            _LOGGER.info("Waiting for transcript...")
            response = await self.stt_client.read_event()

            if response.type == "transcript":
                text = response.data.get("text", "")
                _LOGGER.info("=" * 80)
                _LOGGER.info(f"TRANSCRIPTION: {text}")
                _LOGGER.info("=" * 80)

                if text.strip():
                    await self._query_llm(text)
                else:
                    _LOGGER.warning("Empty transcript")
                    self.state = State.IDLE
            else:
                _LOGGER.warning(f"Unexpected response: {response.type}")
                self.state = State.IDLE

        except Exception as e:
            _LOGGER.error(f"Error getting transcription: {e}")
            raise
        finally:
            if self.stt_client:
                try:
                    await self.stt_client.__aexit__(None, None, None)
                except:
                    pass
                self.stt_client = None

    async def _query_llm(self, user_message: str):
        """Send to LLM and synthesize response."""
        try:
            _LOGGER.info("Sending to LLM...")

            response = await self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200,
            )

            llm_response = response.choices[0].message.content

            _LOGGER.info("=" * 80)
            _LOGGER.info(f"LLM RESPONSE: {llm_response}")
            _LOGGER.info("=" * 80)
            print(f"\n>>> USER: {user_message}")
            print(f">>> ASSISTANT: {llm_response}\n")

            await self._synthesize_and_play(llm_response)

        except Exception as e:
            _LOGGER.error(f"Error querying LLM: {e}", exc_info=True)
            raise

    async def _synthesize_and_play(self, text: str):
        """Send to TTS and play."""
        try:
            _LOGGER.info("Pausing satellite for TTS playback")
            pause_event = Event(type="pause-satellite", data={})
            await self.satellite_client.write_event(pause_event)

            await asyncio.sleep(0.1)

            _LOGGER.info(f"Connecting to TTS service at {TTS_HOST}:{TTS_PORT}")

            async with AsyncClient.from_uri(f"tcp://{TTS_HOST}:{TTS_PORT}") as tts_client:
                synthesize_event = Synthesize(text=text).event()
                await tts_client.write_event(synthesize_event)
                _LOGGER.info("Sent synthesize request to TTS")

                audio_format = None
                total_audio_bytes = 0
                chunk_count = 0

                while True:
                    event = await tts_client.read_event()

                    if event is None:
                        _LOGGER.warning("TTS connection closed")
                        break

                    if event.type == "audio-start":
                        audio_format = {
                            'rate': event.data.get('rate'),
                            'width': event.data.get('width'),
                            'channels': event.data.get('channels')
                        }
                        _LOGGER.info(f"TTS audio started: {audio_format}")
                        await self.satellite_client.write_event(event)

                    elif event.type == "audio-chunk":
                        chunk_count += 1
                        if event.payload:
                            total_audio_bytes += len(event.payload)
                        await self.satellite_client.write_event(event)

                    elif event.type == "audio-stop":
                        _LOGGER.info(f"TTS stopped ({chunk_count} chunks, {total_audio_bytes} bytes)")
                        await self.satellite_client.write_event(event)
                        break

                if audio_format and total_audio_bytes > 0:
                    bytes_per_second = audio_format['rate'] * audio_format['width'] * audio_format['channels']
                    duration_seconds = total_audio_bytes / bytes_per_second
                    wait_time = duration_seconds + 0.5
                    _LOGGER.info(f"Waiting {wait_time:.2f}s for playback")
                else:
                    wait_time = 2.0

                await asyncio.sleep(wait_time)

            await self._resume_satellite()
            self.state = State.IDLE
            _LOGGER.info("Ready for next command")

        except Exception as e:
            _LOGGER.error(f"Error in TTS: {e}", exc_info=True)
            raise

async def main():
    """Main entry point."""
    _LOGGER.info("Starting Wyoming Satellite Voice Assistant with VAD")
    _LOGGER.info(f"Satellite: {SATELLITE_HOST}:{SATELLITE_PORT}")
    _LOGGER.info(f"STT: {STT_HOST}:{STT_PORT}")
    _LOGGER.info(f"TTS: {TTS_HOST}:{TTS_PORT}")
    _LOGGER.info(f"LLM: {LLM_BASE_URL} (model: {LLM_MODEL})")
    _LOGGER.info(f"VAD: threshold={VAD_THRESHOLD}, silence={VAD_SILENCE_DURATION}s")

    llm_client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

    async with AsyncClient.from_uri(f"tcp://{SATELLITE_HOST}:{SATELLITE_PORT}") as satellite_client:
        _LOGGER.info("Connected to satellite")

        run_satellite_event = Event(type="run-satellite", data={})
        await satellite_client.write_event(run_satellite_event)

        run_pipeline_event = Event(
            type="run-pipeline",
            data={
                "start_stage": "wake",
                "end_stage": "asr",
                "restart_on_end": True,
                "snd_format": {
                    "rate": 22050,
                    "width": 2,
                    "channels": 1
                }
            }
        )
        await satellite_client.write_event(run_pipeline_event)
        _LOGGER.info("Sent run-pipeline command")

        handler = AudioHandler(llm_client, satellite_client)

        _LOGGER.info("Voice assistant ready! Speak into the microphone...")
        _LOGGER.info("Flow: Speech → VAD → STT → LLM → TTS → Playback")
        try:
            while True:
                try:
                    event = await satellite_client.read_event()

                    if event is None:
                        _LOGGER.warning("Connection closed")
                        break

                    await handler.handle_event(event)
                except Exception as e:
                    _LOGGER.error(f"Error handling event: {e}", exc_info=True)
                    await handler._reset_state()
                    continue

        except KeyboardInterrupt:
            _LOGGER.info("Shutting down...")
        except Exception as e:
            _LOGGER.error(f"Error in main loop: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
