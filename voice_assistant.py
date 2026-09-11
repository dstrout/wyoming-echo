#!/usr/bin/env python3
"""
Wyoming Satellite with LLM and TTS Integration
Connects to satellite, receives audio, transcribes via STT, sends to LLM, 
synthesizes response via TTS, and plays back on satellite.
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

# Watchdog timeout — resets state if any downstream service hangs
PROCESSING_TIMEOUT = float(os.environ.get("PROCESSING_TIMEOUT", "30.0"))

class State(Enum):
    """Handler states"""
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"

class AudioHandler:
    """Handler for satellite events that orchestrates STT, LLM, and TTS."""
    
    def __init__(self, llm_client: AsyncOpenAI, satellite_client: AsyncClient) -> None:
        self.stt_client: Optional[AsyncClient] = None
        self.audio_format = None
        self.llm_client = llm_client
        self.satellite_client = satellite_client
        
        # State management
        self.state = State.IDLE
        self.processing_task: Optional[asyncio.Task] = None
        
    async def handle_event(self, event: Event) -> bool:
        """Handle incoming events from satellite."""
        
        if event.type == "streaming-started":
            _LOGGER.info("Satellite started streaming audio")
            
        elif event.type == "streaming-stopped":
            _LOGGER.info("Satellite stopped streaming audio")
            
        elif event.type == "audio-start":
            if self.state == State.PROCESSING:
                _LOGGER.warning("Ignoring audio-start - currently processing previous request")
                return True
            
            if self.state == State.CAPTURING:
                _LOGGER.warning("Ignoring audio-start - already capturing audio")
                return True
                
            _LOGGER.info(f"Audio stream started: rate={event.data.get('rate')}, "
                        f"width={event.data.get('width')}, channels={event.data.get('channels')}")
            self.state = State.CAPTURING
            self.audio_format = {
                'rate': event.data.get('rate'),
                'width': event.data.get('width'),
                'channels': event.data.get('channels')
            }
            await self._start_transcription()
            
        elif event.type == "audio-chunk":
            if self.state == State.PROCESSING:
                _LOGGER.debug("Ignoring audio-chunk - currently processing previous request")
                return True

            # Handle first chunk in IDLE state (VAD mode - no audio-start event)
            if self.state == State.IDLE:
                # Extract audio format from first chunk
                if self.audio_format is None:
                    self.audio_format = {
                        'rate': event.data.get('rate'),
                        'width': event.data.get('width'),
                        'channels': event.data.get('channels')
                    }
                    _LOGGER.info(f"Detected audio format from first chunk: {self.audio_format}")

                # Transition to CAPTURING and start transcription
                self.state = State.CAPTURING
                await self._start_transcription()

            # Extract audio format from chunk if we don't have it yet
            if self.audio_format is None:
                self.audio_format = {
                    'rate': event.data.get('rate'),
                    'width': event.data.get('width'),
                    'channels': event.data.get('channels')
                }
                _LOGGER.info(f"Detected audio format from chunk: {self.audio_format}")

            # Forward audio chunk to STT (only in CAPTURING state)
            if self.state == State.CAPTURING and self.stt_client and event.payload:
                chunk = AudioChunk(
                    audio=event.payload,
                    rate=self.audio_format['rate'],
                    width=self.audio_format['width'],
                    channels=self.audio_format['channels']
                )
                await self.stt_client.write_event(chunk.event())
            
        elif event.type == "audio-stop":
            if self.state != State.CAPTURING:
                _LOGGER.warning(f"Received audio-stop in {self.state.value} state - ignoring")
                return True
                
            _LOGGER.info("Audio stream stopped - getting transcription...")
            self.state = State.PROCESSING
            
            # Start processing with timeout
            self.processing_task = asyncio.create_task(self._process_with_timeout())
            
        return True

    async def _process_with_timeout(self):
        """Process request with timeout to prevent stuck state."""
        try:
            await asyncio.wait_for(
                self._stop_transcription(),
                timeout=PROCESSING_TIMEOUT
            )
        except asyncio.TimeoutError:
            _LOGGER.error(f"Processing timeout after {PROCESSING_TIMEOUT}s - resetting to IDLE")
            await self._reset_state()
        except Exception as e:
            _LOGGER.error(f"Error in processing: {e}", exc_info=True)
            await self._reset_state()

    async def _reset_state(self):
        """Reset to IDLE state and clean up resources."""
        _LOGGER.info("Resetting state to IDLE")
        
        # Clean up STT client if it exists
        if self.stt_client:
            try:
                await self.stt_client.__aexit__(None, None, None)
            except:
                pass
            self.stt_client = None
        
        # Try to resume satellite
        try:
            await self._resume_satellite()
        except Exception as e:
            _LOGGER.error(f"Failed to resume satellite during reset: {e}")
        
        self.state = State.IDLE

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
        """Connect to STT service and start transcription."""
        try:
            _LOGGER.info(f"Connecting to STT service at {STT_HOST}:{STT_PORT}")
            self.stt_client = AsyncClient.from_uri(f"tcp://{STT_HOST}:{STT_PORT}")
            await self.stt_client.__aenter__()
            
            # Send transcribe request
            transcribe_event = Transcribe().event()
            await self.stt_client.write_event(transcribe_event)
            
            # Send audio-start
            audio_start = AudioStart(
                rate=self.audio_format['rate'],
                width=self.audio_format['width'],
                channels=self.audio_format['channels']
            )
            await self.stt_client.write_event(audio_start.event())
            _LOGGER.info("Started transcription session")
            
        except Exception as e:
            _LOGGER.error(f"Error starting transcription: {e}")
            # Clean up client if it was created
            if self.stt_client:
                try:
                    await self.stt_client.__aexit__(None, None, None)
                except:
                    pass
                self.stt_client = None
            self.state = State.IDLE
            raise

    async def _stop_transcription(self):
        """Stop transcription, get result, and send to LLM."""
        if not self.stt_client:
            self.state = State.IDLE
            return
            
        try:
            # Send audio-stop
            audio_stop = AudioStop()
            await self.stt_client.write_event(audio_stop.event())
            
            # Read transcript response
            _LOGGER.info("Waiting for transcript...")
            response = await self.stt_client.read_event()
            
            if response.type == "transcript":
                text = response.data.get("text", "")
                _LOGGER.info("=" * 80)
                _LOGGER.info(f"TRANSCRIPTION: {text}")
                _LOGGER.info("=" * 80)

                # Send transcript event back to satellite to stop streaming
                from wyoming.asr import Transcript
                transcript_event = Transcript(text=text).event()
                await self.satellite_client.write_event(transcript_event)
                _LOGGER.debug("Sent transcript to satellite")

                # Send to LLM and then TTS
                if text.strip():
                    await self._query_llm(text)
                else:
                    _LOGGER.warning("Empty transcript, skipping LLM")
                    self.state = State.IDLE
            else:
                _LOGGER.warning(f"Unexpected response type: {response.type}")
                self.state = State.IDLE
                
        except Exception as e:
            _LOGGER.error(f"Error getting transcription: {e}")
            raise  # Re-raise to be caught by timeout handler
        finally:
            # Always clean up STT client
            if self.stt_client:
                try:
                    await self.stt_client.__aexit__(None, None, None)
                except:
                    pass
                self.stt_client = None

    async def _query_llm(self, user_message: str):
        """Send transcript to LLM and synthesize response."""
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
            
            # Synthesize and play response (state is still PROCESSING)
            await self._synthesize_and_play(llm_response)
            
        except Exception as e:
            _LOGGER.error(f"Error querying LLM: {e}", exc_info=True)
            raise  # Re-raise to be caught by timeout handler

    async def _synthesize_and_play(self, text: str):
        """Send text to TTS and forward audio to satellite for playback."""
        try:
            # We're still in PROCESSING state, which blocks new audio capture
            _LOGGER.info("Pausing satellite for TTS playback")
            pause_event = Event(type="pause-satellite", data={})
            await self.satellite_client.write_event(pause_event)
            
            # Small delay to ensure pause takes effect
            await asyncio.sleep(0.1)
            
            _LOGGER.info(f"Connecting to TTS service at {TTS_HOST}:{TTS_PORT}")
            
            async with AsyncClient.from_uri(f"tcp://{TTS_HOST}:{TTS_PORT}") as tts_client:
                # Send synthesize request
                synthesize_event = Synthesize(text=text).event()
                await tts_client.write_event(synthesize_event)
                _LOGGER.info("Sent synthesize request to TTS")
                
                # Track audio format and data for duration calculation
                audio_format = None
                total_audio_bytes = 0
                chunk_count = 0
                
                while True:
                    event = await tts_client.read_event()
                    
                    if event is None:
                        _LOGGER.warning("TTS connection closed unexpectedly")
                        break
                    
                    if event.type == "audio-start":
                        # Extract audio format for duration calculation
                        audio_format = {
                            'rate': event.data.get('rate'),
                            'width': event.data.get('width'),
                            'channels': event.data.get('channels')
                        }
                        _LOGGER.info(f"TTS audio started: rate={audio_format['rate']}, "
                                   f"width={audio_format['width']}, channels={audio_format['channels']}")
                        # Forward to satellite
                        await self.satellite_client.write_event(event)
                        
                    elif event.type == "audio-chunk":
                        chunk_count += 1
                        # Track payload size for duration calculation
                        if event.payload:
                            total_audio_bytes += len(event.payload)
                        # Forward audio chunk to satellite
                        await self.satellite_client.write_event(event)
                        
                    elif event.type == "audio-stop":
                        _LOGGER.info(f"TTS audio stopped (forwarded {chunk_count} chunks, {total_audio_bytes} bytes)")
                        # Forward to satellite
                        await self.satellite_client.write_event(event)
                        break
                    
                    else:
                        _LOGGER.debug(f"Received TTS event: {event.type}")
                
                # Calculate playback duration based on audio format and size
                if audio_format and total_audio_bytes > 0:
                    bytes_per_second = audio_format['rate'] * audio_format['width'] * audio_format['channels']
                    duration_seconds = total_audio_bytes / bytes_per_second
                    # Add 500ms safety margin for audio buffer and system latency
                    wait_time = duration_seconds + 0.5
                    
                    _LOGGER.info(f"Calculated playback duration: {duration_seconds:.2f}s "
                               f"(waiting {wait_time:.2f}s with safety margin)")
                else:
                    # Fallback if we couldn't calculate
                    wait_time = 2.0
                    _LOGGER.warning(f"Could not calculate duration, using fallback wait time: {wait_time}s")
                
                # Wait for audio to finish playing
                await asyncio.sleep(wait_time)
            
            # Resume satellite - ready for next command
            await self._resume_satellite()
            self.state = State.IDLE
            _LOGGER.info("Satellite resumed - ready for next command")
                    
        except Exception as e:
            _LOGGER.error(f"Error synthesizing speech: {e}", exc_info=True)
            raise  # Re-raise to be caught by timeout handler

async def main():
    """Main entry point."""
    _LOGGER.info("Starting Wyoming Satellite Voice Assistant")
    _LOGGER.info(f"Satellite: {SATELLITE_HOST}:{SATELLITE_PORT}")
    _LOGGER.info(f"STT: {STT_HOST}:{STT_PORT}")
    _LOGGER.info(f"TTS: {TTS_HOST}:{TTS_PORT}")
    _LOGGER.info(f"LLM: {LLM_BASE_URL} (model: {LLM_MODEL})")
    
    # Initialize LLM client
    llm_client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )
    
    # Connect to satellite
    async with AsyncClient.from_uri(f"tcp://{SATELLITE_HOST}:{SATELLITE_PORT}") as satellite_client:
        _LOGGER.info("Connected to satellite")
        
        # Tell satellite server is ready
        run_satellite_event = Event(type="run-satellite", data={})
        await satellite_client.write_event(run_satellite_event)
        _LOGGER.info("Sent run-satellite command")
        
        # Send run-pipeline command to start streaming audio
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
        _LOGGER.info("Sent run-pipeline command (wake->asr stage, restart_on_end=True)")
        
        # Create handler with satellite client reference
        handler = AudioHandler(llm_client, satellite_client)
        
        # Listen for events
        _LOGGER.info("Voice assistant ready! Speak into the microphone...")
        _LOGGER.info("Flow: Speech → STT → LLM → TTS → Playback")
        try:
            while True:
                try:
                    event = await satellite_client.read_event()
                    
                    if event is None:
                        _LOGGER.warning("Connection closed by satellite")
                        break
                    
                    await handler.handle_event(event)
                except Exception as e:
                    _LOGGER.error(f"Error handling event: {e}", exc_info=True)
                    # Try to reset state on error
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

