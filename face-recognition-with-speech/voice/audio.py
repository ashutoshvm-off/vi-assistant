"""
Audio / Text-to-Speech module.
Uses edge-tts (Microsoft Azure Neural Voices) — free, no API key needed.
Install: pip install edge-tts pygame
"""

import asyncio
import os
import queue
import tempfile
import threading
import time

import edge_tts
import pygame

DISABLE_LOCAL_TTS = os.getenv("COORD_DISABLE_LOCAL_TTS", "0") == "1"

# ---------------------------------------------------------------------------
# Initialise pygame mixer once; record whether it succeeded.
# ---------------------------------------------------------------------------
_audio_available = False
try:
    pygame.mixer.init()
    _audio_available = True
except pygame.error as e:
    print(f"[audio] Warning: pygame init failed ({e}) — TTS disabled.")

# Run `edge-tts --list-voices` to browse options.
VOICE = "en-US-JennyNeural"

# Single background worker + queue so we never pile up waiting threads.
_speech_queue: queue.Queue = queue.Queue()


def _worker() -> None:
    """Background thread: drain the speech queue one item at a time."""
    while True:
        text = _speech_queue.get()
        if text is None:          # sentinel → shutdown
            break
        _speak_blocking(text)
        _speech_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _speak_blocking(text: str) -> None:
    """Generate TTS audio and play it synchronously (called from worker)."""
    if not _audio_available:
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    filename = tmp.name

    try:
        asyncio.run(edge_tts.Communicate(text, VOICE).save(filename))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[audio] TTS error: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak_async(text: str) -> None:
    """Enqueue *text* to be spoken in the background worker thread."""
    if DISABLE_LOCAL_TTS:
        return
    if _audio_available:
        _speech_queue.put(text)


def cleanup() -> None:
    """Gracefully shut down the audio worker and release pygame resources."""
    _speech_queue.put(None)          # send sentinel to stop worker
    _worker_thread.join(timeout=2)
    if _audio_available:
        pygame.mixer.quit()