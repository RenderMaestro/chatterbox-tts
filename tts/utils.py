import re
import numpy as np

# TTS models degrade on very long inputs — keep chunks under this limit
MAX_CHARS = 250

def split_script(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    Split a script into TTS-safe chunks.

    Strategy:
      1. Split on blank lines (paragraph breaks) first.
      2. If any paragraph exceeds max_chars, further split it on
         sentence boundaries (. ! ?) while respecting the limit.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            chunks.extend(_split_paragraph(para, max_chars))

    return chunks


def _split_paragraph(text: str, max_chars: int) -> list[str]:
    """Split a single paragraph into sentence-aware chunks."""
    # Split after . ! ? followed by whitespace or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        # Single sentence already too long — hard-split on word boundary
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_split(sentence, max_chars))
            continue

        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current.strip())

    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort word-boundary split for extremely long sentences."""
    words = text.split()
    chunks, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = (current + " " + word).strip() if current else word
    if current:
        chunks.append(current)
    return chunks


def merge_audio(
    chunks: list[np.ndarray],
    sr: int,
    pause_sec: float = 0.3,
    breath_sec: float = 0.15,
) -> np.ndarray:
    """Concatenate audio chunks with a short silence between each."""
    pause = np.zeros(int(pause_sec * sr), dtype=np.float32)
    breath = np.zeros(int(breath_sec * sr), dtype=np.float32)
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(chunk.astype(np.float32))
        if i != len(chunks) - 1:
            parts.append(pause)
        else:
            parts.append(breath)
    return np.concatenate(parts)


def save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    """Save a float32 numpy array as a 16-bit PCM WAV file."""
    import wave, struct
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(pcm_int16)}h", *pcm_int16))
    print(f"💾 Saved → {path}")