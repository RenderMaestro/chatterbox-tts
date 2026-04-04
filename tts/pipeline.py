import time
import numpy as np
from tts.generate import generate_chunk, DEFAULT_EXAGGERATION, DEFAULT_CFG_WEIGHT
from tts.utils import split_script, merge_audio, save_wav

def generate_full_audio(
    script: str,
    voice: str | None = None,
    output_path: str | None = None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    pause_sec: float = 0.3,
    skip_failed_chunks: bool = False,
) -> tuple:
    """
    Convert a full script to audio.

    Args:
        script:             Raw script text (paragraphs separated by blank lines).
        voice:              Path to a WAV file for voice cloning (optional).
        output_path:        If provided, saves the result as a WAV file.
        exaggeration:       Expressiveness strength (0.0–1.0).
        cfg_weight:         Classifier-free guidance weight.
        pause_sec:          Silence gap between chunks in seconds.
        skip_failed_chunks: If True, logs errors and continues; otherwise raises.

    Returns:
        (numpy float32 array, sample_rate int)
    """
    chunks = split_script(script)
    total = len(chunks)
    print(f"📄 Script split into {total} chunk(s).")

    audio_chunks: list[np.ndarray] = []
    sr = None
    t0 = time.time()

    for i, chunk in enumerate(chunks, start=1):
        print(f"🎙️  [{i}/{total}] {chunk[:60]}{'…' if len(chunk) > 60 else ''}")
        try:
            audio, sr = generate_chunk(
                chunk,
                voice=voice,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            audio_chunks.append(audio)
        except Exception as exc:
            if skip_failed_chunks:
                print(f"  ❌ Skipping chunk {i} due to error: {exc}")
                continue
            raise

    if not audio_chunks:
        raise RuntimeError("No audio chunks were generated successfully.")

    final_audio = merge_audio(audio_chunks, sr, pause_sec=pause_sec)
    elapsed = time.time() - t0
    duration = len(final_audio) / sr
    print(f"✅ Done — {duration:.1f}s of audio in {elapsed:.1f}s wall time.")

    if output_path:
        save_wav(final_audio, sr, output_path)

    return final_audio, sr