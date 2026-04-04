import time
from tts.model import get_model

# Defaults — tweak here or override per call
DEFAULT_EXAGGERATION = 0.75   # emotional expressiveness (0.0–1.0)
DEFAULT_CFG_WEIGHT   = 0.35   # classifier-free guidance strength

def generate_chunk(
    text: str,
    voice: str | None = None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    retries: int = 2,
) -> tuple:
    """
    Generate audio for a single text chunk.

    Returns:
        (numpy array of float32, sample_rate int)

    Retries up to `retries` times on transient errors before raising.
    """
    model = get_model()
    last_exc = None

    for attempt in range(1, retries + 2):  # +2 so retries=2 → 3 total attempts
        try:
            wav = model.generate(
                text,
                audio_prompt_path=voice,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            return wav.squeeze().cpu().numpy(), model.sr

        except Exception as exc:
            last_exc = exc
            if attempt <= retries:
                wait = attempt * 1.5
                print(f"  ⚠️  Attempt {attempt} failed ({exc}). Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"generate_chunk failed after {retries + 1} attempts."
    ) from last_exc