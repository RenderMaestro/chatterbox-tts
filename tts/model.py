import torch
from chatterbox.tts import ChatterboxTTS

_model = None

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_model() -> ChatterboxTTS:
    global _model
    if _model is None:
        device = get_device()
        print(f"🔄 Loading Chatterbox model on {device.upper()}...")
        _model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Model loaded.")
    return _model

def unload_model():
    """Free VRAM/RAM when the model is no longer needed."""
    global _model
    _model = None