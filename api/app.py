from fastapi import FastAPI
from tts.pipeline import generate_full_audio
import soundfile as sf
import uuid

app = FastAPI()

@app.post("/generate")
def generate(text: str):
    audio, sr = generate_full_audio(text)

    filename = f"output/{uuid.uuid4()}.wav"
    sf.write(filename, audio, sr)

    return {"file": filename}