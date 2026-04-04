import soundfile as sf
from tts.pipeline import generate_full_audio

# Load script
with open("scripts/input.txt", "r", encoding="utf-8") as f:
    SCRIPT = f.read()

VOICE = "voices/delegent.mp3"  # change if needed

print("🚀 Starting audio generation...")

audio, sr = generate_full_audio(SCRIPT, VOICE)

sf.write("output/final_output.wav", audio, sr)

print("✅ Done! File saved at output/final_output.wav")