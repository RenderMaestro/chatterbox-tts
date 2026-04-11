import soundfile as sf
from tts.pipeline import generate_full_audio

# Load script
with open("scripts/demo.txt", "r", encoding="utf-8") as f:
    SCRIPT = f.read()

VOICE = "voices/alex.mp3"  # change if needed

print("🚀 Starting audio generation...")

audio, sr = generate_full_audio(SCRIPT, VOICE,  breath_sec = 0.15,)

sf.write("output/final_output.wav", audio, sr)

print("✅ Done! File saved at output/final_output.wav")