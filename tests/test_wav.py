# test_wav.py
from io import BytesIO
import wave
from pydub import AudioSegment

audio = BytesIO()
with wave.open(audio, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(44100)
    wav.writeframes(b"\x00" * 300 * 44100)  # 300 seconds
audio.seek(0)
audio_segment = AudioSegment.from_wav(audio)
duration = len(audio_segment) / 1000
print(f"Duration: {duration} seconds")