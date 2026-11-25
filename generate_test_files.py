# generate_test_files.py
import wave
import os

output_dir = "test_files"
os.makedirs(output_dir, exist_ok=True)

for i in range(20):
    # Create WAV file (30 seconds)
    with wave.open(f"{output_dir}/voice{i}.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00" * 30 * 44100 * 2)
    # Create TXT file
    with open(f"{output_dir}/voice{i}.txt", "w") as txt:
        txt.write("Test transcript")