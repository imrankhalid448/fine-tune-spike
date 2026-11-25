import os
from pydub import AudioSegment
import shutil

def preprocess_dataset(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the .trans file
    trans_file = os.path.join(input_dir, "H:/My Drive/Running Projects/fine_tune_spike/dev-clean/LibriSpeech/dev-clean/1673/143397/1673-143397.trans")
    transcripts = {}
    with open(trans_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                file_id, transcript = parts
                transcripts[file_id] = transcript

    # Convert .flac to .wav and create .txt files
    for filename in os.listdir(input_dir):
        if filename.endswith(".flac"):
            file_id = filename.replace(".flac", "")
            flac_path = os.path.join(input_dir, filename)
            
            # Convert flac to wav
            audio = AudioSegment.from_file(flac_path, format="flac")
            wav_filename = f"{file_id}.wav"
            wav_path = os.path.join(output_dir, wav_filename)
            audio.export(wav_path, format="wav")
            print(f"Converted {filename} to {wav_filename}")

            # Create corresponding .txt file
            if file_id in transcripts:
                txt_filename = f"{file_id}.txt"
                txt_path = os.path.join(output_dir, txt_filename)
                with open(txt_path, "w") as f:
                    f.write(transcripts[file_id])
                print(f"Created {txt_filename}")

if __name__ == "__main__":
    input_dir = "H:/My Drive/Running Projects/fine_tune_spike/dev-clean/LibriSpeech/dev-clean/1673/143397"
    output_dir = "processed_dataset"
    preprocess_dataset(input_dir, output_dir)