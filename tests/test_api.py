import pytest
import os
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
import zipfile
from io import BytesIO
import soundfile as sf

client = TestClient(app)

def create_test_audio_transcript(index, duration=1, label="positive", count=1):
    files = []
    for i in range(index, index + count):
        basename = f"test_{i}"
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(44100 * duration)))
        audio_path = f"app/static/uploads/{basename}.wav"
        sf.write(audio_path, audio, 44100)
        files.append(("files", (f"txt and Audios/{basename}.wav", open(audio_path, "rb"), "audio/wav")))
        files.append(("files", (f"txt and Audios/{basename}.txt", label.encode(), "text/plain")))
    return files

def create_test_zip(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for _, (filename, file_content, _) in files:
            if isinstance(file_content, bytes):
                zip_file.writestr(filename, file_content)
            else:
                zip_file.writestr(filename, file_content.read())
    zip_buffer.seek(0)
    return [("files", ("test.zip", zip_buffer, "application/zip"))]

def create_test_csv():
    import pandas as pd
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0]
    })
    csv_path = "app/static/uploads/test.csv"
    df.to_csv(csv_path, index=False)
    return [("files", ("test.csv", open(csv_path, "rb"), "text/csv"))]

@pytest.fixture
def cleanup():
    os.environ["TESTING"] = "true"
    yield
    for file in os.listdir("app/static/uploads"):
        if file.startswith("test"):
            path = os.path.join("app/static/uploads", file)
            if os.path.isfile(path):
                os.remove(path)
    os.environ["TESTING"] = "false"

def test_success_path_zip(cleanup):
    os.environ["TESTING"] = "true"
    try:
        # Create 5 audio files with varied labels (3 positive, 2 negative)
        labels = ["positive", "positive", "positive", "negative", "negative"]
        files = []
        for i, label in enumerate(labels):
            files.extend(create_test_audio_transcript(i, duration=1, label=label, count=1))
        zip_files = create_test_zip(files)
        response = client.post("/upload", files=zip_files)
        assert response.status_code == 200
        dataset_id = response.json()["id"]

        response = client.post(f"/preprocess/{dataset_id}")
        assert response.status_code == 200
        assert "features" in response.json()

        response = client.post(f"/eda/{dataset_id}?type=spectrogram")
        assert response.status_code == 200
        assert "visualizations" in response.json()

        response = client.post(f"/train/{dataset_id}?model=rf")
        assert response.status_code == 200
        assert "accuracy" in response.json()
        assert "cm_path" in response.json()
    finally:
        os.environ["TESTING"] = "false"

def test_single_label_failure(cleanup):
    os.environ["TESTING"] = "true"
    try:
        # Create 5 audio files with single label
        files = create_test_audio_transcript(1, duration=1, label="positive", count=5)
        zip_files = create_test_zip(files)
        response = client.post("/upload", files=zip_files)
        assert response.status_code == 200
        dataset_id = response.json()["id"]

        response = client.post(f"/preprocess/{dataset_id}")
        assert response.status_code == 200
        assert "features" in response.json()

        response = client.post(f"/train/{dataset_id}?model=rf")
        assert response.status_code == 400
        assert "Single label" in response.json()["detail"]
    finally:
        os.environ["TESTING"] = "false"

def test_success_path_csv(cleanup):
    os.environ["TESTING"] = "true"
    try:
        files = create_test_csv()
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        dataset_id = response.json()["id"]

        response = client.post(f"/eda/{dataset_id}?type=distribution")
        assert response.status_code == 200
        assert "visualizations" in response.json()

        response = client.post(f"/feature/{dataset_id}?type=normalize")
        assert response.status_code == 200
        assert "features" in response.json()

        response = client.post(f"/train/{dataset_id}?model=rf")
        assert response.status_code == 200
        assert "accuracy" in response.json()
    finally:
        os.environ["TESTING"] = "false"

def test_validation_failure_invalid_file(cleanup):
    os.environ["TESTING"] = "true"
    try:
        files = [("files", ("test.txt", b"invalid", "text/plain"))]
        response = client.post("/upload", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    finally:
        os.environ["TESTING"] = "false"