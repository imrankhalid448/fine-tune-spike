from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import os
import shutil
import sqlite3
import uuid
import pickle
import zipfile
import io
import pandas as pd
import logging
from pydantic import BaseModel
from app.utils import preprocess_audio, train_model, perform_eda, apply_feature_engineering
from app.database import init_db, save_dataset, save_model, get_datasets, get_models, save_version
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/static")

try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise RuntimeError(f"Database initialization failed: {str(e)}")

class DatasetResponse(BaseModel):
    id: str
    type: str
    files: List[str]

class ModelResponse(BaseModel):
    version: str
    accuracy: float
    filename: str
    cm_path: str
    dataset_id: str

class EDAResponse(BaseModel):
    visualizations: List[str]

class FeatureResponse(BaseModel):
    features: str

class ModelSuggestion(BaseModel):
    id: str
    name: str
    description: str

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith(("/upload", "/preprocess", "/eda", "/feature", "/train", "/models", "/datasets", "/save_version")):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": exc.status_code,
            "message": exc.detail
        },
        status_code=exc.status_code
    )

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load index page: {str(e)}")

@app.post("/upload", response_model=DatasetResponse)
async def upload_dataset(files: List[UploadFile] = File(...)):
    dataset_id = str(uuid.uuid4())
    upload_dir = f"app/static/uploads/{dataset_id}"
    audio_dir = os.path.join(upload_dir, "txt and Audios")
    os.makedirs(audio_dir, exist_ok=True)
    files_list = []
    dataset_type = None
    audio_files = []
    transcript_files = []

    try:
        # Step 1: Collect file types
        filenames = [file.filename.lower() for file in files]
        has_zip = any(f.endswith(".zip") for f in filenames)
        has_csv = any(f.endswith(".csv") for f in filenames)
        has_wav = any(f.endswith(".wav") for f in filenames)
        has_txt = any(f.endswith(".txt") for f in filenames)

        # Step 2: Validate dataset type
        valid_extensions = {".zip", ".csv", ".wav"}
        if has_txt and not (has_wav or has_zip):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filenames[0]}")
        
        if has_zip:
            if len(filenames) > 1 or has_csv or has_wav or has_txt:
                raise HTTPException(status_code=400, detail="ZIP files must be uploaded alone")
            dataset_type = "audio"
        elif has_csv:
            if has_wav or has_txt or has_zip:
                raise HTTPException(status_code=400, detail="CSV files cannot be mixed with other file types")
            dataset_type = "tabular"
        elif has_wav and has_txt:
            if has_csv or has_zip:
                raise HTTPException(status_code=400, detail="Audio files cannot be mixed with CSV or ZIP")
            dataset_type = "audio"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filenames[0]}")

        # Step 3: Process files
        for file in files:
            filename_lower = file.filename.lower()
            if filename_lower.endswith(".zip"):
                zip_path = os.path.join(upload_dir, file.filename)
                with open(zip_path, "wb") as f:
                    f.write(await file.read())
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(audio_dir)
                    for f in zip_ref.namelist():
                        if f.lower().endswith(".wav"):
                            audio_files.append(os.path.join("txt and Audios", f).replace('\\', '/'))
                        elif f.lower().endswith(".txt"):
                            transcript_files.append(os.path.join("txt and Audios", f).replace('\\', '/'))
                os.unlink(zip_path)
            elif filename_lower.endswith(".csv"):
                save_path = os.path.join(upload_dir, file.filename)
                with open(save_path, "wb") as f:
                    f.write(await file.read())
                files_list.append(file.filename)
            elif filename_lower.endswith(".wav"):
                normalized_filename = file.filename.replace('\\', '/').split('/')[-1]
                audio_files.append(f"txt and Audios/{normalized_filename}")
                save_path = os.path.join(audio_dir, normalized_filename)
                with open(save_path, "wb") as f:
                    f.write(await file.read())
            elif filename_lower.endswith(".txt"):
                normalized_filename = file.filename.replace('\\', '/').split('/')[-1]
                transcript_files.append(f"txt and Audios/{normalized_filename}")
                save_path = os.path.join(audio_dir, normalized_filename)
                with open(save_path, "wb") as f:
                    f.write(await file.read())

        # Step 4: Validate dataset
        if dataset_type == "audio":
            if not audio_files:
                raise HTTPException(status_code=400, detail="No .wav files found in upload")
            if not transcript_files:
                raise HTTPException(status_code=400, detail="No .txt files found in upload")
            audio_basenames = {os.path.splitext(os.path.basename(f.lower()))[0] for f in audio_files}
            transcript_basenames = {os.path.splitext(os.path.basename(f.lower()))[0] for f in transcript_files}
            if not audio_basenames.intersection(transcript_basenames):
                raise HTTPException(status_code=400, detail="No matching audio and transcript filenames found")
            files_list = audio_files + transcript_files
        elif dataset_type == "tabular":
            if not files_list:
                raise HTTPException(status_code=400, detail="No valid CSV files found in upload")

        save_dataset(dataset_id, files_list, dataset_type)
        return DatasetResponse(id=dataset_id, type=dataset_type, files=files_list)
    except HTTPException as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise e
    except Exception as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/preprocess/{dataset_id}", response_model=FeatureResponse)
async def preprocess_dataset(dataset_id: str):
    try:
        datasets = get_datasets()
        dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if dataset["type"] == "audio":
            audio_files = [f for f in eval(dataset["files"]) if f.endswith(".wav")]
            transcript_files = [f for f in eval(dataset["files"]) if f.endswith(".txt")]
            features, labels, _ = preprocess_audio(dataset_id, audio_files, transcript_files)
            feature_path = f"app/static/uploads/processed_{dataset_id}.pkl"
            with open(feature_path, "wb") as f:
                pickle.dump({"features": features, "labels": labels}, f)
            return FeatureResponse(features=f"Processed {len(features)} audio samples")
        else:
            raise HTTPException(status_code=400, detail="Preprocessing only supported for audio datasets")
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets():
    try:
        datasets = get_datasets()
        return [DatasetResponse(id=d["id"], type=d["type"], files=eval(d["files"])) for d in datasets]
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@app.post("/eda/{dataset_id}", response_model=EDAResponse)
async def perform_eda_endpoint(dataset_id: str, type: str):
    try:
        datasets = get_datasets()
        dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        visualizations = []
        if dataset["type"] == "tabular":
            csv_path = f"app/static/uploads/{dataset_id}/{eval(dataset['files'])[0]}"
            df = pd.read_csv(csv_path)
            visualizations = perform_eda(df, type, dataset_id)
            visualizations = [f"/static/visualizations/{os.path.basename(v)}" for v in visualizations]
        elif dataset["type"] == "audio":
            audio_files = [f for f in eval(dataset["files"]) if f.endswith(".wav")]
            visualizations = preprocess_audio(dataset_id, audio_files, [], only_eda=True, eda_type=type)
            visualizations = [f"/static/visualizations/{os.path.basename(v)}" for v in visualizations]
        return EDAResponse(visualizations=visualizations)
    except Exception as e:
        logger.error(f"EDA error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")

@app.post("/feature/{dataset_id}", response_model=FeatureResponse)
async def feature_engineering(dataset_id: str, type: str):
    datasets = get_datasets()
    dataset = next((d for d in datasets if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset["type"] == "audio":
        raise HTTPException(status_code=400, detail="Feature engineering not supported for audio datasets")

    try:
        csv_path = f"app/static/uploads/{dataset_id}/{eval(dataset['files'])[0]}"
        df = pd.read_csv(csv_path)
        processed_df = apply_feature_engineering(df, type)
        processed_path = f"app/static/uploads/processed_{dataset_id}.csv"
        processed_df.to_csv(processed_path, index=False)
        return FeatureResponse(features=f"Processed {processed_df.shape[1]} features")
    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

@app.get("/models/suggestions/{dataset_id}", response_model=List[ModelSuggestion])
async def get_model_suggestions(dataset_id: str):
    try:
        datasets = get_datasets()
        dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        suggestions = []
        if dataset["type"] == "tabular":
            csv_path = f"app/static/uploads/{dataset_id}/{eval(dataset['files'])[0]}"
            df = pd.read_csv(csv_path)
            target_col = df.columns[-1]
            if df[target_col].dtype in ['int64', 'float64'] and len(df[target_col].unique()) > 10:
                suggestions = [
                    {"id": "lr", "name": "Linear Regression", "description": "Suitable for regression tasks"},
                    {"id": "rf_reg", "name": "Random Forest Regressor", "description": "Robust for regression"},
                ]
            else:
                suggestions = [
                    {"id": "rf", "name": "Random Forest Classifier", "description": "Robust for classification"},
                    {"id": "svm", "name": "SVM", "description": "Effective for high-dimensional data"},
                ]
        elif dataset["type"] == "audio":
            suggestions = [
                {"id": "rf", "name": "Random Forest Classifier", "description": "Baseline for audio classification"},
                {"id": "cnn", "name": "CNN", "description": "Advanced for audio spectrograms"},
            ]
        return suggestions
    except Exception as e:
        logger.error(f"Model suggestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model suggestion failed: {str(e)}")

@app.post("/train/{dataset_id}", response_model=ModelResponse)
async def train(dataset_id: str, model: str):
    try:
        datasets = get_datasets()
        dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if dataset["type"] == "tabular":
            processed_path = f"app/static/uploads/processed_{dataset_id}.csv"
            if not os.path.exists(processed_path):
                raise HTTPException(status_code=400, detail="Dataset not processed")
            df = pd.read_csv(processed_path)
            model_obj, accuracy, cm_path = train_model(df, model, dataset_id)
        elif dataset["type"] == "audio":
            feature_path = f"app/static/uploads/processed_{dataset_id}.pkl"
            if not os.path.exists(feature_path):
                raise HTTPException(status_code=400, detail="Dataset not processed")
            with open(feature_path, "rb") as f:
                data = pickle.load(f)
            model_obj, accuracy, cm_path = train_model(data["features"], data["labels"], dataset_id, model_type=model)

        version = f"v{len(get_models()) + 1}"
        model_path = f"app/static/models/model_{version}.pkl"
        os.makedirs("app/static/models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)

        save_model(version, accuracy, model_path, cm_path, dataset_id)
        return ModelResponse(version=version, accuracy=accuracy, filename=model_path, cm_path=cm_path, dataset_id=dataset_id)
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/save_version/{dataset_id}")
async def save_version_endpoint(dataset_id: str, comment: dict):
    try:
        save_version(dataset_id, comment.get("comment", ""))
        return {"status": "Version saved"}
    except Exception as e:
        logger.error(f"Save version error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Version save failed: {str(e)}")

@app.get("/models", response_model=List[ModelResponse])
async def list_models():
    try:
        models = get_models()
        return [ModelResponse(**m) for m in models]
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/visualizations/{dataset_id}/{plot_type}")
async def get_visualization(dataset_id: str, plot_type: str):
    try:
        plot_path = f"app/static/visualizations/{plot_type}_{dataset_id}.png"
        if os.path.exists(plot_path):
            return FileResponse(plot_path)
        raise HTTPException(status_code=404, detail="Visualization not found")
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")