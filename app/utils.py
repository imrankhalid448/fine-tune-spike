import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import seaborn as sns
import pandas as pd
import os
import soundfile as sf
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_audio(dataset_id, audio_files, transcript_files, only_eda=False, eda_type='spectrogram'):
    visualizations = []
    vis_dir = "app/static/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    if only_eda:
        durations = []
        mfcc_features = []

        for audio_file in audio_files:
            audio_path = f"app/static/uploads/{dataset_id}/{audio_file}"
            try:
                y, sr = librosa.load(audio_path, sr=22050)
                duration = librosa.get_duration(y=y, sr=sr)
                durations.append(duration)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_features.append(np.mean(mfcc, axis=1))
            except Exception as e:
                logger.error(f"Failed to process {audio_file} for EDA: {str(e)}")

        if eda_type == 'missing':
            for i, audio_file in enumerate(audio_files[:3]):
                audio_path = f"app/static/uploads/{dataset_id}/{audio_file}"
                try:
                    y, sr = librosa.load(audio_path, sr=22050)
                    plt.figure(figsize=(6, 4))
                    librosa.display.waveshow(y, sr=sr)
                    plt.title(f"Waveform (Audio {i+1})")
                    plot_path = os.path.join(vis_dir, f"waveform_{dataset_id}_{i}.png")
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()
                    visualizations.append(plot_path.replace("app/", ""))
                    logger.info(f"Saved waveform plot: {plot_path}")
                except Exception as e:
                    logger.error(f"Failed to generate waveform for {audio_file}: {str(e)}")
        elif eda_type == 'distribution':
            if durations:
                plt.figure(figsize=(6, 4))
                plt.hist(durations, bins=20, edgecolor='black')
                plt.title("Audio Duration Distribution")
                plt.xlabel("Duration (seconds)")
                plt.ylabel("Count")
                plot_path = os.path.join(vis_dir, f"duration_{dataset_id}.png")
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                visualizations.append(plot_path.replace("app/", ""))
                logger.info(f"Saved duration distribution plot: {plot_path}")
        elif eda_type == 'correlation':
            if mfcc_features:
                mfcc_df = pd.DataFrame(mfcc_features, columns=[f"MFCC_{i+1}" for i in range(13)])
                plt.figure(figsize=(6, 4))
                sns.heatmap(mfcc_df.corr(), annot=False, cmap='coolwarm')
                plt.title("MFCC Feature Correlation")
                plot_path = os.path.join(vis_dir, f"mfcc_corr_{dataset_id}.png")
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                visualizations.append(plot_path.replace("app/", ""))
                logger.info(f"Saved MFCC correlation plot: {plot_path}")
        else:  # spectrogram
            for i, audio_file in enumerate(audio_files[:3]):
                audio_path = f"app/static/uploads/{dataset_id}/{audio_file}"
                try:
                    y, sr = librosa.load(audio_path, sr=22050)
                    plt.figure(figsize=(6, 4))
                    S = librosa.feature.melspectrogram(y=y, sr=sr)
                    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f"Spectrogram (Audio {i+1})")
                    plot_path = os.path.join(vis_dir, f"spectrogram_{dataset_id}_{i}.png")
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()
                    visualizations.append(plot_path.replace("app/", ""))
                    logger.info(f"Saved spectrogram: {plot_path}")
                except Exception as e:
                    logger.error(f"Failed to generate spectrogram for {audio_file}: {str(e)}")
        return visualizations

    features = []
    labels = []
    for i, (audio_file, transcript_file) in enumerate(zip(audio_files, transcript_files)):
        audio_path = f"app/static/uploads/{dataset_id}/{audio_file}"
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.append(np.mean(mfcc, axis=1))

            transcript_path = f"app/static/uploads/{dataset_id}/{transcript_file}"
            with open(transcript_path, "r") as f:
                labels.append(f.read().strip())

            plt.figure(figsize=(6, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Spectrogram (Audio {i+1})")
            spectrogram_path = os.path.join(vis_dir, f"spectrogram_{dataset_id}_{i}.png")
            plt.savefig(spectrogram_path, bbox_inches='tight')
            plt.close()
            visualizations.append(spectrogram_path.replace("app/", ""))
            logger.info(f"Saved spectrogram: {spectrogram_path}")
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {str(e)}")

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    logger.info(f"Label distribution: {Counter(labels)}")
    return np.array(features), encoded_labels, visualizations

def apply_audio_feature_engineering(dataset_id, audio_files, type):
    features = []
    vis_dir = "app/static/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    for audio_file in audio_files:
        audio_path = f"app/static/uploads/{dataset_id}/{audio_file}"
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            if type == "zcr":
                zcr = librosa.feature.zero_crossing_rate(y)
                features.append(np.mean(zcr))
            elif type == "spectral_centroid":
                sc = librosa.feature.spectral_centroid(y=y, sr=sr)
                features.append(np.mean(sc))
            elif type == "mfcc_advanced":
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                features.append(np.mean(mfcc, axis=1))
        except Exception as e:
            logger.error(f"Failed to extract {type} for {audio_file}: {str(e)}")
    
    if features:
        feature_df = pd.DataFrame(features)
        plt.figure(figsize=(6, 4))
        sns.histplot(feature_df.iloc[:, 0], kde=True)
        plt.title(f"{type.upper()} Distribution")
        plot_path = os.path.join(vis_dir, f"{type}_{dataset_id}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {type} distribution plot: {plot_path}")
        return feature_df, [plot_path.replace("app/", "")]
    return pd.DataFrame(), []

def perform_eda(df, type, dataset_id):
    vis_dir = "app/static/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    visualizations = []

    if type == "missing":
        missing = df.isnull().sum()
        plt.figure(figsize=(8, 4))
        missing.plot(kind='bar')
        plt.title("Missing Values")
        path = os.path.join(vis_dir, f"missing_{dataset_id}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        visualizations.append(path.replace("app/", ""))
        logger.info(f"Saved missing values plot: {path}")
    elif type == "distribution":
        for col in df.select_dtypes(include=['int64', 'float64']).columns[:3]:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            path = os.path.join(vis_dir, f"dist_{dataset_id}_{col}.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            visualizations.append(path.replace("app/", ""))
            logger.info(f"Saved distribution plot: {path}")
    elif type == "correlation":
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        path = os.path.join(vis_dir, f"corr_{dataset_id}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        visualizations.append(path.replace("app/", ""))
        logger.info(f"Saved correlation plot: {path}")
    return visualizations

def apply_feature_engineering(df, type):
    if type == "normalize":
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif type == "encode":
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]
    elif type == "pca":
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        pca = PCA(n_components=2)
        df[['pca1', 'pca2']] = pca.fit_transform(df[numeric_cols])
    return df

def train_model(data, labels=None, dataset_id=None, model_type="rf"):
    vis_dir = "app/static/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        is_regression = y.dtype in ['float64', 'float32'] and len(y.unique()) > 10
    else:
        X = data
        y = labels
        is_regression = False

    n_samples = len(X)
    logger.info(f"Training with {n_samples} samples, unique labels: {len(np.unique(y))}")
    
    if n_samples < 2:
        logger.error("Insufficient samples for training")
        return None, 0.0, ""
    
    test_size = 0.2 if n_samples >= 10 else 0.3 if n_samples >= 5 else 0.5
    if len(np.unique(y)) == 1:
        logger.warning("Single label detected, accuracy will be 0 or 1")
        test_size = 0.5  # Force a split to evaluate

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None if is_regression else y)

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=2)
    elif model_type == "rf_reg":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LinearRegression()
    elif model_type == "svm":
        model = SVC(random_state=42)
    elif model_type == "cnn":
        raise NotImplementedError("CNN not implemented")

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if is_regression or model_type in ["lr", "rf_reg"]:
            accuracy = r2_score(y_test, y_pred)
            cm_path = ""
        else:
            accuracy = accuracy_score(y_test, y_pred)
            unique_labels = np.unique(y)
            if len(unique_labels) > 1:
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
                plt.title("Confusion Matrix")
                cm_path = os.path.join(vis_dir, f"cm_{dataset_id}.png")
                plt.savefig(cm_path, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved confusion matrix: {cm_path}")
            else:
                cm_path = ""
                logger.warning("Single label, skipping confusion matrix")
        
        logger.info(f"Training completed, accuracy: {accuracy}")
        return model, accuracy, cm_path.replace("app/", "") if cm_path else ""
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None, 0.0, ""