import sqlite3
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_db():
    db_path = "app/audio_kaggle_clone.db"
    try:
        # Remove existing database during tests to avoid schema conflicts
        if os.environ.get("TESTING") == "true" and os.path.exists(db_path):
            os.remove(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS datasets
                     (id TEXT PRIMARY KEY, type TEXT, files TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS models
                     (version TEXT PRIMARY KEY, accuracy REAL, filename TEXT, cm_path TEXT, dataset_id TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS versions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT, comment TEXT, timestamp TEXT)''')
        conn.commit()
        logger.info(f"Database initialized at {db_path}")
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        raise
    finally:
        conn.close()

def save_dataset(dataset_id, files, dataset_type):
    try:
        conn = sqlite3.connect("app/audio_kaggle_clone.db")
        c = conn.cursor()
        c.execute("INSERT INTO datasets (id, type, files) VALUES (?, ?, ?)",
                  (dataset_id, dataset_type, repr(files)))
        conn.commit()
        logger.debug(f"Dataset {dataset_id} saved")
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        raise
    finally:
        conn.close()

def save_model(version, accuracy, filename, cm_path, dataset_id):
    try:
        conn = sqlite3.connect("app/audio_kaggle_clone.db")
        c = conn.cursor()
        c.execute("INSERT INTO models (version, accuracy, filename, cm_path, dataset_id) VALUES (?, ?, ?, ?, ?)",
                  (version, accuracy, filename, cm_path, dataset_id))
        conn.commit()
        logger.debug(f"Model {version} saved")
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        raise
    finally:
        conn.close()

def save_version(dataset_id, comment):
    try:
        conn = sqlite3.connect("app/audio_kaggle_clone.db")
        c = conn.cursor()
        c.execute("INSERT INTO versions (dataset_id, comment, timestamp) VALUES (?, ?, datetime('now'))",
                  (dataset_id, comment))
        conn.commit()
        logger.debug(f"Version saved for dataset {dataset_id}")
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        raise
    finally:
        conn.close()

def get_datasets():
    try:
        conn = sqlite3.connect("app/audio_kaggle_clone.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM datasets")
        return [{"id": r["id"], "type": r["type"], "files": r["files"]} for r in c.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        raise
    finally:
        conn.close()

def get_models():
    try:
        conn = sqlite3.connect("app/audio_kaggle_clone.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM models")
        return [{"version": r["version"], "accuracy": r["accuracy"], "filename": r["filename"],
                 "cm_path": r["cm_path"], "dataset_id": r["dataset_id"]} for r in c.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        raise
    finally:
        conn.close()