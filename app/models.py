from pydantic import BaseModel
from typing import List

class Dataset(BaseModel):
    id: str
    audio_files: List[str]
    transcript_files: List[str]

class Model(BaseModel):
    version: str
    accuracy: float
    filename: str
    cm_path: str
    dataset_id: str