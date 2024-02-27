from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url:str
    zip_data_File:Path
    unzip_dir:Path

@dataclass(frozen=True)
class Train_Representation_Model_Config:
    model_root_dir: Path
    represent_file:Path
    train_faces_dir:Path
    model_name: str
    face_detector_backend: str
    target_size: tuple
    align: bool
    enforce_detection: bool
    expand_percentage: int
    normalization:str
    silent: bool