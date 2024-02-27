from face_recognitions.constants import *
from face_recognitions.utils.common import read_yaml,create_directories
from face_recognitions.entity.config_entity import DataIngestionConfig,Train_Representation_Model_Config

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_YAML_PATH,
                 params_filepath=PARAMS_YAML_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config= DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            zip_data_File=config.zip_data_File,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_train_representation_model_config(self)->Train_Representation_Model_Config:
        config=self.config.train_model
        params=self.params
        print(config)
        create_directories([config.model_root_dir])

        train_representation_model_config=Train_Representation_Model_Config(
            model_root_dir=config.model_root_dir,
            represent_file= config.represent_file,
            train_faces_dir=config.train_faces_dir,
            model_name= params.MODEL_NAME,
            face_detector_backend=params.FACE_DETECTOR_BACKEND,
            target_size=params.TARGET_SIZE, 
            align=params.ALIGN,
            enforce_detection=params.ENFORCE_DETECTION,
            expand_percentage=params.EXPAND_PERCENTAGE,
            normalization=params.NORMALIZATION,
            silent=params.SILENT
        )

        return train_representation_model_config