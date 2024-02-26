import os
import urllib.request as request
import zipfile
from face_recognitions import logger
from face_recognitions.utils.common import get_size
from face_recognitions.entity.config_entity import DataIngestionConfig
from face_recognitions.utils.common import create_directories
from pathlib import Path

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    def download_file(self):
        if not os.path.exists(self.config.zip_data_File):
            filename,headers=request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.zip_data_File
            )
            logger.info(f"{filename} download ! with flowwing info : \n {headers}")
        else:
            logger.info(f'File already exists of size: {get_size(Path(self.config.zip_data_File))}')


    def extract_file(self):
        '''
        Zip_file_path:str
        Extract the zip file into data directory
        Function returns None
        
        '''
        unzip_path=self.config.unzip_dir

        logger.info(f"Zip file extracting to {unzip_path}")
        
        # os.makedirs(unzip_path, exist_ok=True)
        create_directories([unzip_path])
        with zipfile.ZipFile(self.config.zip_data_File,'r')as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(f"Zip file extracted to {unzip_path}")