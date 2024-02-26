from face_recognitions.config.configuration import ConfigurationManager
from face_recognitions.components.data_ingestion import DataIngestion
from face_recognitions import logger

STAGE_NAME="Data Ingestion"

class DataingestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config=ConfigurationManager()
        dat_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=dat_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_file()


if __name__=='__main__':
    try:
        logger.info(f">>>>Stage {STAGE_NAME} Started<<<<")
        obj=DataingestionPipeline()
        obj.main()
        logger.info(f">>>>Stage {STAGE_NAME} Completed<<<< \n\n x=============x")
    except Exception as e:
        logger.exception(e)
        raise e