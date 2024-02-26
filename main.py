from face_recognitions import logger
from face_recognitions.pipeline.stage_1_data_ingestion import DataingestionPipeline

STAGE_NAME="Data Ingestion"

try:
    logger.info(f">>>>Stage {STAGE_NAME} Started<<<<")
    obj=DataingestionPipeline()
    obj.main()
    logger.info(f">>>>Stage {STAGE_NAME} Completed<<<< \n\n x=============x")
except Exception as e:
    logger.exception(e)
    raise e