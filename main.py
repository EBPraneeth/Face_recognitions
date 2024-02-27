from face_recognitions import logger
from face_recognitions.pipeline.stage_1_data_ingestion import DataingestionPipeline
from face_recognitions.pipeline.Stage_2_training_representation_model import Training_Representation_model_pipeline
STAGE_NAME="Data Ingestion"

try:
    logger.info(f">>>>Stage {STAGE_NAME} Started<<<<")
    obj=DataingestionPipeline()
    obj.main()
    logger.info(f">>>>Stage {STAGE_NAME} Completed<<<< \n\n x=============x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Training Representation Model"

try:

    logger.info(f">>>>Stage {STAGE_NAME} Started<<<<")
    obj=Training_Representation_model_pipeline()
    obj.main()
    logger.info(f">>>>Stage {STAGE_NAME} Completed<<<< \n\n x=============x")

except Exception as e:
    logger.exception(e)
    raise e
