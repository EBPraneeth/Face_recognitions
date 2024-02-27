from face_recognitions import logger
from face_recognitions.config.configuration import ConfigurationManager
from face_recognitions.components.training_representation_model import Train_Representation_Model

STAGE_NAME="Training Representation Model"

class Training_Representation_model_pipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config=ConfigurationManager()
        train_representation_model_config=config.get_train_representation_model_config()
        train_representation_model=Train_Representation_Model(config=train_representation_model_config)
        train_representation_model.representations()


if __name__=='__main__':
    try:
        logger.info(f">>>>Stage {STAGE_NAME} Started<<<<")
        obj=Training_Representation_model_pipeline()
        obj.main()
        logger.info(f">>>>Stage {STAGE_NAME} Completed<<<< \n\n x=============x")
    except Exception as e:
        logger.exception(e)
        raise e