from src.LSTM_BTC_Prediction.config.configuration import ConfigurationManager
from src.LSTM_BTC_Prediction.components.prepare_base_model import PrepareBaseModel
from src.LSTM_BTC_Prediction import logger 

STAGE_NAME = "Prepare base model stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self) :
        pass
    def main(self):
     config = ConfigurationManager()
     prepare_base_model_config = config.get_prepare_base_model_config()
     prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
     prepare_base_model.get_shape()
     prepare_base_model.get_base_model() 



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e