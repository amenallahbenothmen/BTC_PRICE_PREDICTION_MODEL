from LSTM_BTC_Prediction.config.configuration import ConfigurationManager
from LSTM_BTC_Prediction.components.model_training import Training
from LSTM_BTC_Prediction import logger 

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
     config=ConfigurationManager()
     training_config=config.get_tarining_config()
     training=Training(config=training_config)
     training.get_base_model()
     training.get_data()
     training.train()
     training.log_into_mlflow()

 

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = ModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 
