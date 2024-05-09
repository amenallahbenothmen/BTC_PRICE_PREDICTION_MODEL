from src.LSTM_BTC_Prediction import logger
from src.LSTM_BTC_Prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.LSTM_BTC_Prediction.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from src.LSTM_BTC_Prediction.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.LSTM_BTC_Prediction.pipeline.stage_04_model_training import ModelTrainingPipeline

STAGE_NAME = "DATA Ingestion stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "DATA Preprocessing stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Prepare base model stage"
try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training"

try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = ModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e 