from LSTM_BTC_Prediction import logger 
from LSTM_BTC_Prediction.config.configuration import ConfigurationManager
from LSTM_BTC_Prediction.components.data_ingestion import DataIngestion


STAGE_NAME = "DATA Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self) :
        pass
    def main(self):
     config= ConfigurationManager()
     config=config.get_data_ingestion_config()
     data_ingestion=DataIngestion(config=config)
     data_ingestion.download_BTC()
     data_ingestion.adding_indicators()
     data_ingestion.download_Transaction()
     data_ingestion.download_Blocks()
     data_ingestion.download_INT_RATE()
     data_ingestion.download_STOCK_PRICE()
     data_ingestion.download_INFLATION()
     data_ingestion.save_dataset()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e