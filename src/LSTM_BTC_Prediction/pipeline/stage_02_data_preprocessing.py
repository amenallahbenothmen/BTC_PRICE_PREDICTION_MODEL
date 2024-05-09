from src.LSTM_BTC_Prediction import logger 
from src.LSTM_BTC_Prediction.config.configuration import ConfigurationManager
from src.LSTM_BTC_Prediction.components.data_preprocessing import DataPreprocessing


STAGE_NAME = "DATA Preprocessing stage"

class DataPreprocessingPipeline:

    def __init__(self):
        pass

    def main(self):
     config= ConfigurationManager()
     config=config.get_data_spliting_config()
     data_preprocessing = DataPreprocessing(config)
     data_preprocessing.load_data()
     data_preprocessing.select_features()
     data_preprocessing.split_dataset()
     data_preprocessing.transform_data()
     data_preprocessing.transform_generator()
     data_preprocessing.save_dataset_splited()
     data_preprocessing.save_dataset_splited_transformed()
     data_preprocessing.save_final_dataset()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


       

