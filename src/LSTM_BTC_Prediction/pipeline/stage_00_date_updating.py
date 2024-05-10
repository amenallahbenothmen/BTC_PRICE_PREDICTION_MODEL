from LSTM_BTC_Prediction import logger 
from LSTM_BTC_Prediction.utils.common import update_current_date
from LSTM_BTC_Prediction.constants import *

STAGE_NAME = "Updating Date"

class UpdatingDatePipeline:

    def __init__(self, params_filepath=PARAMS_FILE_PATH):
        self.params_filepath = params_filepath

    def main(self):
        logger.info(f"Starting {STAGE_NAME} stage...")
        try:
            update_current_date(self.params_filepath)
            logger.info(f"{STAGE_NAME} stage completed successfully.")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME} stage: {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = UpdatingDatePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


        
