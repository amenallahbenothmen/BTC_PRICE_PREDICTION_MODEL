import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import L2
import numpy as np 
from pathlib import Path
from LSTM_BTC_Prediction import logger 
from LSTM_BTC_Prediction.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:

    def __init__(self, config:PrepareBaseModelConfig):
        self.config = config

    def get_shape(self):
        try:
            self.trainX = np.load(self.config.shape_path) 
            logger.info("Downloading trainX successful")
        except Exception as e:
            logger.error(f"Error occurred during downloading trainX: {e}")

    def get_base_model(self):
        try:
            self.get_shape()  # Call get_shape to ensure trainX is loaded before building the model

            self.model = Sequential()
            self.model.add(Bidirectional(LSTM(units=self.config.lstm_units_1, return_sequences=True, kernel_regularizer=L2(self.config.l2_reg_1)),
                                         input_shape=(self.trainX.shape[1], self.trainX.shape[2])))
            self.model.add(Dropout(self.config.dropout_rate_1))
            self.model.add(Bidirectional(LSTM(units=self.config.lstm_units_2, kernel_regularizer=L2(self.config.l2_reg_2))))
            self.model.add(Dropout(self.config.dropout_rate_2))
            self.model.add(Dense(units=1))
            self.model.summary()
            logger.info("Compiling model completed")
            self.save_model(model=self.model, path=self.config.full_model_path)
            logger.info('model saved')
        except Exception as e:
            logger.error(f"Error occurred during building the base model: {e}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
