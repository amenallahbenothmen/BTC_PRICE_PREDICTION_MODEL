import pandas as pd 
import os 
from src.LSTM_BTC_Prediction import logger 
from sklearn.preprocessing import MinMaxScaler
from src.LSTM_BTC_Prediction.entity.config_entity import DataPreprocessingConfig
import numpy as np 
from typing import Tuple

class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def load_data(self):
        try:
            file_path = os.path.join(self.config.root_dir, f"{self.config.dataset_name}.csv")
            self.df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully for {self.config.dataset_name}")
        except Exception as e:
            logger.error(f"Error occurred during data loading for {self.config.dataset_name}: {e}")
            raise e

    def select_features(self):
        try:
            self.df = self.df[self.config.features]
            logger.info(f"Features selected successfully for {self.config.dataset_name}")
        except Exception as e:
            logger.error(f"Error occurred during feature selection for {self.config.dataset_name}: {e}")
            raise e

    def split_dataset(self):
        try:
            self.df_train = self.df[:int(len(self.df) * 0.6)]
            self.df_val = self.df[int(len(self.df) * 0.6):int(len(self.df) * 0.8)]
            self.df_test = self.df[int(len(self.df) * 0.8):]
            logger.info(f"Dataset split successfully for {self.config.dataset_name}")
        except Exception as e:
            logger.error(f"Error occurred during dataset splitting for {self.config.dataset_name}: {e}")
            raise e
    def transform_data(self):
        try:
            scaler_train = MinMaxScaler(feature_range=(0, 1))
            scaler_val = MinMaxScaler(feature_range=(0, 1))
            scaler_test = MinMaxScaler(feature_range=(0, 1))
            self.df_train_scaled = scaler_train.fit_transform(self.df_train)
            self.df_val_scaled = scaler_val.fit_transform(self.df_val)
            self.df_test_scaled = scaler_test.fit_transform(self.df_test)
            logger.info(f"Data transformed successfully for {self.config.dataset_name}")
        except Exception as e:
            logger.error(f"Error occurred during data transformation for {self.config.dataset_name}: {e}")
            raise e

    def dataset_generator_lstm(self, dataset: np.ndarray, look_back: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - forecast_horizon + 1):
            window_size_x = dataset[i:(i + look_back), :]
            window_size_y = dataset[i + look_back:i + look_back + forecast_horizon, 0]
            dataX.append(window_size_x)
            dataY.append(window_size_y)
        return np.array(dataX), np.array(dataY) 

    def transform_generator(self):
        try:
            self.trainX,self.trainY=self.dataset_generator_lstm(self.df_train_scaled, look_back=self.config.look_back, forecast_horizon=self.config.forecast_horizon)
            self.valX,self.valY=self.dataset_generator_lstm(self.df_val_scaled, look_back=self.config.look_back, forecast_horizon=self.config.forecast_horizon)
            self.testX,self.testY=self.dataset_generator_lstm(self.df_test_scaled, look_back=self.config.look_back, forecast_horizon=self.config.forecast_horizon)
            logger.info(f"Data transformation and generation completed successfully")
        except Exception as e:
            logger.error(f"Error occurred during data transformation and generation: {e}")
            raise e

    def save_final_dataset(self): 
        try:
            os.makedirs("artifacts/data_final", exist_ok=True) 
            np.save(os.path.join(self.config.data_final_dir, "trainX.npy"), self.trainX)
            np.save(os.path.join(self.config.data_final_dir, "trainY.npy"), self.trainY)
            np.save(os.path.join(self.config.data_final_dir, "valX.npy"), self.valX)
            np.save(os.path.join(self.config.data_final_dir, "valY.npy"), self.valY)
            np.save(os.path.join(self.config.data_final_dir, "testX.npy"), self.testX)
            np.save(os.path.join(self.config.data_final_dir, "testY.npy"), self.testY)
            logger.info(f"Final datasets saved successfully at: {self.config.data_final_dir}")
        except Exception as e:
            logger.error(f"Error occurred during saving final datasets: {e}")
            raise e  

    def save_dataset_splited(self):
        try:
            os.makedirs("artifacts/data_split", exist_ok=True)
            file_path_train = os.path.join(self.config.data_dir, f"train.csv")
            self.df_train.to_csv(file_path_train)
            file_path_val = os.path.join(self.config.data_dir, f"val.csv")
            self.df_val.to_csv(file_path_val)
            file_path_test = os.path.join(self.config.data_dir, f"test.csv")
            self.df_test.to_csv(file_path_test)
            logger.info(f"Splited dataset saved successfully for {self.config.dataset_name}")
        except Exception as e:
            logger.error(f"Error occurred during saving splited dataset for {self.config.dataset_name}: {e}")
            raise e

    def save_dataset_splited_transformed(self):
        try:
            os.makedirs("artifacts/data_scaled", exist_ok=True)
            file_path_train = os.path.join(self.config.data_scaled_dir, f"train_scaled.csv")
            pd.DataFrame(self.df_train_scaled).to_csv(file_path_train)
            file_path_val = os.path.join(self.config.data_scaled_dir, f"val_scaled.csv")
            pd.DataFrame(self.df_val_scaled).to_csv(file_path_val)
            file_path_test = os.path.join(self.config.data_scaled_dir, f"test_scaled.csv")
            pd.DataFrame(self.df_test_scaled).to_csv(file_path_test)
            logger.info(f"Splited and transformed dataset saved successfully for {self.config.dataset_name}")
        except Exception as e:
            logger.error(f"Error occurred during saving splited and transformed dataset for {self.config.dataset_name}: {e}")
            raise e
