import tensorflow as tf
from LSTM_BTC_Prediction import logger  
import numpy as np 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os 
from LSTM_BTC_Prediction.entity.config_entity import TrainingConfig
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import pandas as pd 
from LSTM_BTC_Prediction.utils.common import save_json
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config 

    def get_base_model(self):
        try:
            self.model = tf.keras.models.load_model(self.config.full_model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f'Error loading model: {e}')

    def get_data(self):
        try:
            self.trainX = np.load(os.path.join(self.config.training_data, "trainX.npy"))
            self.trainY = np.load(os.path.join(self.config.training_data, "trainY.npy"))
            self.valX = np.load(os.path.join(self.config.training_data, "valX.npy"))
            self.valY = np.load(os.path.join(self.config.training_data, "valY.npy"))
            self.testX = np.load(os.path.join(self.config.training_data, "testX.npy"))
            self.testY = np.load(os.path.join(self.config.training_data, "testY.npy"))
            self.df_test = pd.read_csv(os.path.join(self.config.data_dir, "test.csv"))
            self.df_train = pd.read_csv(os.path.join(self.config.data_dir, "train.csv"))
            self.df_val = pd.read_csv(os.path.join(self.config.data_dir, "val.csv"))
            logger.info("DATA loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}") 

    def calculate_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def save_score(self, train_loss, val_loss, test_rmse):
        self.scores = {"train_loss": train_loss, "val_loss": val_loss, "test_rmse": test_rmse}
        save_json(path=Path("scores.json"), data=self.scores)       

    def train(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mean_squared_error'
        ) 
        checkpoint_path = self.config.trained_model_path
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path, 
            monitor='val_loss',
            verbose=1, 
            save_best_only=True,
            mode='min'
        )
        earlystopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.config.patience, 
            restore_best_weights=True
        )
        callbacks = [checkpoint, earlystopping]

        history = self.model.fit(
            self.trainX, 
            self.trainY, 
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            verbose=1, 
            shuffle=False, 
            validation_data=(self.valX, self.valY),
            callbacks=callbacks
        )

        max_test = self.df_test['close'].max()
        min_test = self.df_test['close'].min()
        max_val = self.df_val['close'].max()
        min_val = self.df_val['close'].min()
        max_train = self.df_train['close'].max()
        min_train = self.df_train['close'].min()                

        train_loss = history.history['loss'][-1] * (max_train - min_train) + min_train
        val_loss = history.history['val_loss'][-1] * (max_val - min_val) + min_val

        test_predictions = self.model.predict(self.testX)
        test_predictions = test_predictions * (max_test - min_test) + min_test

        prediction=test_predictions.reshape(-1,1).flatten()[-self.config.forecast_horizon:]

        self.save_prediction(prediction=prediction)
        self.save_model_to_dir()

        actual_price = self.testY * (max_test - min_test) + min_test

        test_rmse = self.calculate_rmse(actual_price, test_predictions)

        self.save_score(train_loss, val_loss, test_rmse)

    def save_model_to_dir(self):
        model=tf.keras.models.load_model(self.config.trained_model_path)
        model.save(self.config.saved_model_path)
 
    def save_prediction(self,prediction:np.array):
         np.save(self.config.result_path,prediction)


    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"train_loss": self.scores["train_loss"], "val_loss": self.scores["val_loss"], "test_rmse": self.scores["test_rmse"]}
            )
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="LSTM_BTC_PREDECTION")
            else:
                mlflow.keras.log_model(self.model, "model")

