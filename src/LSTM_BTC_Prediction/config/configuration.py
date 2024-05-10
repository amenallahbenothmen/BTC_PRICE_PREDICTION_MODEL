from LSTM_BTC_Prediction.constants  import *
from LSTM_BTC_Prediction.utils.common import read_yaml,create_directories
from LSTM_BTC_Prediction.entity.config_entity import DataIngestionConfig,DataPreprocessingConfig,PrepareBaseModelConfig,TrainingConfig

class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath=PARAMS_FILE_PATH):

            self.config=read_yaml(config_filepath) 
            self.params=read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config=self.config.data_ingestion

        create_directories([config.root_dir])  

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL_Transaction=config.source_URL_Transaction,
            source_URL_blocks=config.source_URL_blocks,
            dataset_name= config.dataset_name,
            current_date=self.params.CURRENT_DATE

        )
        return data_ingestion_config

    
    def get_data_spliting_config(self) -> DataPreprocessingConfig:
        
        config=self.config.data_preprocessing

        create_directories([config.root_dir])  

        data_preprocessing_config=DataPreprocessingConfig(
            root_dir=config.root_dir,
            dataset_name= config.dataset_name,
            features=config.features,
            data_dir=config.data_dir,
            data_scaled_dir=config.data_scaled_dir,
            data_final_dir=config.data_final_dir,
            look_back=self.params.LOOK_BACK,
            forecast_horizon=self.params.FORECAST_HORIZON
            )

        return data_preprocessing_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config=self.config.prepare_base_model

        create_directories([config.root_dir])  

        prepare_base_model_config=PrepareBaseModelConfig(
            root_dir= config.root_dir,
            full_model_path = config.full_model_path,
            shape_path = config.shape_path,
            lstm_units_1= self.params.LSTM_UNITS_1,
            lstm_units_2 = self.params.LSTM_UNITS_2,
            dropout_rate_1= self.params.DROPOUT_RATE_1,
            dropout_rate_2 = self.params.DROPOUT_RATE_2 ,
            l2_reg_1=self.params.L2_REG_1,
            l2_reg_2=self.params.L2_REG_2  )
        
        return prepare_base_model_config

    def get_tarining_config(self) ->TrainingConfig:
        training=self.config.training
        prepare_base_model=self.config.prepare_base_model
        params=self.params
        training_data=self.config.data_preprocessing.data_final_dir

        create_directories([training.root_dir])
        training_config=TrainingConfig(
            root_dir=training.root_dir,
            trained_model_path=training.trained_model_path,
            full_model_path=prepare_base_model.full_model_path,
            training_data=training_data,
            data_dir=self.config.data_preprocessing.data_dir,
            batch_size=params.BATCH_SIZE,
            epochs=params.EPOCHS,
            patience=params.PATIENCE,
            learning_rate=params.LEARNING_RATE,
            all_params=params,
            mlflow_uri="https://dagshub.com/amenallahbenothmen/BTC_PRICE_PREDICTION.mlflow"
        )
        return training_config   
