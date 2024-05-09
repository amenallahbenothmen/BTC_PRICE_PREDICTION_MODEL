from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL_Transaction:str
    source_URL_blocks:str
    dataset_name:str
    
@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir:Path
    data_dir:Path
    dataset_name:str
    data_scaled_dir:Path
    data_final_dir:Path
    features:list[str]
    look_back : int
    forecast_horizon:int
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir:Path
    full_model_path:Path
    shape_path:Path
    lstm_units_1:int 
    lstm_units_2 :int 
    dropout_rate_1:float
    dropout_rate_2:float
    l2_reg_1:float
    l2_reg_2:float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path 
    trained_model_path:Path
    full_model_path :Path
    training_data :Path
    data_dir :Path
    batch_size : int 
    epochs : int 
    patience : int 
    learning_rate: float 
    all_params : dict 
    mlflow_uri:str

