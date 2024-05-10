from tensorflow.keras.models import load_model
import os 

class PredictionPipeline:

    def __init__(self):
        pass
    def predict(self):
        model=load_model(os.path.join("model","model.keras"))
        

