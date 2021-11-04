
import json
import numpy as np
import joblib
import pandas as pd
import os

from azureml.core.model import Model
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_hyper.pkl')
    model = joblib.load(model_path)

def run(data): 
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
