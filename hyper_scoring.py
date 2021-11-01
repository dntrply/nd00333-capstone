
import json
import numpy as np
import azureml.train.automl
import joblib
import pandas as pd

from azureml.core.model import Model
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_hyper.pkl')

    model = joblib.load(model_path)

def run(raw_data): 
    # the parameters here have to match those in decorator, both 'Inputs' and 
    # 'GlobalParameters' here are case sensitive
    try:
        data = json.loads(raw_data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
