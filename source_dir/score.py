import json
import numpy as np
import azureml.train.automl
import joblib

from azureml.core.model import Model
import logging

DEPLOYED_AUTOML_MODEL_PATH = 'outputs/best_automl.pkl'

def init():
    global model
    logging.basicConfig(level=logging.DEBUG)
    print(Model.get_model_path(model_name='best_automl.pkl'))
    # Replace filename if needed.
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), os.path.basename(CAPSTONE_DEPLOYED_MODEL_PATH))
    model_path = Model.get_model_path(model_name='best_automl.pkl')

    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

def run(data): 
    # the parameters here have to match those in decorator, both 'Inputs' and 
    # 'GlobalParameters' here are case sensitive
    try:
        data = json.loads(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error


