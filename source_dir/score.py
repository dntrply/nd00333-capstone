import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from sklearn.linear_model import Ridge

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

DEPLOYED_AUTOML_MODEL_PATH = 'outputs/best_automl.pkl'

def init():
    global model
    # Replace filename if needed.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), os.path.basename(CAPSTONE_DEPLOYED_MODEL_PATH))

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


