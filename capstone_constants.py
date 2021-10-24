"""Constants used by the Capstone project

THese are all the consatnats used by the Capstone project.
"""

# Name of the experiment
AUTOML_EXPERIMENT_NAME = 'exp-capstone-automl'
HYPER_EXPERIMENT_NAME = 'exp-capstone-hyper'

#name of the compute cluster
COMPUTE_CLUSTER_AUTOML = "CPU-CC-AUTOML"  # CPU Compute Cluster for AUTOML
COMPUTE_CLUSTER_HYPER = "CPU-CC-HYPER"

# Tabular wine data URI - external to Mirosoft Azure ML ecosystem
TABULAR_WINE_DATA_URI = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Wine Dataset name in AzureML
DATASET_NAME = 'White Wine Data'
DATASET_DESCRIPTION = 'White Wind data set obtained form the UCI datasets'


# Training related constants
TRAIN_DATA_DIR = 'train_normalized_data'
TRAIN_DATA_FILE = 'train_normalized.csv'
TRAIN_NORMALIZATION_PARAMETERS_FILE = 'normaliztion_parameeters.csv'

# curated environment to be used for HyperML
CURATED_ENV_NAME = 'AzureML-Tutorial'

# deployed hyper model constants
DEPLOYED_HYPER_MODEL_NAME = 'wine-taste-hyper'
DEPLOYED_HYPER_MODEL_PATH = 'outputs/best_hyperdrive.pkl' 

# deployed automl model constants
DEPLOYED_AUTOML_MODEL_PATH = 'outputs/best_automl.pkl'
DEPLOYED_AUTOML_MODEL_DESCRIPTION = 'AutoML Registered Model'

# AutoML config settings
LABEL_COLUMN_NAME = 'quality'
DEBUG_LOG = 'capstone_automlml.log'

# constnace for inference/deployment service
INFERENCE_SOURCE_DIRECTORY = './source_dir'
INFERENCE_SCORING_SCRIPT = 'score.py'

DEPLOYED_SERVICE = 'white-wine-service'

ENV_DIR - '.' 
