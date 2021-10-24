import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.workspace import Workspace
from azureml.core.datastore import Datastore

from sklearn.linear_model import LogisticRegression

import capstone_constants as c_constants




def get_dataset(ws):
    """Get (or create the dataset) for training

    Returns:
    TabularDataset object

    Parameters:
    None"""

    # Create the Azure ML dataset from the preferred source
    # Note that thsi source is the same as used in train.py
    # grab the data and create a dataset
    # See if the dataset already exists - if so, skip the Dataset creation pieces

    ds_name = c_constants.DATASET_NAME
    dsets = ws.datasets.keys()

    if ds_name in dsets:
        # dataset exists
        train_ds = ws.datasets[ds_name]
    else:
        # Data set not found. Must create it
        # This is the original white wine data - not normalized
        # We will normalize the data and then create a dataset that
        # then will continue to be used
        ds = TabularDatasetFactory.from_delimited_files(c_constants.TABULAR_WINE_DATA_URI, separator=';')
        
        X, y, norm_df = clean_data(ds.to_pandas_dataframe())
        
        # Split the data into train/test sets
        # Note that the training data would be used for AutoML; 
        # the test data is not put to use in this project
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        
        # Add x and y together
        # save the new df to disk as a csv
        # Upload to a datastore
        # load from datastore as an Azure TabularDataSet
        
        # Concat two pandas dataframes togethere
        train_data = pd.concat([X_train, y_train], axis=1)
        
        # From here on - X_train contains both input + output

        # save and reload the clean data so that Azure ML can use it
        # See https://stackoverflow.com/questions/60380154/upload-dataframe-as-dataset-in-azure-machine-learning
        
        # To be able to load to datastore - the data needs to be in a folder.
        # Thus first create the directory if it does not exist
        file_path = os.path.join('.', c_constants.TRAIN_DATA_DIR, c_constants.TRAIN_DATA_FILE)
        norm_file_path = os.path.join('.', c_constants.TRAIN_DATA_DIR, c_constants.TRAIN_NORMALIZATION_PARAMETERS_FILE)
        if c_constants.TRAIN_DATA_DIR not in os.listdir():
          os.mkdir(os.path.join('.', c_constants.TRAIN_DATA_DIR))

          
        # now save the training data to disk
        train_data.to_csv(file_path, index=False)
        norm_df.to_csv(norm_file_path, index=False)
        
        # upload the file to the default datastore
        datastore = ws.get_default_datastore()
        datastore.upload(src_dir=c_constants.TRAIN_DATA_DIR, target_path=c_constants.TRAIN_DATA_DIR, overwrite=True)

        # Now Create the training dataset 
        train_ds = TabularDatasetFactory.from_delimited_files(datastore.path(os.path.join(c_constants.TRAIN_DATA_DIR, c_constants.TRAIN_DATA_FILE)))
        
        # Register the dataset so that on repeated runs, the data does not have to be fetched evey time
        train_ds = train_ds.register(workspace=ws, name=ds_name, description=c_constants.DATASET_DESCRIPTION)

    return train_ds




def clean_data(all_data):

    y_df = all_data.pop('quality').apply(lambda s: 1 if s >= 7 else 0)
    x_df = all_data
    
    # Clean and normalize the data
    x_df=(all_data-all_data.mean())/all_data.std()
    
    # Retrun the normalization information as well
    norm_df = pd.DataFrame(data=[all_data.mean(), all_data.std()], index=['Mean', 'Std'], columns=all_data.columns)
    
    return x_df, y_df, norm_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs(c_constants.TRAIN_DATA_DIR, exist_ok=True)
    
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=model, filename=c_constants.DEPLOYED_HYPER_MODEL_PATH)

# Create TabularDataset using TabularDatasetFactory
# Data is located at:

ds = TabularDatasetFactory.from_delimited_files(c_constants.TABULAR_WINE_DATA_URI, separator=';')

x, y, _ = clean_data(ds.to_pandas_dataframe())

# : Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

run = Run.get_context()




if __name__ == '__main__':
    main()