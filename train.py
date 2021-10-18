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

from sklearn.linear_model import LogisticRegression
CAPSTONE_TABULAR_WINE_DATA = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'


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

# Create TabularDataset using TabularDatasetFactory
# Data is located at:

ds = TabularDatasetFactory.from_delimited_files(CAPSTONE_TABULAR_WINE_DATA, separator=';')

x, y, _ = clean_data(ds.to_pandas_dataframe())

# : Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

run = Run.get_context()



if __name__ == '__main__':
    main()