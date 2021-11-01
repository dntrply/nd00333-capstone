import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

from sklearn.linear_model import LogisticRegression


def split_data(all_data):

    y_df = all_data.pop('diagnosis')
    x_df = all_data
    
    return x_df, y_df


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

    os.makedirs('outputs', exist_ok=True)
    
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=model, filename=os.path.join('outputs', 'best_hyper.pkl'))

# Create TabularDataset using TabularDatasetFactory
# Data is located at:

ds = TabularDatasetFactory.from_delimited_files('https://github.com/dntrply/nd00333-capstone/raw/master/dataset/Breast_cancer_data.csv')

x, y = split_data(ds.to_pandas_dataframe())

# : Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

run = Run.get_context()




if __name__ == '__main__':
    main()
