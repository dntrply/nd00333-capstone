# Determine Presence of Breast Cancer

This machine learning program detects the presence (or absence) of breast cancer from pertinent data regarding physical characteristics. 

## Project Set Up and Installation

This project comprises training two models - one using AutoML and the second using HyperDrive. The best model in each case is registered. From amongst the registered models, the model with the greater accuracy is then deployed as an endpoint service. Finally, this service is invoked to make predictions. The diagram below captures the general flow and the main aspects of building the models

![image](https://user-images.githubusercontent.com/17679107/140242515-4dd97e1a-1686-4bad-a6f0-d883cda595dc.png)

## Dataset

### Overview
The dataset is at [Breast Cancer Prediction Dataset](https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset). An understanding of the data can be had at https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset/discussion/66975#509394

### Task
The task is to predict the presence of breast cancer given certain physical characteristics. There are 5 features or characteristics of the cell with a 'diagnosis' label indicating the cell is cancerous or not.

### Access
The data is downloaded as a csv file from https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset . It is then made avalabe at a publicly available github such as https://github.com/dntrply/nd00333-capstone/raw/master/dataset/Breast_cancer_data.csv
The data is read into the AzureML project using the Tabular Dataset Factory function to read from a file/URL. Once ingested, the Tabular Dataset is used from there on.

## Automated ML
The objective is a binary classification (cancerous or not) and so the primary_metric chosen is 'AUC_weighted'. Early stopping is enabled and an experiment timeout is set so as to limit the total time. Early stopping is enabled to prevent overfitting. The appropriate data and label name are specified.

### Results
From the output, it would appear that the best model is an ensemble of models. The accuracy from the best model is **0.930**

**Screenshots of the AutoML RunDetails widget**
![image](https://user-images.githubusercontent.com/17679107/140244754-10a1c576-0a76-4bd9-aa00-bd6f6f6eb9f8.png)
![image](https://user-images.githubusercontent.com/17679107/140244779-7a3f8ffc-8162-4264-a523-9a16efbd08e3.png)

**Screenshots of the best AutoML model with parameters**
![image](https://user-images.githubusercontent.com/17679107/140245359-cd978f3d-fb20-4544-9464-c22e160aa2bc.png)
![image](https://user-images.githubusercontent.com/17679107/140245261-af44248c-5539-4ec9-9e82-4cc5d2763b78.png)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
Given the nature of the data and the desired outcome, a LogisticsRegression model is chosen. We go with the scikit-learn implementation. Two parameters, inverse regularization (--C) and maximum number of iterations (---max_iter) were chosen to be optimized. The sampling chosen was random parameter sampling, with a set of discrete values provided for each parameter. The Bandit Policy with a slack factor of 0.1 was chosen as the early termination policy. The primary metric was Accuracy with a goal set to maximize this primary metric.
Training code 'train.py' was provided. It exercised the regression code and saved the ensuing accuracy and model (later used by HyperDrive to evaluate the best model)
An environment was specified - in this case the primary consideration being the conda package scikit-learn


### Results
The accuracy of the best model was **0.938**. The corresponding parameter values were --C of 10 and --max_iter of 400.

**Screenshots of the HyperParameter RunDetails widget**
![image](https://user-images.githubusercontent.com/17679107/140244847-0c08b901-9317-4d76-9bfa-9c604fefa9be.png)
![image](https://user-images.githubusercontent.com/17679107/140244898-192da5e5-f79e-4a0e-a82a-1e5096b0643f.png)
![image](https://user-images.githubusercontent.com/17679107/140244929-00bb93e6-99bd-42df-93d2-1c06412ffd9c.png)
![image](https://user-images.githubusercontent.com/17679107/140244975-70226696-3e03-40d5-9000-4c9918837ba6.png)


**Screenshots of the best HyperParameter model with parameters**
![image](https://user-images.githubusercontent.com/17679107/140245040-f52d1ad0-8a4f-498e-ae80-c0b1ae4c3719.png)
![image](https://user-images.githubusercontent.com/17679107/140245112-3fec77f7-4a15-4c38-9e38-67828f85cead.png)
![image](https://user-images.githubusercontent.com/17679107/140245157-163da4db-5460-43b7-838a-377de9d8bce7.png)


## Model Deployment
The HyperDrive model had a slightly better accuracy and was chosen to be deployed. Deployment consists of specifying an Inference configuration and a deployment configuration. Inference configuration consists of specifying the environment, the scoring code (with init and run functions). Once the deployment is successful, the scoring_URI can be retrieved from the deployment service. The scoring URI can then be used to make a HTTP request with the input data. The input data in this example is a batch array of parameter values. The endpoint is capable of taking the batch array, invoking the model prediction, and returning the predicted results. 

An example is:
```
    [[9.504, 12.44, 60.34, 273.9, 0.1024],
    [15.37, 22.76, 100.2, 728.2, 0.092], 
    [21.09, 26.57, 142.7, 1311.0, 0.1141],
    [11.04, 14.93, 70.67, 372.7, 0.07987]]
```

with the results

```
    [1, 0, 0, 1]
```
    
The results indicate that the first and last samples are likely cancerous.

**Screenshot showing model endpoint as Healthy**
![image](https://user-images.githubusercontent.com/17679107/140250416-a2b75e9e-c97f-45bb-9fa2-af6d1eb61f85.png)

## Suggestions for improvements
* Experiment with normalizing the data to determine if model accuracy can be improved.
* Determine if the input features are highly correlated. If so, remove highly correlated features prior to training
* Provide enhanced instrumentation/logging especially during inference
* Convert the model to ONNX format for greater interoperability


## Screen Recording
A video screencast demonstrating the project [may be found here](https://drive.google.com/file/d/1qg_g7IlZ6UwxP-ROVeSyin4fIQg8iOik/view?usp=sharing)



