# Introduction

In this repository we use machine learning models namely, LSTM, Linear regression, Support Vector regression, Random Forest and Stochastic gradient descent to predict cryptocurrency prices in the BTCTUSD market using order book data.

This README is in line with the Thesis report submitted so please find information about the below appendix sections:

## Appendix A: Python code for data extraction from Binance API

### `data_extraction.py`
This file is where we collect raw L2 orderbook data up to 10 levels at t secs and the Actual price is collected at t+10 secs and then stored as a row and so on we collect data and output it to a csv file.

### `config.py`
This file stores the 'api_key' and 'api_secret' with is obtained from Binance. Also since I have used my personal keys I will not be sharing this file in the code submitted. (To run data_extraction.py these keys will have to be obtained) [Binance API link](https://www.binance.com/en/my/settings/api-management)

### `Utils.py`
This files contains common utility functions which are used through the code.

##
## Appendix B: Python code for data engineering and training all the models discussed in the paper.

### `training.py`
This file contains the bulk of data cleaning, data visualization, feature importance, data engineering, split into train test data, hyper-parameter tune models, train models and then compare the best models and take top 3 models for real time testing. 

##
## Appendix C: Python code for real-time testing the top 3 best models

### `visualisation.py`
This file contains a streamlit app which is used for visualization of the model performance on real world data. It contains information such as the MAE, MAPE, benchmark running error(to beat) and prediction running error.

### `model_test.py`
This file contains code which deals with raw orderbook data fetching and transformations needed to get the data in the form ready for the predictive model to ingest and predict, this predicted data is then passed to `visualisation.py` for visualization.

### `Testing model(Folder)`
This folder contains the model.pkl, standard_scaler.pkl(for preprocessing), target_standard_scaler.pkl(Used for preprocessing the target and inverse tranforming the predicted values)