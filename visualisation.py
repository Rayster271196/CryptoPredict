import streamlit as st
import pandas as pd
import time
import numpy as np
# import random
from datetime import datetime
from Utils import IfExistsMoveTo

from model_test import test, clearFilesInDirectory
from data_extraction import SYMBOL



def calculate_mae(predicted, actual):
    absolute_errors = np.abs(predicted - actual)
    mae = np.mean(absolute_errors)
    return mae

def calculate_mape(predicted, actual):
    absolute_percentage_errors = np.abs((predicted - actual) / actual) * 100
    mape = np.mean(absolute_percentage_errors)
    return mape

# Example usage
predicted_prices_arr = np.array([])
actual_prices_arr = np.array([])
df_to_store = pd.DataFrame()

# i=100
# Function to simulate real-time data updates
def generate_data():
    global df_to_store
    global predicted_prices_arr, actual_prices_arr
    placeholder_predicted_value = st.empty()
    placeholder_actual_price = st.empty()
    placeholder_MAE = st.empty()
    placeholder_MAPE = st.empty()
    placeholder_counter = st.empty()
    while True:
        # Simulate fetching new predicted and actual prices
        with st.spinner('Please wait for a moment...'):
            predicted_price, actual_price = test()
        # st.success('Predicted values generated')
        predicted_prices_arr  = np.append(predicted_prices_arr,np.array(predicted_price))
        actual_prices_arr  = np.append(actual_prices_arr,np.array(actual_price))
        # predicted_price, actual_price = i+random.randint(0, 10), i+random.randint(0, 10)

        placeholder_predicted_value.text(f"Predicted Price: {predicted_price}")
        placeholder_actual_price.text(f"Actual Price: {actual_price}")
        placeholder_MAE.text(f"MAE: {calculate_mae(predicted_prices_arr, actual_prices_arr)}")
        placeholder_MAPE.text(f"MAPE: {calculate_mape(predicted_prices_arr, actual_prices_arr)}")
        placeholder_counter.text(f"Counter: {len(predicted_prices_arr)}")
        # i+=random.randint(0, 10)

        # Create a DataFrame with the current timestamp and prices
        # print(f"Predicted: {predicted_price}, Actual Price: {actual_price}")
        # print(f"Predicted arr: {predicted_prices_arr}")
        # print()
        data = {'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], 'Predicted Price': predicted_price, 'Actual Price': actual_price}
        df_to_store = df_to_store.append(data, ignore_index=True)
        

        if (len(df_to_store) % 10 == 0):
            clearFilesInDirectory('Testing model/Data/')
            df_to_store.to_csv(
                f'Testing model/Data/Visualization_newformat_{SYMBOL}_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv', index=False)
        df = pd.DataFrame(data)
        
        # Yield the DataFrame
        yield df
        
        # Pause for 10 seconds
        # time.sleep(10)

# Streamlit application
def main():
    IfExistsMoveTo('Testing model/Data/','Testing model/Data Archive')

    st.title('Stock Price Comparison')
    
    # Create a placeholder for the plot
    plot_placeholder = st.empty()
    
    # Create a generator for real-time data updates
    
    data_generator = generate_data()
    
    # Initialize an empty DataFrame for storing the data
    df = pd.DataFrame(columns=['Timestamp', 'Predicted Price', 'Actual Price'])
    
    # Continuously update the plot
    while True:
        # Get the next batch of data from the generator
        new_data = next(data_generator)
        # print(new_data)
        
        # Append the new data to the existing DataFrame
        df = df.append(new_data, ignore_index=True)
        
        # Update the plot
        plot_placeholder.line_chart(df.set_index('Timestamp'))
    
# Run the Streamlit application
if __name__ == '__main__':
    main()
