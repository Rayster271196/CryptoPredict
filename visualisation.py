import streamlit as st
import pandas as pd
import time
import numpy as np
from datetime import datetime

from Utils import if_exists_move_to, calculate_mae, calculate_mape
from model_test import test, clearFilesInDirectory
from data_extraction import SYMBOL


predicted_prices_arr = np.array([])
actual_prices_arr = np.array([])
pred_running_error = 0
benchmark_running_error = 0
df_to_store = pd.DataFrame()
MODEL = "SGD Regression"


def generate_data():
    global pred_running_error, benchmark_running_error
    global df_to_store
    global predicted_prices_arr, actual_prices_arr

    placeholder_predicted_value = st.empty()
    placeholder_actual_price = st.empty()
    placeholder_MAE = st.empty()
    placeholder_MAPE = st.empty()
    placeholder_pred_running_error = st.empty()
    placeholder_benchmark_running_error = st.empty()
    placeholder_counter = st.empty()

    while True:
        testGenerator = test()
        predicted_price = next(testGenerator)
        placeholder_predicted_value.text(f"Model Predicted Price({time.strftime('%H-%M-%S')}): {predicted_price}")
        
        with st.spinner('Please wait for a moment...'):
            actual_price = next(testGenerator)

        predicted_prices_arr  = np.append(predicted_prices_arr,np.array(predicted_price))
        actual_prices_arr  = np.append(actual_prices_arr,np.array(actual_price))
        pred_running_error = pred_running_error+abs(predicted_price-actual_price)
        if len(predicted_prices_arr)>1:
            benchmark_running_error += abs(actual_prices_arr[len(actual_prices_arr)-1] - actual_prices_arr[len(actual_prices_arr)-2])
        
        placeholder_actual_price.text(f"Actual Price({time.strftime('%H-%M-%S')})): {actual_price}")
        placeholder_MAE.text(f"MAE: {calculate_mae(predicted_prices_arr, actual_prices_arr)}")
        placeholder_MAPE.text(f"MAPE: {calculate_mape(predicted_prices_arr, actual_prices_arr)}")
        placeholder_pred_running_error.text(f"Running Error: {pred_running_error} {'👍' if pred_running_error < benchmark_running_error else '👎'}")
        placeholder_benchmark_running_error.text(f"Actual Running Error: {benchmark_running_error}")
        placeholder_counter.text(f"Counter: {len(predicted_prices_arr)}")
        
        data = {'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], 'Predicted Price': predicted_price, 'Actual Price': actual_price}
        df_to_store = df_to_store.append(data, ignore_index=True)
        

        if (len(df_to_store) % 10 == 0):
            clearFilesInDirectory('Testing model/Data/')
            df_to_store.to_csv(
                f'Testing model/Data/Visualization_newformat_{SYMBOL}_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv', index=False)
        df = pd.DataFrame(data)

        yield df
        

def main():
    if_exists_move_to('Testing model/Data/','Testing model/Data Archive')
    st.title(f'Stock Price Comparison {SYMBOL} {MODEL}')
    
    plot_placeholder = st.empty()
    
    # Create a generator for real-time data updates
    data_generator = generate_data()
    df = pd.DataFrame(columns=['Timestamp', 'Predicted Price', 'Actual Price'])
    
    # Continuously update the plot
    while True:
        # Get the next batch of predictions from the generator
        new_data = next(data_generator)
        df = df.append(new_data, ignore_index=True)
        plot_placeholder.line_chart(df.set_index('Timestamp'))
    
if __name__ == '__main__':
    main()
