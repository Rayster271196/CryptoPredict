import joblib
from sklearn.metrics import mean_absolute_error
import numpy as np

from data_extraction import *

DATA_LIMIT = 10
loaded_rf = joblib.load("Testing model\model.pkl")

# Getting data from Binance and perform data engineering
def getTestingdata():
    depth = client.get_order_book(symbol=SYMBOL, limit=DATA_LIMIT)
    df_depth = pd.DataFrame(depth)
    df_depth = col_separator(df_depth)
    
    df_depth['Actual price'] = 0
    df_depth = df_depth.astype(str)
    cols = list(df_depth.columns)

    cols.remove('Actual price')
    test_df = df_depth.groupby('Actual price')[
        cols].agg(' '.join).reset_index()

    new_df = pd.DataFrame()

    for col in cols:
        new = test_df[col].str.split(" ", n=len(df_depth), expand=True)
        for i in range(len(df_depth)):
            new_df[f'{col}#'+str(i)] = new[i]

    new_df = new_df.astype('float')

    lst = [i for i in range(DATA_LIMIT)]
    new_final_df = pd.DataFrame()
    for i in lst:
        for _,value in enumerate(new_df.columns):
            if '#' in value: #exclude the actual price
                if f"bid_price#{i}" in value:
                    bid_price = value
                if f"ask_price#{i}" in value:
                    ask_price = value
                if f"bid_volume#{i}" in value:
                    bid_volume = value
                if f"ask_volume#{i}" in value:
                    ask_volume = value
        new_final_df[f"rank#{i}"] = ((new_df[ask_volume] * new_df[bid_price]) + (new_df[bid_volume] * new_df[ask_price]))/(new_df[bid_volume] + new_df[ask_volume])
    return new_final_df

# Hanldes the preprocess of the orderbook data
def preprocess(test_df):
    cols = list(test_df.columns)
    features = test_df[cols]
    standard_scaler_loaded = joblib.load("Testing model\standard_scaler.pkl")
    features = standard_scaler_loaded.transform(features.values)
    test_df[cols] = features
    return test_df


# We fetch the orderbook data and make our prediction and yield predicted price and actual price
def test():
    model = loaded_rf
    # get raw orderbook data
    test_df = getTestingdata()

    # preprocess data before prediction
    X_test = preprocess(test_df)
    prediction = model.predict(X_test)
    scalar = joblib.load("Testing model\\target_standard_scaler.pkl")
    predicted_prices_inv = scalar.inverse_transform(np.array(prediction).reshape(-1,1))
    predicted_value = predicted_prices_inv[0][0]
    yield predicted_value
    time.sleep(TIME_TO_SLEEP)

    actual_value = get_mid_price(pd.DataFrame(client.get_order_book(symbol=SYMBOL, limit=1)))
    yield actual_value