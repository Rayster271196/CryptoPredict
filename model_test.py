import joblib
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
# from sklearn.decomposition import PCA
import pickle


from data_extraction import *

DATA_LIMIT = 10
loaded_rf = joblib.load("Testing model\model.pkl")

def analyse_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    # print(f"Prediction: {prediction} vs Actual: {y_test}")
    #print(f"Score: {model.score(X_test,y_test)}")
    print(f"Mean_abs_error : {mean_absolute_error(y_test,prediction)}")
    y_true, Y_pred = np.array(y_test), np.array(prediction)
    y_true = y_true.astype(float)
    Y_pred = Y_pred.astype(float)
    # mean absolute percentage error
    print(f"MAPE: {np.mean(np.abs((y_true - Y_pred) / y_true)) * 100}")

    print(f"Prediction: {Y_pred} vs Actual: {y_true}")


def getTestingdata():
    # df_final = pd.DataFrame()

    depth = client.get_order_book(symbol=SYMBOL, limit=DATA_LIMIT)
    df_depth = pd.DataFrame(depth)
    df_depth = col_separator(df_depth)

    # time.sleep(TIME_TO_SLEEP)

    # df_depth['Actual price'] = "{:.2f}".format(
    #     float(client.get_symbol_ticker(symbol=SYMBOL)['price']))
    df_depth['Actual price'] = 0 # Used for groupBy
    df_depth = df_depth.astype(str)


    cols = list(df_depth.columns)

    cols.remove('Actual price')
    test_df = df_depth.groupby('Actual price')[
        cols].agg(' '.join).reset_index()

    new_df = pd.DataFrame()
    # new_df['Actual price'] = test_df['Actual price']
    # new_df['Predicted price'] = model.predict()
    for col in cols:
        new = test_df[col].str.split(" ", n=len(df_depth), expand=True)
        for i in range(len(df_depth)):
            new_df[f'{col}#'+str(i)] = new[i]

    # print(new_df)
    new_df = new_df.astype('float')
    # new_df.drop(['Actual price'], axis=1)
    # print(new_df)

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

def preprocess(test_df):
    cols = list(test_df.columns)

    ## standardise the data
    features = test_df[cols]
    standard_scaler_loaded = joblib.load("Testing model\standard_scaler.pkl")
    # norm_scaler = standard_scaler_loaded.fit(features.values)
    features = standard_scaler_loaded.transform(features.values)
    test_df[cols] = features
    
    return test_df

    



def test():
    model = loaded_rf
    test_df = getTestingdata()
    X_test = preprocess(test_df)
    prediction = model.predict(X_test)
    scalar = joblib.load("Testing model\\target_standard_scaler.pkl")
    predicted_prices_inv = scalar.inverse_transform(np.array(prediction).reshape(-1,1))
    # print(predicted_prices_inv[0][0])
    predicted_value = predicted_prices_inv[0][0]

    time.sleep(TIME_TO_SLEEP)

    actual_value = get_mid_price(pd.DataFrame(client.get_order_book(symbol=SYMBOL, limit=1)))
    # print(actual_value)

    return (predicted_value, actual_value)