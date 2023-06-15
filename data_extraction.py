# import pickle
import config
import pandas as pd
from binance.client import Client
import time
# from sklearn.preprocessing import PolynomialFeatures
# import numpy as np
from tqdm import tqdm
import os
import shutil
from enum import Enum
import traceback

from Utils import clearFilesInDirectory, IfExistsMoveTo

# Dangerous code, use with caution or if you're like me just use it!
import warnings
warnings.filterwarnings("ignore")

# pickled_model = pickle.load(open('./model/model.pkl', 'rb'))
# print("Model loaded!")


Tickers = Enum('Tickers', ['BNBBUSD', 'ETHUSDT'])

HOURS = 48
MINS = HOURS * 60
SYMBOL = Tickers.ETHUSDT.name
TIME_TO_SLEEP = 10
NUMBER_OF_ROWS_REQUIRED = int((MINS * 60)/10)
client = Client(config.api_key, config.api_secret)

def col_separator(df):
    df_new = pd.DataFrame()
    df_new['bid_price'] = df['bids'].apply(lambda x: float(x[0]))
    df_new['ask_price'] = df['asks'].apply(lambda x: float(x[0]))
    df_new['bid_volume'] = df['bids'].apply(lambda x: float(x[1]))
    df_new['ask_volume'] = df['asks'].apply(lambda x: float(x[1]))
    # df_new['weighted_bid'] = df_new['bid_price'] * df_new['bid_volume']
    # df_new['weighted_ask'] = df_new['ask_price'] * df_new['ask_volume']
    # df_new.drop(["bid_price", "ask_price", "bid_volume", "ask_volume"],axis=1, inplace=True)
    # df_new = df_new.astype(str).astype(float)
    # df_new = df_new.mul([i for i in range(len(df_new),0,-1)], axis=0)
    # df_new['weighted_bid']= df_new['weighted_bid'].apply(lambda x: "{:.2f}".format(float(x)))
    # df_new['weighted_ask']= df_new['weighted_ask'].apply(lambda x: "{:.2f}".format(float(x)))

    # print(df_new)
    return df_new


def get_mid_price(df):
    df_new = pd.DataFrame()
    df_new['bid_price'] = df['bids'].apply(lambda x: float(x[0]))
    df_new['ask_price'] = df['asks'].apply(lambda x: float(x[0]))
    df_new['bid_volume'] = df['bids'].apply(lambda x: float(x[1]))
    df_new['ask_volume'] = df['asks'].apply(lambda x: float(x[1]))
    return (((df_new['bid_price'] * df_new['ask_volume']) + (df_new['ask_price'] * df_new['bid_volume'])) / (df_new['bid_volume'] + df_new['ask_volume'])).values[0]
    # return (df_new.iloc[0, 0] + df_new.iloc[0, 1]) / 2


def main():
    IfExistsMoveTo('../data/checkpoints')
    df_final = pd.DataFrame()
    pbar = tqdm(total=NUMBER_OF_ROWS_REQUIRED)

    for i in range(NUMBER_OF_ROWS_REQUIRED):
        depth = client.get_order_book(symbol=SYMBOL, limit=10)
        df_depth = pd.DataFrame(depth)
        df_depth = col_separator(df_depth)

        time.sleep(TIME_TO_SLEEP)

        # df_depth['Actual price'] = "{:.2f}".format(float(client.get_symbol_ticker(symbol=SYMBOL)['price']))
        df_depth['Actual price'] = get_mid_price(
            pd.DataFrame(client.get_order_book(symbol=SYMBOL, limit=1)))
        
        df_depth = df_depth.astype(str)
        print(df_depth.head())

        cols = list(df_depth.columns)

        cols.remove('Actual price')
        test_df = df_depth.groupby('Actual price')[
            cols].agg(' '.join).reset_index()
        # print(test_df)
        print(test_df.head())
        new_df = pd.DataFrame()
        new_df['Actual price'] = test_df['Actual price']
        for col in cols:
            new = test_df[col].str.split(" ", n=len(df_depth), expand=True)
            print(new.head())
            for i in range(len(df_depth)):
                new_df[f'{col}#'+str(i)] = new[i]

        df_final = df_final.append(new_df, ignore_index=True)
        # create csv checkpoints every 10 rows so we do not loose all data if something goes wrong in the middle
        if len(df_final) % 10 == 0:
            clearFilesInDirectory('../data/checkpoints')
            df_final.to_csv(
                f'../data/checkpoints/{SYMBOL}_newformat_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv', index=False)
        pbar.update(1)
        # print(new_df)
        print(df_final)
    pbar.close()

    print(df_final)

    df_final.to_csv(
        f'../data/{SYMBOL}_newformat_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv', index=False)
    # clearFilesInDirectory('../data/checkpoints')


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        traceback.print_exc()
    # test()
    print("Done")
