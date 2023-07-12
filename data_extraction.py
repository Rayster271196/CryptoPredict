import config
import pandas as pd
from binance.client import Client
import time
from tqdm import tqdm
import traceback
import asyncio
from enum import Enum

import warnings
warnings.filterwarnings("ignore")

from Utils import clearFilesInDirectory, IfExistsMoveTo

Tickers = Enum('Tickers', ['BTCTUSD'])

HOURS = 240
MINS = HOURS * 60
SYMBOL = Tickers.BTCTUSD.name
TIME_TO_SLEEP = 10
NUMBER_OF_ROWS_REQUIRED = int((MINS * 60)/10)
client = Client(config.api_key, config.api_secret)

def col_separator(df):
    df_new = pd.DataFrame()
    df_new['bid_price'] = df['bids'].apply(lambda x: float(x[0]))
    df_new['ask_price'] = df['asks'].apply(lambda x: float(x[0]))
    df_new['bid_volume'] = df['bids'].apply(lambda x: float(x[1]))
    df_new['ask_volume'] = df['asks'].apply(lambda x: float(x[1]))
    return df_new


def get_mid_price(df):
    df_new = pd.DataFrame()
    df_new['bid_price'] = df['bids'].apply(lambda x: float(x[0]))
    df_new['ask_price'] = df['asks'].apply(lambda x: float(x[0]))
    df_new['bid_volume'] = df['bids'].apply(lambda x: float(x[1]))
    df_new['ask_volume'] = df['asks'].apply(lambda x: float(x[1]))
    return (((df_new['bid_price'] * df_new['ask_volume']) + (df_new['ask_price'] * df_new['bid_volume'])) / (df_new['bid_volume'] + df_new['ask_volume'])).values[0]


async def main():
    print(SYMBOL)
    IfExistsMoveTo('../data/checkpoints')
    df_final = pd.DataFrame()
    pbar = tqdm(total=NUMBER_OF_ROWS_REQUIRED)

    for i in range(NUMBER_OF_ROWS_REQUIRED):
        depth = client.get_order_book(symbol=SYMBOL, limit=10)
        df_depth = pd.DataFrame(depth)
        df_depth = col_separator(df_depth)

        await asyncio.sleep(TIME_TO_SLEEP)

        df_depth['Actual price'] = get_mid_price(
            pd.DataFrame(client.get_order_book(symbol=SYMBOL, limit=1)))

        df_depth = df_depth.astype(str)

        cols = list(df_depth.columns)

        cols.remove('Actual price')
        test_df = df_depth.groupby('Actual price')[
            cols].agg(' '.join).reset_index()
        
        new_df = pd.DataFrame()
        new_df['Actual price'] = test_df['Actual price']
        for col in cols:
            new = test_df[col].str.split(" ", n=len(df_depth), expand=True)
            for i in range(len(df_depth)):
                new_df[f'{col}#'+str(i)] = new[i]

        new_df['Timestamp'] = time.strftime("%Y-%m-%d__%H-%M-%S")
        df_final = df_final.append(new_df, ignore_index=True)
        if len(df_final) % 10 == 0:
            clearFilesInDirectory('../data/checkpoints')
            df_final.to_csv(
                f'../data/checkpoints/{SYMBOL}_newformat_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv', index=False)
        pbar.update(1)
    pbar.close()

    print(df_final)

    df_final.to_csv(
        f'../data/{SYMBOL}_newformat_{time.strftime("%Y-%m-%d__%H-%M-%S")}.csv', index=False)
    

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as _:
        traceback.print_exc()
