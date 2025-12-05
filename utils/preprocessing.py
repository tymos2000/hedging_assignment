import numpy as np
import pandas as pd
from tools import get_delta, implied_vol


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def construct_validation_table(data: pd.DataFrame) -> pd.DataFrame:
    nan_counts = (
        data
        .assign(iv_is_nan = (data['IV'].isna() & (data['TTM'] != 0)).astype(int))
        .groupby(['expiration_date', 'initial_moneyness'])['iv_is_nan']
        .sum()
    )
    table = nan_counts.unstack(fill_value=0)
    return table

def process_old(data: pd.DataFrame, lifetime: int, interest_rate: pd.DataFrame) -> pd.DataFrame:
    # drop missing values
    data = data.dropna()

    # ----- drop index column -----
    data.drop(columns=[data.columns[0]], inplace=True)

    # ----- add strike price -----
    strike_col_name = data.columns[2] # third column
    strike = float(strike_col_name.lstrip('C'))
    data['K'] = strike
    # rename column
    data['C'] = data[strike_col_name]
    data.drop(columns=[strike_col_name], inplace=True)

    # ----- rename underlying to S -----
    data['S'] = data['Underlying']
    data.drop(columns=['Underlying'], inplace=True)

    # ----- add TTM and restrict dates
    maturity_date = data['Date'].iloc[-1] # last date in the dataset

    data['TTM'] = (maturity_date - data['Date']).dt.days

    data = data[data['TTM'] <= lifetime].reset_index(drop=True)

    data['TTM'] = data['TTM'] / 365.0
    
    # add interest rate
    interest_rate = interest_rate[['Date', 'r']]

    data = data.merge(interest_rate, on='Date', how='left')
    
    return data

# processing for TSLA dataset structure
def process(data, r):
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'])
    data['expiration_date'] = pd.to_datetime(data['expiration_date'])

    # add TTM and restrict to 45 days
    data = data[(data['expiration_date'] - data['date']).dt.days <= 45]
    data['TTM'] = (data['expiration_date'] - data['date']).dt.days / 365.0

    # add interest rate
    r['date'] = pd.to_datetime(r['date'])
    r = r[['date', 'r']]
    data = data.merge(r, on='date', how='left')

    # add IV
    data['IV'] = data.apply(
        lambda row: implied_vol(
            C = row['C'],
            S = row['S'],
            K = row['K'],
            r = row['r'],
            ttm = row['TTM']
        ),
        axis=1
    )

    # add delta per row
    data['delta'] = data.apply(
        lambda row: get_delta(
            S = row['S'],
            K = row['K'],
            r = row['r'],
            sigma = row['IV'],
            ttm = row['TTM']
        ),
        axis=1
    )

    # eliminate expiration dates for which ATM/ITM/OTM has NaN IV
    validation_table = construct_validation_table(data)

    valid_expiries = validation_table[
        (validation_table["ATM"] == 0) &
        (validation_table["ITM"] == 0) &
        (validation_table["OTM"] == 0)
    ].index

    data = data[data['expiration_date'].isin(valid_expiries)]

    return data

if __name__ == "__main__":
    rates = load_data("data/raw/interest_rate.csv")
    data = load_data("data/raw/IBM.csv")
    processed = process(data, rates)
    processed.to_csv("data/processed/IBM_processed.csv", index=False)