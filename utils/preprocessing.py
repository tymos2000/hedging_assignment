import numpy as np
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

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


def process(data, r):
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'])
    data['expiration_date'] = pd.to_datetime(data['expiration_date'])

    # add interest rate
    r['date'] = pd.to_datetime(r['date'])
    r = r[['date', 'r']]
    data = data.merge(r, on='date', how='left')

    # add TTM
    data['TTM'] = (data['expiration_date'] - data['date']).dt.days / 365.0

    return data

if __name__ == "__main__":
    rates = load_data("data/raw/interest_rate.csv")
    data = load_data("data/raw/TSLA.csv")
    processed = process(data, rates)
    processed.to_csv("data/processed/TSLA_processed.csv", index=False)
