import numpy as np
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def process(data: pd.DataFrame, lifetime: int, interest_rate: pd.DataFrame) -> pd.DataFrame:
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

if __name__ == "__main__":
    expiration_dates = [
        '2023-01-20',
        '2023-02-17',
        '2023-03-17',
        '2023-04-21',
        '2023-05-19',
        '2023-06-16',
        '2023-07-21',
        '2023-08-18',
        '2023-09-15',
        '2023-10-20',
        '2023-11-17',
        '2023-12-15'
    ]
    rates = load_data("data/raw/interest_rate_nov_2022_to_dec_2023.csv")
    for date in expiration_dates:
        data = load_data(f"data/raw/IBM_{date}.csv")
        data = process(data, lifetime=45, interest_rate=rates)
        data.to_csv(f"data/processed/IBM_{date}_processed_45.csv", index=False)
