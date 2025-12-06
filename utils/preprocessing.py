import numpy as np
import pandas as pd
from tools import get_delta, get_vega, implied_vol


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

def validate_option_history(data, all_trading_days):
    data = data.sort_values('date')
    expiration = data['expiration_date'].iloc[0]

    expected_start = expiration - pd.Timedelta(days=90)

    actual_trading_days = pd.Index(sorted(data['date'].unique()))

    expected_trading_days = all_trading_days[
        (all_trading_days >= expected_start)
        & (all_trading_days <= expiration)
    ]

    is_correct = actual_trading_days.equals(expected_trading_days)
    return is_correct

def classify_moneyness(delta):
    if delta >= 0.65:
        return 'ITM'
    elif delta <= 0.35:
        return 'OTM'
    elif 0.45 <= delta <= 0.55:
        return 'ATM'
    else:
        return None

# processing for TSLA dataset structure
def process(data, r):
    data['date'] = pd.to_datetime(data['date'])
    data['expiration_date'] = pd.to_datetime(data['expiration_date'])

    all_trading_days = pd.Index(sorted(data['date'].unique()))

    # add TTM and restrict to 45 days
    data = data[(data['expiration_date'] - data['date']).dt.days <= 45]
    data['TTM'] = (data['expiration_date'] - data['date']).dt.days / 365.0

    

    validation = (
        data
        .groupby('option_id')
        .apply(lambda opt: validate_option_history(opt, all_trading_days))
        .reset_index(name='is_valid')
    )

    valid_ids = validation.loc[validation['is_valid'], 'option_id']
    data = data[data['option_id'].isin(valid_ids)]

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

    # eliminate rows with any NaNs beside the last day
    mask_relevant = data['TTM'] != 0
    invalid_options = (
        data[mask_relevant]
        .groupby('option_id')
        .apply(lambda g: g.isna().any(axis=1).any())
    )
    invalid_ids = invalid_options[invalid_options].index

    data = data[~data['option_id'].isin(invalid_ids)]

    first_rows = (
        data.sort_values('date')
        .groupby('option_id')
        .first()
        .reset_index()
    )
    first_rows['delta_start'] = first_rows['delta']
    first_rows['initial_moneyness'] = first_rows['delta'].apply(classify_moneyness)
    first_rows = first_rows.dropna(subset=['initial_moneyness'])

    moneyness_info = first_rows[['option_id', 'initial_moneyness', 'delta_start']]

    data = data.merge(moneyness_info, on='option_id', how='inner')



    print('Options left:', data['option_id'].nunique())
    return data

def process_delta_vega(data, r):
    data['date'] = pd.to_datetime(data['date'])
    data['expiration_date'] = pd.to_datetime(data['expiration_date'])

    all_trading_days = pd.Index(sorted(data['date'].unique()))

    # add TTM and restrict to 45 days
    data = data[(data['expiration_date'] - data['date']).dt.days <= 90]

    data['TTM'] = (data['expiration_date'] - data['date']).dt.days / 365.0

    

    validation = (
        data
        .groupby('option_id')
        .apply(lambda opt: validate_option_history(opt, all_trading_days))
        .reset_index(name='is_valid')
    )

    valid_ids = validation.loc[validation['is_valid'], 'option_id']
    data = data[data['option_id'].isin(valid_ids)]

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

    # add vega per row
    data['vega'] = data.apply(
        lambda row: get_vega(
            S = row['S'],
            K = row['K'],
            r = row['r'],
            sigma = row['IV'],
            ttm = row['TTM']
        ),
        axis=1
    )

    # eliminate options with any NaNs beside the last day
    mask_relevant = data['TTM'] != 0
    invalid_options = (
        data[mask_relevant]
        .groupby('option_id')
        .apply(lambda g: g.isna().any(axis=1).any())
    )
    invalid_ids = invalid_options[invalid_options].index

    data = data[~data['option_id'].isin(invalid_ids)]

    first_rows = (
        data.sort_values('date')
        .groupby('option_id')
        .first()
        .reset_index()
    )
    first_rows['delta_start'] = first_rows['delta']
    first_rows['initial_moneyness'] = first_rows['delta'].apply(classify_moneyness)
    first_rows = first_rows.dropna(subset=['initial_moneyness'])

    moneyness_info = first_rows[['option_id', 'initial_moneyness', 'delta_start']]

    data = data.merge(moneyness_info, on='option_id', how='inner')



    print('Options left:', data['option_id'].nunique())
    return data

if __name__ == "__main__":
    ticker = 'AAPL'
    rates = load_data("data/raw/interest_rate.csv")
    data = load_data(f"data/raw/{ticker}.csv")
    processed = process_delta_vega(data, rates)
    summary = data.groupby('expiration_date')['K'].unique().sort_index()
    print('Unique expiration dates:', processed['expiration_date'].nunique())
    print(summary)
    processed.to_csv(f"data/processed/{ticker}_processed_vega.csv", index=False)