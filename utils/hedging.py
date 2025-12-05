from typing import Dict, Any, List
import numpy as np
import pandas as pd

def delta_hedge(data: pd.DataFrame, hedge_frequency: int = 1) -> Dict[str, Any]:
    """
    data: DataFrame
        Contains data for a single option series
    """

    data = data.sort_values(by='date').reset_index(drop=True)
    N = len(data)

    delta = data.loc[0, 'delta']

    errors = []

    for i in range(1, N):
        C1 = data.loc[i - 1, 'C']
        C2 = data.loc[i, 'C']
        S1 = data.loc[i - 1, 'S']
        S2 = data.loc[i, 'S']

        A = (C2 - C1) - delta * (S2 - S1)

        errors.append([data.loc[i, 'TTM'], A])

        if i % hedge_frequency == 0:
            delta = data.loc[i, 'delta']
    
    error_df = pd.DataFrame(errors, columns=['TTM', 'error'])

    result = {
        'expiration_date' : data['expiration_date'].iloc[0],
        'K' : data['K'].iloc[0],
        'initial_moneyness' : data['initial_moneyness'].iloc[0],
        'hedge_frequency' : hedge_frequency,
        'errors' : error_df,
        'mse' : np.mean(error_df['error']**2),
        'mean_error' : np.mean(error_df['error']),
        'std_error' : np.std(error_df['error'])
    }
    return result



def run_delta_hedge_analysis(data: pd.DataFrame,
                             frequencies: List[int]) -> pd.DataFrame:
    results = []

    for mon in ['ATM', 'ITM', 'OTM']:
        data_mon = data[data['initial_moneyness'] == mon].copy()

        for freq in frequencies:
            for _, opt_data in data_mon.groupby('option_id'):
                res = delta_hedge(data = opt_data,
                                hedge_frequency = freq)
                if res is not None:
                    results.append(res)
                else:
                    print(f"Skipping option_id {opt_data['option_id'].iloc[0]}.")
        
    summary = pd.DataFrame(results)
    return summary

def collect_errors(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        df = row['errors']
        df = df 

