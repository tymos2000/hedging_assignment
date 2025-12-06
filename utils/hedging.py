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

def delta_vega_hedge(target_data: pd.DataFrame,
                     rep_data: pd.DataFrame,
                     hedge_frequency: int = 1) -> Dict[str, Any]:
    """
    data: DataFrame
        Contains data for a single option series
    """
    target_data = target_data.sort_values(by='date').reset_index(drop=True).copy()
    rep_data = rep_data.sort_values('date').copy()

    # align rep_date on target date gird
    rep_data = rep_data[['date', 'C', 'delta', 'vega']].copy()
    rep_data = rep_data.set_index('date')
    rep_data = rep_data.reindex(target_data['date'])

    if rep_data[['C', 'delta', 'vega']].isna().any().any():
        return None
    
    rep_data = rep_data.reset_index().rename(columns={'index': 'date'})

    N = len(target_data)

    delta_target = target_data.loc[0, 'delta']
    delta_rep = rep_data.loc[0, 'delta']
    vega_target = target_data.loc[0, 'vega']
    vega_rep = rep_data.loc[0, 'vega']

    if vega_rep == 0:
        return None

    amount_S = delta_target - (vega_target / vega_rep) * delta_rep
    amount_rep = vega_target / vega_rep 

    errors = []

    for i in range(1, N):
        C1_target = target_data.loc[i - 1, 'C']
        C2_target = target_data.loc[i, 'C']
        S1_target = target_data.loc[i - 1, 'S']
        S2_target = target_data.loc[i, 'S']

        C1_rep = rep_data.loc[i - 1, 'C']
        C2_rep = rep_data.loc[i, 'C']

        A = (C2_target - C1_target) - (amount_S * (S2_target - S1_target) + amount_rep * (C2_rep - C1_rep))

        errors.append([target_data.loc[i, 'TTM'], A])

        if i % hedge_frequency == 0:
            delta_target = target_data.loc[i, 'delta']
            delta_rep = rep_data.loc[i, 'delta']
            vega_target = target_data.loc[i, 'vega']
            vega_rep = rep_data.loc[i, 'vega']

            if vega_rep == 0:
                return None
            
            amount_S = delta_target - (vega_target / vega_rep) * delta_rep
            amount_rep = vega_target / vega_rep

    error_df = pd.DataFrame(errors, columns=['TTM', 'error']) 
    result = {
        'expiration_date' : target_data['expiration_date'].iloc[0],
        'K' : target_data['K'].iloc[0],
        'initial_moneyness' : target_data['initial_moneyness'].iloc[0],
        'hedge_frequency' : hedge_frequency,
        'errors' : error_df,
        'mse' : np.mean(error_df['error']**2),
        'mean_error' : np.mean(error_df['error']),
        'std_error' : np.std(error_df['error'])
    }
    return result

def run_delta_vega_hedge_analysis(data: pd.DataFrame,
                                  pairs: pd.DataFrame,
                                  frequencies: List[int]) -> pd.DataFrame:
    # ensure expiration_date is datetime
    data = data.copy()
    data['expiration_date'] = pd.to_datetime(data['expiration_date'])
    data['date'] = pd.to_datetime(data['date'])

    # build mapping: target option_id -> hedge_option_id
    pair_map: Dict[str, str] = (
        pairs.drop_duplicates(subset=['option_id'])
             .set_index('option_id')['hedge_option_id']
             .to_dict()
    )

    results: List[Dict[str, Any]] = []

    # loop over moneyness buckets
    for mon in ['ATM', 'ITM', 'OTM']:
        data_mon = data[data['initial_moneyness'] == mon].copy()

        # group by target option_id
        for option_id, opt_data in data_mon.groupby('option_id'):

            # check if this option has a hedge partner
            if option_id not in pair_map:
                # no longer-maturity same-strike option
                continue

            hedge_id = pair_map[option_id]
            rep_data = data[data['option_id'] == hedge_id].copy()
            if rep_data.empty:
                continue

            # restrict target to last 45 *calendar* days before its maturity
            # (if you already did this earlier, this is just a safeguard)
            exp_date = opt_data['expiration_date'].iloc[0]
            mask_45 = (exp_date - opt_data['date']).dt.days.between(0, 45)
            target_45 = opt_data[mask_45].copy().sort_values('date')

            # need at least 2 observations to compute P&L
            if len(target_45) < 2:
                continue

            for freq in frequencies:
                res = delta_vega_hedge(
                    target_data=target_45,
                    rep_data=rep_data,
                    hedge_frequency=freq
                )
                if res is not None:
                    results.append(res)
                else:
                    # e.g. missing alignment or zero vega
                    # print(f"Skipping option_id {option_id} at freq={freq}.")
                    pass

    summary = pd.DataFrame(results)
    return summary



