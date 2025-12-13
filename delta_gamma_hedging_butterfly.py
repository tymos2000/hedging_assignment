import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ticker = 'AAPL'
data = pd.read_csv(f"/Users/snoopy/Desktop/Project/Pulled/hedging_assignment/data/processed/AAPL_processed_vega.csv")

# ---- 1) Per-option table with one row per option_id ----
per_option = (
    data[['option_id', 'expiration_date', 'K', 'delta_start']]
    .drop_duplicates()
    .copy()
)

per_option['expiration_date'] = pd.to_datetime(per_option['expiration_date'])

butterflies = []  # will store one butterfly per expiry (if found)

# ---- 2) Loop over expirations ----
for exp, grp in per_option.groupby('expiration_date'):
    # all strikes for this expiry
    strikes = sorted(grp['K'].unique())
    if len(strikes) < 3:
        # not enough strikes to form a butterfly
        continue

    # ---- 2a) Find ATM-ish strike K2 based on delta_start ≈ 0.5 ----
    # average delta_start per strike
    avg_delta_by_K = (
        grp.groupby('K')['delta_start']
           .mean()
           .reset_index()
    )
    # get row where |delta - 0.5| is minimised
    idx_atm = (avg_delta_by_K['delta_start'] - 0.5).abs().idxmin()
    K2 = float(avg_delta_by_K.loc[idx_atm, 'K'])

    # ---- 2b) Find K1 < K2 and K3 > K2 giving "most symmetric" spacing ----
    Ks_below = [k for k in strikes if k < K2]
    Ks_above = [k for k in strikes if k > K2]

    if not Ks_below or not Ks_above:
        # can't form butterfly around ATM if no both sides exist
        continue

    best_tuple = None
    best_asym = np.inf

    # Try all combos K1 < K2 and K3 > K2
    for K1 in Ks_below:
        for K3 in Ks_above:
            d1 = K2 - K1
            d2 = K3 - K2
            if d1 <= 0 or d2 <= 0:
                continue

            # asymmetry measure: how unequal are the wings
            asym = abs(d1 - d2)

            # optional: avoid extremely tiny wings (e.g. 0.5 vs 0.5)
            # if d1 < 0.5 * (min(strikes[1:] - strikes[:-1])) or d2 < same: skip
            # (you can leave it out if you don't care)

            if asym < best_asym:
                best_asym = asym
                best_tuple = (K1, K2, K3, d1, d2)

    if best_tuple is None:
        continue

    K1, K2, K3, d1, d2 = best_tuple

    # ---- 2c) pick the concrete option_ids for K1, K2, K3 ----
    opt1 = grp[grp['K'] == K1].iloc[0]['option_id']
    opt2 = grp[grp['K'] == K2].iloc[0]['option_id']
    opt3 = grp[grp['K'] == K3].iloc[0]['option_id']

    butterflies.append({
        'expiration_date': exp,
        'K1': K1,
        'K2': K2,
        'K3': K3,
        'opt_id_K1': opt1,
        'opt_id_K2': opt2,
        'opt_id_K3': opt3,
        'wing_left': d1,
        'wing_right': d2,
        'asymmetry': best_asym
    })

# ---- 3) Collect all found butterflies in a DataFrame ----
butterfly_df = pd.DataFrame(butterflies)[1:]

print("Number of expirations with a usable butterfly:", len(butterfly_df))

# ============================================================
# 2) Helper: make sure date column is datetime
# ============================================================
def prepare_data(data):
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    return df


# ============================================================
# 3) Helper: get initial delta for one option
# ============================================================
def get_delta_start(data, option_id):
    sub = data[data['option_id'] == option_id].sort_values('date')
    if sub.empty:
        raise ValueError(f"No data for option_id {option_id}")
    if 'delta_start' in sub.columns:
        return sub.iloc[0]['delta_start']
    else:
        return sub.iloc[0]['delta']


# ============================================================
# 4) Compute MSE for a single butterfly over one time window
# ============================================================
def hedging_mse_for_butterfly(
    data,
    opt_id_K1,
    opt_id_K2,
    opt_id_K3,
    start_date,
    end_date,
    max_x=12
):
    """
    Compute MSE over rehedging intervals x = 1,...,max_x-1 for one butterfly.

    Returns:
        X:       array of hedge intervals
        MSE_arr: array of MSE values (same length as X)
    """

    option_ids = [opt_id_K1, opt_id_K2, opt_id_K3]

    df_filtered = data[
        data['option_id'].isin(option_ids)
        & (data['date'] >= start_date)
        & (data['date'] <= end_date)
    ].copy()

    if df_filtered.empty:
        # no data in this window → return NaNs
        X = np.arange(1, max_x)
        return X, np.full_like(X, np.nan, dtype=float)

    # use K1 leg for the date grid
    df2 = df_filtered[df_filtered['option_id'] == opt_id_K1].copy()
    df2 = df2.sort_values('date')

    if df2.shape[0] < 2:
        X = np.arange(1, max_x)
        return X, np.full_like(X, np.nan, dtype=float)

    date_vector_np = df2['date'].to_numpy()
    # optional: skip very first day (like in your original code)
    date_vector_np = date_vector_np[1:]

    MSE_list = []
    X_list = []

    for x in range(1, max_x):
        pi = []

        # initial delta
        delta1 = get_delta_start(data, opt_id_K1)
        delta2 = get_delta_start(data, opt_id_K2)
        delta3 = get_delta_start(data, opt_id_K3)
        delta = delta1 - delta2 + delta3

        for i, current_date in enumerate(date_vector_np):
            mask_date = (df_filtered['date'] == current_date)

            try:
                C1 = df_filtered[mask_date & (df_filtered['option_id'] == opt_id_K1)].iloc[0]['C']
                C2 = df_filtered[mask_date & (df_filtered['option_id'] == opt_id_K2)].iloc[0]['C']
                C3 = df_filtered[mask_date & (df_filtered['option_id'] == opt_id_K3)].iloc[0]['C']
                C = C1 - C2 + C3

                S = df_filtered[mask_date & (df_filtered['option_id'] == opt_id_K1)].iloc[0]['S']
            except IndexError:
                # missing data for this day → skip this date
                continue

            pi.append(C - delta * S)

            # new model deltas at this date
            try:
                d1 = data[(data['date'] == current_date) & (data['option_id'] == opt_id_K1)].sort_values('date').iloc[0]['delta']
                d2 = data[(data['date'] == current_date) & (data['option_id'] == opt_id_K2)].sort_values('date').iloc[0]['delta']
                d3 = data[(data['date'] == current_date) & (data['option_id'] == opt_id_K3)].sort_values('date').iloc[0]['delta']
            except IndexError:
                # missing delta → keep old delta
                d1, d2, d3 = delta1, delta2, delta3

            if i % x == 0:
                delta = d1 - d2 + d3

        if len(pi) < 2:
            MSE_list.append(np.nan)
            X_list.append(x)
            continue

        # MSE über P&L-Schritte
        MSE = 0.0
        for j in range(1, len(pi)):
            MSE += (pi[j] - pi[j-1])**2

        # normalisieren mit C(start)^2
        try:
            start_mask = (df_filtered['date'] == start_date)
            C1_0 = df_filtered[start_mask & (df_filtered['option_id'] == opt_id_K1)].iloc[0]['C']
            C2_0 = df_filtered[start_mask & (df_filtered['option_id'] == opt_id_K2)].iloc[0]['C']
            C3_0 = df_filtered[start_mask & (df_filtered['option_id'] == opt_id_K3)].iloc[0]['C']
            C0 = C1_0 - C2_0 + C3_0
        except IndexError:
            # if we don't have exactly start_date, use first available date in window
            df_start = df_filtered.sort_values('date').groupby('option_id').first().reset_index()
            C1_0 = df_start[df_start['option_id'] == opt_id_K1].iloc[0]['C']
            C2_0 = df_start[df_start['option_id'] == opt_id_K2].iloc[0]['C']
            C3_0 = df_start[df_start['option_id'] == opt_id_K3].iloc[0]['C']
            C0 = C1_0 + C2_0 + C3_0

        MSE = MSE / (len(pi) - 1)

        MSE_list.append(MSE)
        X_list.append(x)

    return np.array(X_list), np.array(MSE_list)


# ============================================================
# 5) Run over all butterflies and plot
# ============================================================
def run_all_butterflies(data, butterfly_df,
                        window_days_before_exp=30,
                        max_x=12,
                        plot_individual=True):
    """
    Loop over all butterflies in butterfly_df, compute MSE vs. rehedge interval,
    and plot mean ± std over all butterflies.
    """

    data = prepare_data(data)
    bf = butterfly_df.copy()
    bf['expiration_date'] = pd.to_datetime(bf['expiration_date'])

    all_MSE = []
    X_ref = None
    used_butterflies = 0

    plt.figure(figsize=(8, 5))

    for idx, row in bf.iterrows():
        opt_id_K1 = row['opt_id_K1']
        opt_id_K2 = row['opt_id_K2']
        opt_id_K3 = row['opt_id_K3']
        exp_date = row['expiration_date']

        start_date = exp_date - pd.Timedelta(days=window_days_before_exp)
        end_date = exp_date

        X, MSE = hedging_mse_for_butterfly(
            data,
            opt_id_K1,
            opt_id_K2,
            opt_id_K3,
            start_date,
            end_date,
            max_x=max_x
        )

        if np.all(np.isnan(MSE)):
            continue  # skip completely empty

        if X_ref is None:
            X_ref = X
        else:
            # ensure same X-grid
            if not np.array_equal(X_ref, X):
                # in practice this shouldn't happen, but we can guard anyway
                min_len = min(len(X_ref), len(X))
                X = X[:min_len]
                MSE = MSE[:min_len]
                X_ref = X_ref[:min_len]

        all_MSE.append(MSE)
        used_butterflies += 1

        if plot_individual:
            plt.scatter(X, MSE, alpha=0.15, s=10)

    if used_butterflies == 0:
        raise RuntimeError("No butterflies produced valid MSE values.")

    all_MSE_arr = np.vstack(all_MSE)
    mean = np.nanmean(all_MSE_arr, axis=0)
    std = np.nanstd(all_MSE_arr, axis=0, ddof=1)

    # mean curve with error bars
    plt.errorbar(
        X_ref,
        mean,
        yerr=std,
        fmt='-o',
        capsize=4,
        label='Mean ± 1 Std',
        linewidth=2
    )

    plt.xlabel("Rehedge interval x (steps between delta updates)")
    plt.ylabel("Normalised MSE")
    plt.title(f"Hedging error vs. rehedge interval\n(over {used_butterflies} butterflies)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return X_ref, mean, std


# ============================================================
# 6) Script entry point
# ============================================================
if __name__ == "__main__":
    # Here you just need to have `data` and `butterfly_df` defined.
    # Then uncomment the next line:

    X, mean_MSE, std_MSE = run_all_butterflies(data, butterfly_df)

    # print("X (rehedge intervals):", X)
    # print("Mean MSE:", mean_MSE)
    # print("Std MSE:", std_MSE)
    pass
