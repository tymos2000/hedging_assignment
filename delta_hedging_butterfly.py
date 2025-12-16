import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ticker = 'AAPL'
data = pd.read_csv(f"AAPL_processed_vega.csv")

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
butterfly_df = pd.DataFrame(butterflies)

print("Number of expirations with a usable butterfly:", len(butterfly_df))
print(butterfly_df)

'''
option_id = ['AAPL_2022-05-20_K155', 'AAPL_2022-03-18_K155', 'AAPL_2022-05-20_K165', 'AAPL_2022-03-18_K170', 'AAPL_2022-03-18_K185', 'AAPL_2022-05-20_K185']


start_date = '2022-02-22'
end_date = '2022-03-18'

df_filtered = data[data['option_id'].isin(option_id) & (data['date'] >= start_date) & (data['date'] <= end_date)].copy()
df2 = data[(data['option_id'] == 'AAPL_2022-05-20_K155') & (data['date']>= start_date) & (data['date']<= end_date)].copy()
'''








'''
option_id1h = 'AAPL_2022-05-20_K155'
option_id2h = 'AAPL_2022-05-20_K165'
option_id3h = 'AAPL_2022-05-20_K185'
option_id1 = 'AAPL_2022-03-18_K155'
option_id2 = 'AAPL_2022-03-18_K170'
option_id3 = 'AAPL_2022-03-18_K185'
'''



def statistics(option_id1, option_id2, option_id3, start_date, end_date, max_x=12):
    option_id = [option_id1, option_id2, option_id3]

    df_filtered = data[
        data['option_id'].isin(option_id)
        & (data['date'] >= start_date)
        & (data['date'] <= end_date)
    ].copy()

    df2 = data[
        (data['option_id'] == option_id1)
        & (data['date'] >= start_date)
        & (data['date'] <= end_date)
    ].copy()

    date_vector_np = df2['date'].to_numpy()
    date_vector_np = date_vector_np[1:]  # wie bei dir

    MSE4plot = []
    X = []

    for x in range(1, max_x):

        pi = []

        delta1 = data[(data['option_id'] == option_id1)].iloc[0]['delta_start']
        delta2 = data[(data['option_id'] == option_id2)].iloc[0]['delta_start']
        delta3 = data[(data['option_id'] == option_id3)].iloc[0]['delta_start']
        delta = delta1 - delta2 + delta3

        for i, current_date in enumerate(date_vector_np):

            C1 = df_filtered[
                (df_filtered['date'] == current_date)
                & (df_filtered['option_id'] == option_id1)
            ].iloc[0]['C']
            C2 = df_filtered[
                (df_filtered['date'] == current_date)
                & (df_filtered['option_id'] == option_id2)
            ].iloc[0]['C']
            C3 = df_filtered[
                (df_filtered['date'] == current_date)
                & (df_filtered['option_id'] == option_id3)
            ].iloc[0]['C']
            C = C1 - 2*C2 + C3

            S = df_filtered[
                (df_filtered['date'] == current_date)
                & (df_filtered['option_id'] == option_id1)
            ].iloc[0]['S']

            pi.append(C - delta * S)

            delta1 = data[
                (data['date'] == current_date)
                & (data['option_id'] == option_id1)
            ].iloc[0]['delta']
            delta2 = data[
                (data['date'] == current_date)
                & (data['option_id'] == option_id2)
            ].iloc[0]['delta']
            delta3 = data[
                (data['date'] == current_date)
                & (data['option_id'] == option_id3)
            ].iloc[0]['delta']

            if i % x == 0:
                delta = delta1 - delta2 + delta3

        # MSE berechnen
        MSE = 0.0
        for i in range(1, len(pi)):
            MSE += (pi[i] - pi[i-1])**2

        current_date = start_date
        C1 = df_filtered[
            (df_filtered['date'] == current_date)
            & (df_filtered['option_id'] == option_id1)
        ].iloc[0]['C']
        C2 = df_filtered[
            (df_filtered['date'] == current_date)
            & (df_filtered['option_id'] == option_id2)
        ].iloc[0]['C']
        C3 = df_filtered[
            (df_filtered['date'] == current_date)
            & (df_filtered['option_id'] == option_id3)
        ].iloc[0]['C']
        C = C1 + C2 + C3

        MSE = MSE / (len(pi) - 1)

        X.append(x)
        MSE4plot.append(MSE)

    # Einzelnes Portfolio (optional) plotten:
    plt.scatter(X, MSE4plot, label=option_id1)

    return np.array(X), np.array(MSE4plot)


# ---- Portfolios durchlaufen ----

X, MSE4plota = statistics(
    'AAPL_2022-05-20_K150',
    'AAPL_2022-05-20_K165',
    'AAPL_2022-05-20_K180',
    '2022-04-01',
    '2022-05-01'
)

X, MSE4plotb = statistics(
    'AAPL_2021-03-19_K120',
    'AAPL_2021-03-19_K130',
    'AAPL_2021-03-19_K145',
    '2021-02-01',
    '2021-03-01'
)

X, MSE4plotc = statistics(
    'AAPL_2022-05-20_K155',
    'AAPL_2022-05-20_K165',
    'AAPL_2022-05-20_K185',
    '2022-04-01',
    '2022-05-01'
)

X, MSE4plotd = statistics(
    'AAPL_2022-03-18_K155',
    'AAPL_2022-03-18_K170',
    'AAPL_2022-03-18_K185',
    '2022-02-01',
    '2022-03-01'
)

# ---- Mean & Std über Portfolios ----

MSE_matrix = np.vstack([MSE4plota, MSE4plotb, MSE4plotc, MSE4plotd])
mean = MSE_matrix.mean(axis=0)
standard_deviation = MSE_matrix.std(axis=0, ddof=1)  # sample std

# Plot: Einzel-Portfolios + Mittelwert mit Errorbars
plt.figure()
'''
plt.scatter(X, MSE4plota, alpha=0.6, label='Portfolio A')
plt.scatter(X, MSE4plotb, alpha=0.6, label='Portfolio B')
plt.scatter(X, MSE4plotc, alpha=0.6, label='Portfolio C')
plt.scatter(X, MSE4plotd, alpha=0.6, label='Portfolio D')
'''
plt.errorbar(
    X,
    mean,
    yerr=standard_deviation,
    fmt='-o',
    capsize=4,
    label='Mean ± 1 Std'
)

plt.xlabel("Rehedge-Intervall [Days]")
plt.ylabel("MSE (normalized)")
plt.legend()
plt.grid(True)
plt.show()
