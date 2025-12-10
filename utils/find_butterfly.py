import pandas as pd
import numpy as np

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
print(butterfly_df.head())
