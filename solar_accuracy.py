import pandas as pd, numpy as np, datetime, time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor

CSV       = "niteditedfinal.csv"
TEMP_COL  = "Temperature"
GHI_COL   = "GHI"

# Load and clean data
df = (pd.read_csv(CSV)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[TEMP_COL, GHI_COL]))

# Features: cyclic encoding of time
df["sin_hour"]   = np.sin(2 * np.pi * df["Hour"]   / 24)
df["cos_hour"]   = np.cos(2 * np.pi * df["Hour"]   / 24)
df["sin_month"]  = np.sin(2 * np.pi * df["Month"]  / 12)
df["cos_month"]  = np.cos(2 * np.pi * df["Month"]  / 12)
df["sin_minute"] = np.sin(2 * np.pi * df["Minute"] / 60)
df["cos_minute"] = np.cos(2 * np.pi * df["Minute"] / 60)

FEATURES = ["sin_hour", "cos_hour", "sin_month", "cos_month", "sin_minute", "cos_minute"]

train_df = df[df["Year"].isin([2017, 2018])]
test_df  = df[df["Year"] == 2019]

X_train = train_df[FEATURES].values
X_test  = test_df[FEATURES].values
yT_train, yT_test = train_df[TEMP_COL].values, test_df[TEMP_COL].values
yG_train, yG_test = train_df[GHI_COL ].values, test_df[GHI_COL ].values

# Models
models = {
    "XGBoost":  {"temp": XGBRegressor(verbosity=0, random_state=42),
                 "ghi" : XGBRegressor(verbosity=0, random_state=42)},
    "AdaBoost": {"temp": AdaBoostRegressor(random_state=42),
                 "ghi" : AdaBoostRegressor(random_state=42)},
    "CatBoost": {"temp": CatBoostRegressor(verbose=0, random_state=42),
                 "ghi" : CatBoostRegressor(verbose=0, random_state=42)}
}

# Train all models
for m in models.values():
    m["temp"].fit(X_train, yT_train)
    m["ghi" ].fit(X_train, yG_train)

# Precompute metrics
metrics = {}
for name, m in models.items():
    pT = m["temp"].predict(X_test)
    pG = m["ghi" ].predict(X_test)
    metrics[name] = {
        "R2_T" : r2_score(yT_test, pT),
        "R2_G" : r2_score(yG_test, pG),
        "MAE_T": mean_absolute_error(yT_test, pT),
        "MAE_G": mean_absolute_error(yG_test, pG),
        "MSE_T": mean_squared_error(yT_test, pT),
        "MSE_G": mean_squared_error(yG_test, pG)
    }

# Constants for power calculation
ETA, AREA = 0.18, 7.4322

try:
    while True:
        now = datetime.datetime.now()
        print(f"\n {now.strftime('%Y-%m-%d %H:%M:%S')}")

        feat = np.array([[
            np.sin(2*np.pi*now.hour/24),
            np.cos(2*np.pi*now.hour/24),
            np.sin(2*np.pi*now.month/12),
            np.cos(2*np.pi*now.month/12),
            np.sin(2*np.pi*now.minute/60),
            np.cos(2*np.pi*now.minute/60)
        ]])

        for name, m in models.items():
            t = np.clip(m["temp"].predict(feat)[0], -50, 60)
            g = max(0, m["ghi"].predict(feat)[0])
            p = ETA * AREA * g * max(0, 1 - 0.05 * (t - 25))
            acc = metrics[name]

            print(f"{name:8s} | T={t:5.1f}°C  GHI={g:7.1f} W/m²  P={p:7.2f} W")
            print(f"         MAE_T={acc['MAE_T']:.2f}  MSE_T={acc['MSE_T']:.2f}  R²_T={acc['R2_T']:.3f}")
            print(f"         MAE_G={acc['MAE_G']:.2f}  MSE_G={acc['MSE_G']:.2f}  R²_G={acc['R2_G']:.3f}")

        time.sleep(10)

except KeyboardInterrupt:
    print("\n⛔ Stopped by user.")
