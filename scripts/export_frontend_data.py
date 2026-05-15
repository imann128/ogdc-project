"""
export_frontend_data.py
=======================
step 10 - export json data files for the react frontend

reads the cleaned csv and processed data, exports 3 json files
into frontend/public/ so the react app can fetch them.

input:   data/processed/ogdc_cleaned.csv
outputs: frontend/public/price_data.json
         frontend/public/kpis.json
         frontend/public/annual_stats.json

also copies all png images from outputs/images/ -> frontend/public/
so the react app can display them.

run:     python scripts/export_frontend_data.py
"""

import sys, os, shutil, json
import pandas as pd
import numpy as np

# hardcoded paths
base = os.path.dirname(os.path.abspath(__file__))  # scripts folder
data_dir = os.path.join(base, "..", "data", "processed")
img_dir = os.path.join(base, "..", "outputs", "images")
frontend_dir = os.path.join(base, "..", "frontend", "public")

# ensure directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(frontend_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "ogdc_cleaned.csv")

# load cleaned data
print("loading cleaned data...")
df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
df.sort_index(inplace=True)

# rename columns to lowercase - DO THIS FIRST
df.rename(columns={
    "Price": "price",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Volume": "volume",
    "Return": "returns",
    "Change %": "changep"
}, inplace=True)

# now check if columns exist and handle missing ones
print(f"columns found: {df.columns.tolist()}")

df.dropna(subset=["price"], inplace=True)
df["returns"] = df["returns"].fillna(0)
df["year"] = df.index.year

# compute technical indicators
log_ret = np.log(df["price"] / df["price"].shift(1)).fillna(0)
df["hv20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
df["sma20"] = df["price"].rolling(20).mean()
df["sma50"] = df["price"].rolling(50).mean()
df["sma200"] = df["price"].rolling(200).mean()
df["bb_upper"] = df["sma20"] + 2 * df["price"].rolling(20).std()
df["bb_lower"] = df["sma20"] - 2 * df["price"].rolling(20).std()

# build price_data.json
print("\nbuilding price_data.json...")
price_cols = ["price", "open", "high", "low", "volume", "returns", "hv20", 
              "sma20", "sma50", "sma200", "bb_upper", "bb_lower"]

# only keep columns that exist
price_cols = [c for c in price_cols if c in df.columns]
price_df = df[price_cols].copy().fillna(0)
price_df.index = price_df.index.strftime("%Y-%m-%d")

price_list = []
for d, row in price_df.iterrows():
    row_dict = {"date": d}
    for k, v in row.items():
        # handle both numpy types and regular floats
        if pd.isna(v):
            row_dict[k] = 0
        else:
            row_dict[k] = round(float(v), 4)
    price_list.append(row_dict)

out_path = os.path.join(frontend_dir, "price_data.json")
with open(out_path, "w") as f:
    json.dump(price_list, f, separators=(",", ":"))
print(f"  saved price_data.json ({len(price_list)} rows)")

# build kpis.json
print("\nbuilding kpis.json...")
latest = df["price"].iloc[-1]
start = df["price"].iloc[0]
total_ret = (latest / start - 1) * 100
mean_ret = df["returns"].mean()
std_ret = df["returns"].std()
sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
max_dd = ((df["price"] / df["price"].cummax()) - 1).min() * 100
hv_now = df["hv20"].iloc[-1]

kpis = {
    "latestPrice": round(float(latest), 2),
    "totalReturn": round(float(total_ret), 2),
    "sharpe": round(float(sharpe), 3),
    "maxDrawdown": round(float(max_dd), 2),
    "currentHV": round(float(hv_now), 2),
    "observations": len(df),
    "meanDailyRet": round(float(mean_ret), 4),
    "stdDailyRet": round(float(std_ret), 4),
    "dateStart": df.index.min().strftime("%Y-%m-%d"),
    "dateEnd": df.index.max().strftime("%Y-%m-%d"),
}

out_path = os.path.join(frontend_dir, "kpis.json")
with open(out_path, "w") as f:
    json.dump(kpis, f, indent=2)
print(f"  saved kpis.json")
for k, v in kpis.items():
    print(f"    {k}: {v}")

# build annual_stats.json
print("\nbuilding annual_stats.json...")
annual = df.groupby("year")["returns"].agg(
    mean="mean", std="std", min="min", max="max", count="count"
).round(4).reset_index()
annual_list = annual.to_dict(orient="records")

out_path = os.path.join(frontend_dir, "annual_stats.json")
with open(out_path, "w") as f:
    json.dump(annual_list, f, indent=2)
print(f"  saved annual_stats.json ({len(annual_list)} years)")

# copy all pngs to frontend/public
print("\ncopying png charts to frontend/public...")
copied = 0

if os.path.exists(img_dir):
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith(".png"):
            src = os.path.join(img_dir, fname)
            dst = os.path.join(frontend_dir, fname)
            shutil.copy2(src, dst)
            copied += 1
    print(f"  copied {copied} png files to frontend/public/")
else:
    print(f"  warning: image directory not found: {img_dir}")

print("\nfrontend data export complete.")
print(f"  all files in: {frontend_dir}")