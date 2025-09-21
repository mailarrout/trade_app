import os
import pandas as pd
import streamlit as st

# --- CONFIG ---
FOLDER = r"C:\Users\HP\OneDrive\Desktop\Shoonya\trade_app\modules\logs"
CSV_FILE = os.path.join(FOLDER, "all_positions_combined.csv")

# --- LOAD DATA ---
@st.cache_data
def load_data(file_path: str):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_csv(file_path, on_bad_lines="skip")
    # parse timestamp with dd/mm/yyyy format
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    return df

df = load_data(CSV_FILE)

st.title("📊 PnL Analysis Dashboard")

if df.empty:
    st.error(f"No valid data found. Make sure {CSV_FILE} exists.")
    st.stop()

# --- FILTERING OPTIONS ---
timeframe = st.radio("Choose timeframe", ["Daily", "Weekly", "Monthly"])

products = st.multiselect(
    "Filter by Product",
    sorted(df["Product"].dropna().unique()),
    default=None,
)

strategies = st.multiselect(
    "Filter by Strategy",
    sorted(df["Strategy"].dropna().unique()),
    default=None,
)

filtered = df.copy()
if products:
    filtered = filtered[filtered["Product"].isin(products)]
if strategies:
    filtered = filtered[filtered["Strategy"].isin(strategies)]

# --- GROUPING ---
if timeframe == "Daily":
    filtered["Period"] = filtered["Timestamp"].dt.date

elif timeframe == "Weekly":
    period = filtered["Timestamp"].dt.to_period("W")
    filtered["Period"] = period.apply(
        lambda r: r.start_time.date() if pd.notna(r) else pd.NaT
    )

else:  # Monthly
    period = filtered["Timestamp"].dt.to_period("M")
    filtered["Period"] = period.apply(
        lambda r: r.start_time.date() if pd.notna(r) else pd.NaT
    )

summary = (
    filtered.groupby(["Period", "Product", "Strategy"])["PnL"]
    .sum()
    .reset_index()
    .sort_values("Period")
)

# Ensure Period is pure date
summary["Period"] = pd.to_datetime(summary["Period"]).dt.date

# --- SHOW RESULTS ---
st.subheader("PnL Summary")
st.dataframe(summary)

# --- LINE CHART ---
st.subheader("PnL Trend (Line Chart)")
if not summary.empty:
    pivot = summary.pivot_table(
        index="Period",
        columns=["Product", "Strategy"],
        values="PnL",
        aggfunc="sum",
    )

    # flatten MultiIndex columns
    pivot.columns = [f"{p} | {s}" for p, s in pivot.columns]
    pivot = pivot.sort_index()

    # convert index to date
    pivot.index = pd.to_datetime(pivot.index).date

    st.line_chart(pivot)
else:
    st.info("No data available for the selected filters.")

# --- BAR CHART ---
st.subheader("PnL by Period (Bar Chart)")
if not summary.empty:
    bar_pivot = summary.pivot_table(
        index="Period",
        columns=["Product", "Strategy"],
        values="PnL",
        aggfunc="sum",
    )
    bar_pivot.columns = [f"{p} | {s}" for p, s in bar_pivot.columns]
    bar_pivot = bar_pivot.sort_index()

    # convert index to date
    bar_pivot.index = pd.to_datetime(bar_pivot.index).date

    st.bar_chart(bar_pivot)
else:
    st.info("No data available for the selected filters.")
