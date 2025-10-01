import os
import pandas as pd
import streamlit as st
import glob
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
st.markdown("### 📊 Daily PnL Dashboard - Stocks & Options")

# Reduce font size of Streamlit metrics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG PATHS ---
OPTIONS_FOLDER = r"C:\Users\HP\OneDrive\Desktop\Shoonya\trade_app\modules\logs"
STOCKS_FOLDER = r"C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade\logs"

OPTIONS_CSV = os.path.join(OPTIONS_FOLDER, "option_positions.csv")
STOCKS_CSV = os.path.join(STOCKS_FOLDER, "stock_positions.csv")

os.makedirs(OPTIONS_FOLDER, exist_ok=True)
os.makedirs(STOCKS_FOLDER, exist_ok=True)


# --- COMBINE POSITION FILES ---
def combine_position_files(folder_path, output_file):
    try:
        pattern = os.path.join(folder_path, "*_positions.csv")
        all_files = glob.glob(pattern)


        if not all_files:
            # Create an empty template file with expected columns
            empty_df = pd.DataFrame(columns=["Timestamp", "Symbol", "Product", "Strategy", "PnL"])
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            empty_df.to_csv(output_file, index=False)
            return empty_df

        if os.path.exists(output_file):
            os.remove(output_file)

        combined_df = pd.DataFrame()
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path, on_bad_lines="skip")
                if not df.empty:
                    df["SourceFile"] = os.path.basename(file_path)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                st.error(f"Error loading {os.path.basename(file_path)}: {e}")

        if not combined_df.empty:
            combined_df["Timestamp"] = pd.to_datetime(combined_df["Timestamp"], errors="coerce")
            combined_df["Date"] = combined_df["Timestamp"].dt.date
            combined_df = combined_df.dropna(subset=["Timestamp"])
            combined_df = combined_df.sort_values("Timestamp")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_csv(output_file, index=False)

        return combined_df

    except Exception as e:
        st.error(f"Error combining files in {folder_path}: {e}")
        return pd.DataFrame()


# --- LOAD DATA WITH CACHE ---
@st.cache_data
def load_data(_folder_path, _output_file, data_type):
    try:
        if not os.path.exists(_output_file):
            return combine_position_files(_folder_path, _output_file), data_type

        pattern = os.path.join(_folder_path, "*_positions.csv")
        position_files = glob.glob(pattern)

        needs_update = False
        if position_files:
            combined_mtime = os.path.getmtime(_output_file)
            for file_path in position_files:
                if os.path.getmtime(file_path) > combined_mtime:
                    needs_update = True
                    break
        else:
            needs_update = True

        if needs_update:
            return combine_position_files(_folder_path, _output_file), data_type
        else:
            df = pd.read_csv(_output_file, on_bad_lines="skip")
            if df.empty:
                return df, data_type
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            if "Date" not in df.columns:
                df["Date"] = df["Timestamp"].dt.date
            return df, data_type

    except Exception as e:
        st.error(f"Error loading data for {data_type}: {e}")
        return pd.DataFrame(), data_type


# --- DAILY DATA PREP ---
def prepare_daily_data(df, selected_strategies=None):
    if df.empty:
        return pd.DataFrame()

    try:
        df = df.copy()
        if selected_strategies and "Strategy" in df.columns:
            df = df[df["Strategy"].isin(selected_strategies)]

        if "Date" not in df.columns or "PnL" not in df.columns:
            return pd.DataFrame()

        group_cols = ["Date"]
        extra_cols = [col for col in ["Product", "Strategy"] if col in df.columns]
        group_cols.extend(extra_cols)

        daily_data = (
            df.groupby(group_cols, dropna=False)["PnL"]
            .sum()
            .reset_index()
            .sort_values("Date")
        )
        if "DataSource" in df.columns:
            daily_data["DataSource"] = df.groupby(group_cols)["DataSource"].first().values

        return daily_data

    except Exception as e:
        st.error(f"Error preparing daily data: {e}")
        return pd.DataFrame()


# --- PIVOT CREATION ---
def create_pivot_for_chart(daily_data, data_type):
    if daily_data.empty:
        return pd.DataFrame()

    try:
        daily_data_str = daily_data.copy()
        daily_data_str["Date_Str"] = daily_data_str["Date"].astype(str)

        date_order = sorted(daily_data_str["Date"].unique())
        date_str_order = [
            date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            for date in date_order
        ]

        columns = []
        if data_type == "Both" and "DataSource" in daily_data_str.columns:
            columns.append("DataSource")
        for col in ["Product", "Strategy"]:
            if col in daily_data_str.columns:
                columns.append(col)

        if not columns:
            pivot_data = daily_data_str.set_index("Date_Str")["PnL"].to_frame()
        else:
            pivot_data = daily_data_str.pivot_table(
                index="Date_Str",
                columns=columns,
                values="PnL",
                aggfunc="sum",
                fill_value=0,
            )
            if isinstance(pivot_data.columns, pd.MultiIndex):
                pivot_data.columns = [" | ".join(map(str, col)) for col in pivot_data.columns]

        pivot_data = pivot_data.reindex(date_str_order).fillna(0)
        return pivot_data

    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        return pd.DataFrame()


# --- SUMMARY TABLE ---
def create_summary_table(daily_data, data_type):
    if daily_data.empty:
        return pd.DataFrame()

    try:
        if data_type == "Both" and "DataSource" in daily_data.columns:
            daily_summary = daily_data.groupby(["Date", "DataSource"])["PnL"].sum().reset_index()
            pivot_summary = daily_summary.pivot_table(
                index="Date", columns="DataSource", values="PnL", aggfunc="sum", fill_value=0
            ).reset_index()
            for source in ["Options", "Stocks"]:
                if source not in pivot_summary.columns:
                    pivot_summary[source] = 0
            pivot_summary["Total PnL"] = pivot_summary.sum(axis=1, numeric_only=True)
            pivot_summary["Cumulative PnL"] = pivot_summary["Total PnL"].cumsum()
            return pivot_summary
        else:
            daily_summary = daily_data.groupby("Date")["PnL"].sum().reset_index()
            daily_summary["Cumulative PnL"] = daily_summary["PnL"].cumsum()
            return daily_summary

    except Exception as e:
        st.error(f"Error creating summary table: {e}")
        return pd.DataFrame()


# --- MAIN APP ---
data_type = st.radio("Select Data Type:", options=["Options", "Stocks", "Both"], horizontal=True)
force_refresh = st.checkbox("Force Refresh All Data")

try:
    if data_type == "Options":
        if force_refresh and os.path.exists(OPTIONS_CSV):
            os.remove(OPTIONS_CSV)
        df, source = load_data(OPTIONS_FOLDER, OPTIONS_CSV, "Options")

    elif data_type == "Stocks":
        if force_refresh and os.path.exists(STOCKS_CSV):
            os.remove(STOCKS_CSV)
        df, source = load_data(STOCKS_FOLDER, STOCKS_CSV, "Stocks")

    else:
        if force_refresh:
            if os.path.exists(OPTIONS_CSV):
                os.remove(OPTIONS_CSV)
            if os.path.exists(STOCKS_CSV):
                os.remove(STOCKS_CSV)
        options_df, _ = load_data(OPTIONS_FOLDER, OPTIONS_CSV, "Options")
        stocks_df, _ = load_data(STOCKS_FOLDER, STOCKS_CSV, "Stocks")
        if not options_df.empty and not stocks_df.empty:
            options_df["DataSource"] = "Options"
            stocks_df["DataSource"] = "Stocks"
            df = pd.concat([options_df, stocks_df], ignore_index=True)
        elif not options_df.empty:
            df = options_df
            df["DataSource"] = "Options"
        elif not stocks_df.empty:
            df = stocks_df
            df["DataSource"] = "Stocks"
        else:
            df = pd.DataFrame()
except Exception as e:
    st.error(f"Error during data loading: {e}")
    df = pd.DataFrame()

if df.empty:
    st.info("📭 No data available. Please check if position files exist.")
    st.stop()

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Type", data_type)
with col2:
    if not df.empty and "Date" in df.columns:
        try:
            dates = pd.to_datetime(df["Date"]).dt.date
            date_range = f"{dates.min()} to {dates.max()}"
        except:
            date_range = "Invalid dates"
        st.metric("Data Period", date_range)
    else:
        st.metric("Data Period", "N/A")
with col3:
    st.metric("Total Records", f"{len(df):,}")
with col4:
    pnl = df["PnL"].sum() if "PnL" in df.columns else 0
    st.metric("Total PnL", f"₹{pnl:,.2f}")

# --- FILTERS ---
st.markdown("#### 🔍 Filters")
available_strategies = []
if "Strategy" in df.columns:
    available_strategies = sorted(df["Strategy"].dropna().unique())
selected_strategies = st.multiselect("Select Strategies:", options=available_strategies, default=available_strategies)

if data_type == "Both" and "DataSource" in df.columns:
    data_sources = sorted(df["DataSource"].unique())
    selected_sources = st.multiselect("Select Data Sources:", options=data_sources, default=data_sources)
    df = df[df["DataSource"].isin(selected_sources)]

# --- PREPARE DATA ---
daily_data = prepare_daily_data(df, selected_strategies)

if not daily_data.empty:
    st.markdown("#### 📈 Daily PnL Trend")
    pivot_data = create_pivot_for_chart(daily_data, data_type)
    if not pivot_data.empty:
        tab1, tab2 = st.tabs(["Line Chart", "Bar Chart"])
        with tab1:
            st.line_chart(pivot_data, height=400)
        with tab2:
            st.bar_chart(pivot_data, height=400)

    st.markdown("#### 📋 Daily Summary")
    summary_table = create_summary_table(daily_data, data_type)
    if not summary_table.empty:
        format_dict = {
            col: "₹{:,.2f}"
            for col in summary_table.columns
            if col != "Date" and pd.api.types.is_numeric_dtype(summary_table[col])
        }
        st.dataframe(summary_table.style.format(format_dict), width="stretch")
else:
    st.info("No daily data available for the selected filters.")

# --- REFRESH BUTTON ---
if st.button("🔄 Refresh Data", width="stretch"):
    st.cache_data.clear()
    st.rerun()

# --- RAW DATA ---
with st.expander("🔍 View Raw Data"):
    if not df.empty:
        display_cols = [
            col
            for col in ["Date", "DataSource", "Symbol", "Product", "Strategy", "PnL", "SourceFile"]
            if col in df.columns
        ]
        st.dataframe(df[display_cols], width="stretch")
