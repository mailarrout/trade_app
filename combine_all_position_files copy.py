import os
import pandas as pd
import streamlit as st
import glob

# Set page configuration to use wide layout
st.set_page_config(layout="wide")

# --- CONFIG ---
FOLDER = r"C:\Users\HP\OneDrive\Desktop\Shoonya\trade_app\modules\logs"
CSV_FILE = os.path.join(FOLDER, "all_positions_combined.csv")

# --- COMBINE ONLY _positions.csv FILES ---
def combine_position_files():
    """Combine only _positions.csv files into one master file"""
    pattern = os.path.join(FOLDER, "*_positions.csv")
    all_files = glob.glob(pattern)
    
    if not all_files:
        st.error("No *_positions.csv files found!")
        return pd.DataFrame()
    
    combined_df = pd.DataFrame()
    
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip")
            df['SourceFile'] = os.path.basename(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            st.error(f"Error loading {os.path.basename(file_path)}: {e}")
    
    if not combined_df.empty:
        # Parse timestamp and extract only the date part
        combined_df['Timestamp'] = pd.to_datetime(
            combined_df['Timestamp'], 
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        
        # Convert to date only (remove time component)
        combined_df['Date'] = combined_df['Timestamp'].dt.date
        
        combined_df = combined_df.dropna(subset=['Timestamp'])
        combined_df = combined_df.sort_values('Timestamp')
        combined_df.to_csv(CSV_FILE, index=False)
        st.success(f"✅ Combined {len(all_files)} files → {len(combined_df)} records")
    
    return combined_df

# --- LOAD DATA ---
@st.cache_data
def load_data():
    """Load existing combined file or create new one"""
    if not os.path.exists(CSV_FILE):
        return combine_position_files()
    
    combined_mtime = os.path.getmtime(CSV_FILE)
    pattern = os.path.join(FOLDER, "*_positions.csv")
    position_files = glob.glob(pattern)
    
    needs_update = False
    for file_path in position_files:
        if os.path.getmtime(file_path) > combined_mtime:
            needs_update = True
            break
    
    if needs_update:
        return combine_position_files()
    else:
        df = pd.read_csv(CSV_FILE, on_bad_lines="skip")
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'], 
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        # Ensure we have the Date column
        if 'Date' not in df.columns:
            df['Date'] = df['Timestamp'].dt.date
        return df

# --- PREPARE DAILY DATA ---
def prepare_daily_data(df, selected_strategies=None):
    """Prepare daily PnL data using the Date column (without time)"""
    df = df.copy()
    
    # Apply strategy filter if provided
    if selected_strategies:
        df = df[df['Strategy'].isin(selected_strategies)]
    
    # Use the Date column that already has no time component
    daily_data = (
        df.groupby(['Date', 'Product', 'Strategy'])['PnL']
        .sum()
        .reset_index()
        .sort_values('Date')
    )
    
    return daily_data

# --- MAIN APP ---
st.title("📊 Daily PnL Dashboard")

# Load data
df = load_data()

if df.empty:
    st.error("No data available. Please check your *_positions.csv files.")
    st.stop()

# Summary info in columns for better use of width
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data Period", f"{df['Date'].min()} to {df['Date'].max()}")
with col2:
    st.metric("Total Records", f"{len(df):,}")
with col3:
    st.metric("Total PnL", f"₹{df['PnL'].sum():,.2f}")

# --- STRATEGY FILTER ---
st.subheader("🔍 Filters")

# Strategy filter
available_strategies = sorted(df['Strategy'].dropna().unique())
selected_strategies = st.multiselect(
    "Select Strategies:",
    options=available_strategies,
    default=available_strategies,
    help="Select one or more strategies to filter the data"
)

# Prepare daily data with filters
daily_data = prepare_daily_data(df, selected_strategies)

# --- DISPLAY CHARTS ---
st.subheader("📈 Daily PnL Trend")

if not daily_data.empty:
    # Simple approach: Use string dates to avoid time display
    daily_data_str = daily_data.copy()
    daily_data_str['Date_Str'] = daily_data_str['Date'].astype(str)
    
    # Create pivot table with string dates
    pivot_data = daily_data_str.pivot_table(
        index="Date_Str",
        columns=["Product", "Strategy"],
        values="PnL",
        aggfunc="sum",
        fill_value=0
    )
    
    # Flatten column names
    if isinstance(pivot_data.columns, pd.MultiIndex):
        pivot_data.columns = [f"{p} | {s}" for p, s in pivot_data.columns]
    
    # Sort by the actual date values (not string sorting)
    date_order = sorted(daily_data_str['Date'].unique())
    date_str_order = [date.strftime('%Y-%m-%d') for date in date_order]
    pivot_data = pivot_data.reindex(date_str_order)

    # Display charts with string dates to avoid time display
    st.write("**Line Chart**")
    st.line_chart(pivot_data, height=400)
    
    st.write("**Bar Chart**")
    st.bar_chart(pivot_data, height=400)

    # --- SUMMARY TABLE ---
    st.subheader("📋 Daily Summary")
    
    daily_summary = daily_data.groupby('Date')['PnL'].sum().reset_index()
    daily_summary['Cumulative PnL'] = daily_summary['PnL'].cumsum()
    
    st.dataframe(daily_summary.style.format({
        'PnL': '₹{:,.2f}',
        'Cumulative PnL': '₹{:,.2f}'
    }), width='stretch')

else:
    st.info("No data available for the selected strategies.")

# Refresh button centered
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

# --- SHOW RAW DATA ---
with st.expander("🔍 View Raw Data"):
    st.dataframe(df[['Date', 'Symbol', 'Product', 'Strategy', 'PnL']], width='stretch')