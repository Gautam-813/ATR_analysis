
import pandas as pd
import numpy as np

def process_data(file_path):
    print("Loading data...")
    df = pd.read_parquet(file_path)
    
    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'], format='%Y.%m.%d %H:%M:%S')
    df = df.sort_values('Time')
    
    # Calculate True Range (TR)
    print("Calculating True Range...")
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                np.maximum(abs(df['High'] - df['PrevClose']), 
                           abs(df['Low'] - df['PrevClose'])))
    
    # ATR (14-period standard for minute)
    # We might want a larger ATR for smoother results, let's use 14 for minute and aggregate.
    df['ATR_1m'] = df['TR'].rolling(window=14).mean()
    
    # Extract time features
    df['Hour'] = df['Time'].dt.hour
    df['Date'] = df['Time'].dt.date
    df['DayOfWeek'] = df['Time'].dt.day_name()
    df['Month'] = df['Time'].dt.to_period('M')
    
    # Session Definition (GMT approximation - adjust if broker time is different)
    # Sydney: 22-07, Tokyo: 00-09, London: 08-17, New York: 13-22
    def get_session(hour):
        if 8 <= hour < 13: return 'London'
        if 13 <= hour < 17: return 'London/NY Overlap'
        if 17 <= hour < 22: return 'New York'
        if 22 <= hour or hour < 0: return 'Sydney' # Minimal
        if 0 <= hour < 8: return 'Tokyo/Sydney'
        return 'Other'

    df['Session'] = df['Hour'].apply(get_session)
    
    # Aggregate Metrics (Using Mean for ATR)
    print("Aggregating Daily Metrics...")
    daily_atr = df.groupby('Date').agg({
        'TR': 'mean', # Average ATR per minute for that day
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).rename(columns={'TR': 'Avg_ATR'})
    
    daily_atr = daily_atr.reset_index()
    daily_atr['Date'] = pd.to_datetime(daily_atr['Date'])
    daily_atr['Year'] = daily_atr['Date'].dt.year
    daily_atr['Month_Num'] = daily_atr['Date'].dt.month
    daily_atr['Month_Name'] = daily_atr['Date'].dt.month_name()
    daily_atr['Month_Year'] = daily_atr['Date'].dt.to_period('M').astype(str)
    
    # Mark special days
    daily_atr['Is_Week_Start'] = daily_atr['Date'].dt.dayofweek == 0 # Monday
    daily_atr['Is_Week_End'] = daily_atr['Date'].dt.dayofweek == 4   # Friday
    
    # Month First/Last Day
    month_bounds = daily_atr.groupby('Month_Year')['Date'].agg(['min', 'max']).reset_index()
    daily_atr = daily_atr.merge(month_bounds, on='Month_Year')
    daily_atr['Is_Month_Start'] = daily_atr['Date'] == daily_atr['min']
    daily_atr['Is_Month_End'] = daily_atr['Date'] == daily_atr['max']
    
    # Session-wise analysis
    print("Aggregating Session Metrics...")
    session_atr = df.groupby(['Date', 'Session'])['TR'].mean().unstack()
    
    return df, daily_atr, session_atr

# Test processing
if __name__ == "__main__":
    path = r"d:\date-wise\03-03-2026\atr_analysis_data\XAUUSD_M1_Data.parquet"
    df, daily, session = process_data(path)
    print("Daily Head:")
    print(daily.head())
    print("\nSession Head:")
    print(session.head())
    
    # Save processed data for faster streamlit loading
    daily.to_csv(r"d:\date-wise\03-03-2026\atr_analysis_data\processed_daily.csv", index=False)
    session.to_csv(r"d:\date-wise\03-03-2026\atr_analysis_data\processed_sessions.csv")
    print("Processed data saved.")
