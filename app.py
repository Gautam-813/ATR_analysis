
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import io
import contextlib
import pandas_ta as ta
import quantstats as qs
from scipy import stats
import hashlib
from statsmodels.tsa.stattools import ccf
import datetime
import math
try:
    from groq import Groq
except ImportError:
    Groq = None

import sklearn
import sklearn.cluster
import sklearn.preprocessing
import statsmodels.api as sm
from arch import arch_model
import seaborn as sns
import matplotlib.pyplot as plt
import pypfopt
import xlsxwriter

st.set_page_config(page_title="XAUUSD ATR Analysis Dashboard", layout="wide")

# Theme / Styling (Black Text, White Background)
st.markdown("""
    <style>
    .main { background-color: #ffffff; color: #000000; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    [data-testid="stSidebar"] { background-color: #f1f3f5; }
    h1, h2, h3, p, span, label { color: #000000 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f8f9fa; border-radius: 4px; padding: 10px; }
    
    /* Weekly Block Styling */
    .week-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 20px;
    }
    .week-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 15px;
        width: 300px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .week-header {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.1em;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
    }
    .day-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 5px;
    }
    .day-box {
        aspect-ratio: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 500;
        cursor: help;
    }
    .day-label {
        font-size: 0.6em;
        opacity: 0.7;
        margin-bottom: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 XAUUSD Multi-Timeframe ATR Analysis")

@st.cache_data
def load_asset_data(symbol):
    import os
    # Mapping symbols to filenames
    filenames = {
        "GOLD": "XAUUSD_M1_Data.parquet",
        "DXY": "DXY_M1_Data.parquet",
        "EURUSD": "EURUSD_M1_Data.parquet"
    }
    fname = filenames.get(symbol)
    if not fname: return None
    
    possible_paths = [
        fname,
        f"atr_analysis_data/{fname}",
        f"d:\\date-wise\\03-03-2026\\atr_analysis_data\\{fname}"
    ]
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if file_path is None: return None
    
    df = pd.read_parquet(file_path)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    df['Time'] = pd.to_datetime(df['Time'])
    return df

@st.cache_data
def get_available_models(gemini_key, groq_key=None):
    models = []
    # 1. Gemini Models
    try:
        genai.configure(api_key=gemini_key)
        gemini_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        models.extend([m for m in gemini_models if "gemini" in m.lower()])
    except Exception:
        models.extend(["models/gemini-1.5-flash", "models/gemini-2.0-flash"])
    
    # 2. Groq Models (Manual list since their API is different)
    if groq_key and Groq:
        models.extend([
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768",
            "groq/deepseek-r1-distill-llama-70b"
        ])
    
    models.sort()
    return models

@st.cache_data
def resample_and_calculate(df, timeframe, atr_period):
    resampled = df.resample(timeframe, on='Time').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    resampled['PrevClose'] = resampled['Close'].shift(1)
    resampled['TR'] = np.maximum(resampled['High'] - resampled['Low'], 
                        np.maximum(abs(resampled['High'] - resampled['PrevClose']), 
                                   abs(resampled['Low'] - resampled['PrevClose'])))
    resampled['ATR'] = resampled['TR'].rolling(window=atr_period).mean()
    resampled = resampled.reset_index()
    resampled['Hour'] = resampled['Time'].dt.hour
    resampled['Date'] = resampled['Time'].dt.date
    resampled['Year'] = resampled['Time'].dt.year
    resampled['Month_Year'] = resampled['Time'].dt.to_period('M').astype(str)
    resampled['Month_Num'] = resampled['Time'].dt.month
    resampled['Month_Name'] = resampled['Time'].dt.month_name()
    resampled['Week_Number'] = resampled['Time'].dt.isocalendar().week
    
    def get_session(hour):
        if 8 <= hour < 13: return 'London'
        if 13 <= hour < 17: return 'London/NY Overlap'
        if 17 <= hour < 22: return 'New York'
        if 22 <= hour or hour < 0: return 'Sydney'
        if 0 <= hour < 8: return 'Tokyo/Sydney'
        return 'Other'
    resampled['Session'] = resampled['Hour'].apply(get_session)
    
    daily_metrics = resampled.groupby('Date').agg({'ATR': 'mean', 'Year': 'first', 'Month_Year': 'first', 'Month_Num': 'first', 'Month_Name': 'first', 'Week_Number': 'first'}).rename(columns={'ATR': 'Avg_ATR'}).reset_index()
    daily_metrics['Date'] = pd.to_datetime(daily_metrics['Date'])
    daily_metrics['Day_Name'] = daily_metrics['Date'].dt.day_name()
    daily_metrics['Is_Week_Start'] = daily_metrics['Date'].dt.dayofweek == 0
    daily_metrics['Is_Week_End'] = daily_metrics['Date'].dt.dayofweek == 4
    mb = daily_metrics.groupby('Month_Year')['Date'].agg(['min', 'max']).reset_index()
    daily_metrics = daily_metrics.merge(mb, on='Month_Year')
    daily_metrics['Is_Month_Start'] = daily_metrics['Date'] == daily_metrics['min']
    daily_metrics['Is_Month_End'] = daily_metrics['Date'] == daily_metrics['max']
    
    # --- Session-specific Calculation (Always uses 1H resolution for depth) ---
    res_session = df.resample('1H', on='Time').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    res_session['PrevClose'] = res_session['Close'].shift(1)
    res_session['TR'] = np.maximum(res_session['High'] - res_session['Low'], 
                         np.maximum(abs(res_session['High'] - res_session['PrevClose']), 
                                    abs(res_session['Low'] - res_session['PrevClose'])))
    res_session['ATR'] = res_session['TR'].rolling(window=atr_period).mean()
    res_session = res_session.reset_index()
    res_session['Hour'] = res_session['Time'].dt.hour
    res_session['Date'] = res_session['Time'].dt.date
    res_session['Session'] = res_session['Hour'].apply(get_session)
    
    session_metrics = res_session.groupby(['Date', 'Session'])['ATR'].mean().unstack().reset_index()
    
    # Final downcast of results
    daily_metrics['Avg_ATR'] = daily_metrics['Avg_ATR'].astype('float32')
    
    return daily_metrics, session_metrics

# Sidebar
st.sidebar.header("⚙️ Dashboard Settings")
tf_opt = {"1 Min": "1min", "5 Min": "5min", "15 Min": "15min", "1 Hour": "1H", "4 Hour": "4H", "Daily": "1D", "Weekly": "W"}
sel_tf = st.sidebar.selectbox("Analysis Timeframe", options=list(tf_opt.keys()))
atr_p = st.sidebar.number_input("ATR Period (Rolling)", 1, 200, 14)

st.sidebar.divider()

# API Key Manager (BYOK - Bring Your Own Key)
with st.sidebar.expander("🔑 AI API Keys", expanded=False):
    st.caption("Custom keys are stored in memory and wiped on refresh.")
    u_gemini_key = st.text_input("Gemini API Key", type="password")
    u_groq_key = st.text_input("Groq API Key", type="password")
    
    # Resolve keys (Priority: User Input > Secrets)
    final_gemini_key = u_gemini_key if u_gemini_key else st.secrets.get("GEMINI_API_KEY", "")
    final_groq_key = u_groq_key if u_groq_key else st.secrets.get("GROQ_API_KEY", "")
    
    if final_gemini_key: st.success("Gemini: Connected ✅")
    if final_groq_key: st.success("Groq: Connected ✅")

# Load Primary and Secondary Data
raw = load_asset_data("GOLD")
if raw is None:
    st.error("Primary Gold data not found!")
    st.stop()

# Support Assets for AI Analysis
raw_dxy = load_asset_data("DXY")
raw_eur = load_asset_data("EURUSD")

# Standard Calculations for Gold (Main Dashboard)
daily, sessions = resample_and_calculate(raw, tf_opt[sel_tf], atr_p)

years = sorted(daily['Year'].unique())
sel_years = st.sidebar.multiselect("Select Years", years, default=years)
filtered_daily = daily[daily['Year'].isin(sel_years)]

if not filtered_daily.empty:
    min_d, max_d = filtered_daily['Date'].min().date(), filtered_daily['Date'].max().date()
    dr = st.sidebar.date_input("Range", (min_d, max_d), min_d, max_d)
    if isinstance(dr, tuple) and len(dr) == 2:
        filtered_daily = daily[(daily['Date'].dt.date >= dr[0]) & (daily['Date'].dt.date <= dr[1])]
        sessions_f = sessions[(pd.to_datetime(sessions['Date']).dt.date >= dr[0]) & (pd.to_datetime(sessions['Date']).dt.date <= dr[1])]
    else:
        sessions_f = sessions[pd.to_datetime(sessions['Date']).dt.year.isin(sel_years)]
else:
    sessions_f = sessions.head(0)

# Metrics
c1, c2, c3, c4 = st.columns(4)
overall_avg = filtered_daily['Avg_ATR'].mean()
c1.metric("Overall Avg ATR", f"{overall_avg:.4f}")
c2.metric("Days", len(filtered_daily))
s_cols = [c for c in sessions.columns if c != 'Date']
if not sessions_f.empty:
    s_avg = sessions_f[s_cols].mean()
    c3.metric("Pick Session", s_avg.idxmax())
    c4.metric("Session ATR", f"{s_avg.max():.4f}")

# Initialize Session States
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

st.divider()

t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs(["🌐 Sessions", "📅 Week Start/End", "🗓️ Month Start/End", "🧱 Weekly Blocks", "🗓️ Monthly Blocks", "📈 Trends", "📊 Quant Stats", "📅 Sequential Analysis", "🕯️ Directional ATR", "🤖 AI Assistant", "🔗 Lead-Lag Engine"])

with t1:
    if not sessions_f.empty:
        st.subheader("Session Performance Summary")
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            st.markdown("**Avg ATR per Session**")
            sd = s_avg.reset_index(); sd.columns = ['Session', 'ATR']
            st.plotly_chart(px.bar(sd, x='Session', y='ATR', color='Session', template='plotly_white'), use_container_width=True)
        
        with col_s2:
            st.markdown("**Volatility Composition (by Day of Week)**")
            # Calculate avg session ATR per Day of Week
            # Melding data for plotly
            sess_melt = sessions_f.copy()
            sess_melt['Date'] = pd.to_datetime(sess_melt['Date'])
            sess_melt['DayOfWeek'] = sess_melt['Date'].dt.day_name()
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
            # Melting for stacked bar
            df_stack = sess_melt.melt(id_vars=['Date', 'DayOfWeek'], value_vars=s_cols, var_name='Session', value_name='ATR')
            df_stack_avg = df_stack.groupby(['DayOfWeek', 'Session'])['ATR'].mean().reset_index()
            
            fig_stack = px.bar(df_stack_avg, x='DayOfWeek', y='ATR', color='Session', 
                               category_orders={'DayOfWeek': day_order},
                               template='plotly_white', barmode='stack')
            st.plotly_chart(fig_stack, use_container_width=True)
            
        st.divider()
        st.subheader("ATR Session Contribution (Daily Timeline)")
        st.markdown("This chart shows how different sessions contributed to the daily volatility over time. Use the slider below to zoom in on specific periods.")
        fig_timeline = px.bar(df_stack, x='Date', y='ATR', color='Session', template='plotly_white')
        fig_timeline.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_timeline, use_container_width=True)

with t2:
    st.subheader("Week Start (Mon) vs End (Fri)")
    ws = filtered_daily[filtered_daily['Is_Week_Start']]['Avg_ATR'].mean()
    we = filtered_daily[filtered_daily['Is_Week_End']]['Avg_ATR'].mean()
    st.plotly_chart(go.Figure([go.Bar(x=['Mon', 'Fri'], y=[ws, we], marker_color=['#4dabf7', '#ff8787'])]).update_layout(template='plotly_white'))

with t3:
    st.subheader("Month Start vs Month End")
    ms = filtered_daily[filtered_daily['Is_Month_Start']]['Avg_ATR'].mean()
    me = filtered_daily[filtered_daily['Is_Month_End']]['Avg_ATR'].mean()
    st.plotly_chart(go.Figure([go.Bar(x=['Start', 'End'], y=[ms, me], marker_color=['#20c997', '#fab005'])]).update_layout(template='plotly_white'))

with t4:
    st.subheader("Weekly ATR Blocks (Daily Average)")
    st.markdown("Each block shows the **Day name** and the **Average ATR** for that day. Darker colors = Higher volatility.")
    
    if not filtered_daily.empty:
        base_avg = filtered_daily['Avg_ATR'].mean()
        day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
        
        for (y, m), m_grp in filtered_daily.groupby(['Year', 'Month_Name'], sort=False):
            st.markdown(f"#### {m} {y}")
            html = '<div class="week-container">'
            
            for w, w_data in m_grp.groupby('Week_Number'):
                html += f'<div class="week-card"><div class="week-header">Week {w}</div><div class="day-grid">'
                
                for d_idx in range(5):
                    d_label = day_names[d_idx]
                    d_rec = w_data[w_data['Date'].dt.dayofweek == d_idx]
                    
                    if not d_rec.empty:
                        val = d_rec['Avg_ATR'].values[0]
                        d_date = pd.to_datetime(d_rec['Date'].values[0]).strftime('%m/%d')
                        is_start = d_rec['Is_Month_Start'].values[0]
                        is_end = d_rec['Is_Month_End'].values[0]
                        
                        ratio = min(max(val / (base_avg * 1.5), 0.1), 1.0)
                        bg = f"rgba(0,128,128,{ratio})"
                        txt = "white" if ratio > 0.5 else "black"
                        
                        # Highlighting logic
                        border_style = "border: 1px solid transparent;"
                        if is_start: border_style = "border: 3px solid #FFD700; box-shadow: 0 0 8px rgba(255, 215, 0, 0.5);"
                        elif is_end: border_style = "border: 3px solid #FF4500; box-shadow: 0 0 8px rgba(255, 69, 0, 0.5);"
                        
                        html += f'''
                        <div class="day-box" style="background-color:{bg}; color:{txt}; {border_style} cursor:pointer;" title="{d_date} | ATR: {val:.4f}">
                            <div style="font-size:0.8em; font-weight:bold; margin-bottom:2px;">{d_label}{' (S)' if is_start else ''}{' (E)' if is_end else ''}</div>
                            <div style="font-size:1em;">{val:.2f}</div>
                        </div>'''
                    else:
                        html += f'''
                        <div class="day-box" style="background-color:#f1f3f5; color:#ced4da; border: 1px solid transparent;">
                            <div style="font-size:0.8em; margin-bottom:2px;">{d_label}</div>
                            <div style="font-size:1em;">-</div>
                        </div>'''
                
                html += '</div></div>'
            st.markdown(html + '</div>', unsafe_allow_html=True)
            st.divider()

with t5:
    st.subheader("Monthly ATR Blocks")
    if not filtered_daily.empty:
        base_avg = filtered_daily['Avg_ATR'].mean()
        mm = filtered_daily.groupby(['Year', 'Month_Name', 'Month_Num']).agg({'Avg_ATR': 'mean'}).reset_index().sort_values(['Year', 'Month_Num'])
        for yr in sorted(mm['Year'].unique()):
            st.markdown(f"#### {yr}")
            yr_data = mm[mm['Year'] == yr]
            html = '<div class="week-container">'
            m_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            for m_idx in range(1, 13):
                rec = yr_data[yr_data['Month_Num'] == m_idx]
                if not rec.empty:
                    val = rec['Avg_ATR'].values[0]
                    ratio = min(max(val / (base_avg * 1.5), 0.1), 1.0)
                    m_label = m_names[m_idx]
                    html += f'''
                    <div class="week-card" style="width:80px;padding:5px">
                        <div class="week-header" style="font-size:0.7em;text-align:center">{m_label}</div>
                        <div class="day-box" style="background-color:rgba(255,140,0,{ratio}); color:{"white" if ratio>0.5 else "black"}; height:40px;" title="{m_label} {yr} | Avg ATR: {val:.4f}">
                            {val:.2f}
                        </div>
                    </div>'''
                else:
                    html += f'''
                    <div class="week-card" style="width:80px;padding:5px;opacity:0.3">
                        <div class="week-header" style="font-size:0.7em;text-align:center">{m_names[m_idx]}</div>
                        <div class="day-box" style="height:40px;">-</div>
                    </div>'''
            st.markdown(html + '</div>', unsafe_allow_html=True); st.divider()
    else:
        st.info("No data available for the selected range.")

with t6:
    st.subheader("📈 Volatility Trends & Health")
    st.markdown("Professional quant metrics to identify the current market regime.")
    
    if not filtered_daily.empty:
        avg_val = filtered_daily['Avg_ATR'].mean()
        std_val = filtered_daily['Avg_ATR'].std()
        curr_val = filtered_daily['Avg_ATR'].iloc[-1]
        
        # Calculate Z-Score
        z_score = (curr_val - avg_val) / std_val if std_val > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean ATR", f"{avg_val:.4f}")
        c2.metric("Current ATR (End of Period)", f"{curr_val:.4f}", f"{z_score:.2f} σ")
        
        # Determine Regime
        regime = "Normal"
        regime_color = "#333"
        if z_score > 1.5: regime, regime_color = "🔥 High Volatility", "#e03131"
        elif z_score < -1.5: regime, regime_color = "❄️ Low Volatility", "#1971c2"
        
        c3.markdown(f'''
            <div style="background-color:{regime_color}; color:white; padding:10px; border-radius:8px; text-align:center;">
                <div style="font-size:0.8em; opacity:0.8;">Market Regime</div>
                <div style="font-size:1.2em; font-weight:bold;">{regime}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("#### ATR Distribution (Frequency)")
        fig_dist = px.histogram(filtered_daily, x='Avg_ATR', 
                               nbins=50, template='plotly_white',
                               title="How often certain ATR values occur")
        fig_dist.update_traces(marker_color='#34495e')
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No data available.")

with t7:
    st.subheader("📊 Quant Stats (Seasonality Analysis)")
    st.markdown("Comparing volatility cycles across Days, Weeks, and Months.")
    
    if not filtered_daily.empty:
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("#### 📅 Day of the Week Stats")
            d_season = filtered_daily.groupby('Day_Name')['Avg_ATR'].mean().reset_index()
            d_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            fig_d = px.bar(d_season, x='Day_Name', y='Avg_ATR', 
                           category_orders={'Day_Name': d_order},
                           color='Avg_ATR', color_continuous_scale='Blues',
                           template='plotly_white')
            st.plotly_chart(fig_d, use_container_width=True)
        
        with col_q2:
            st.markdown("#### 🗓️ Monthly Seasonality")
            m_season = filtered_daily.groupby('Month_Name')['Avg_ATR'].mean().reset_index()
            m_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            fig_m = px.bar(m_season, x='Month_Name', y='Avg_ATR', 
                           category_orders={'Month_Name': m_order},
                           color='Avg_ATR', color_continuous_scale='Viridis',
                           template='plotly_white')
            st.plotly_chart(fig_m, use_container_width=True)
            
        st.divider()
        st.markdown("#### 📉 Weekly Seasonality (Week 1-52)")
        w_season = filtered_daily.groupby('Week_Number')['Avg_ATR'].mean().reset_index()
        fig_w = px.bar(w_season, x='Week_Number', y='Avg_ATR', 
                       color='Avg_ATR', color_continuous_scale='Cividis',
                       template='plotly_white')
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("No data available.")

with t8:
    st.subheader("📅 Sequential ATR Timelines")
    st.markdown("Detailed breakdown of volatility across the entire history.")
    
    if not filtered_daily.empty:
        # 1. Main Sequential DAILY Timeline (Darker & High Contrast)
        st.markdown("#### 📊 Full History: Daily ATR Timeline")
        st.markdown("This chart allows you to see the big picture. Use the slider to zoom.")
        
        fig_daily_seq = px.bar(filtered_daily, x='Date', y='Avg_ATR', 
                               template='plotly_white', title="Every Trading Day")
        fig_daily_seq.update_layout(
            template='plotly_white', 
            title="Every Trading Day",
            xaxis_type='category', # Removes empty gaps (weekends)
            margin=dict(l=0, r=0, t=30, b=0)
        )
        # Change to solid dark color for visibility
        fig_daily_seq.update_traces(marker_color='#004c4c', marker_line_width=0)
        fig_daily_seq.update_xaxes(rangeslider_visible=True, tickangle=45)
        st.plotly_chart(fig_daily_seq, use_container_width=True)
        
        st.divider()
        
        # 2. Sequential WEEKLY Timeline
        st.markdown("#### 📅 Sequential WEEKLY ATR")
        st.markdown("Weekly volatility average over time.")
        hist_weekly = filtered_daily.groupby(['Year', 'Week_Number']).agg({'Avg_ATR': 'mean'}).reset_index()
        hist_weekly['Label'] = "W" + hist_weekly['Week_Number'].astype(str) + " " + hist_weekly['Year'].astype(str)
        
        fig_week_seq = px.bar(hist_weekly, x='Label', y='Avg_ATR', 
                              template='plotly_white')
        fig_week_seq.update_traces(marker_color='#5d6d7e')
        fig_week_seq.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_week_seq, use_container_width=True)
        
        st.divider()
        
        # 3. Sequential MONTHLY Timeline
        st.markdown("#### 📜 Full History: Monthly ATR Summary")
        hist_monthly = filtered_daily.groupby(['Year', 'Month_Name', 'Month_Num']).agg({
            'Avg_ATR': 'mean',
            'Date': 'min'
        }).reset_index().sort_values(['Year', 'Month_Num'])
        hist_monthly['Label'] = hist_monthly['Month_Name'].str[:3] + " " + hist_monthly['Year'].astype(str)
        
        fig_hist = px.bar(hist_monthly, x='Label', y='Avg_ATR', 
                          template='plotly_white', title="Monthly Average Flow")
        fig_hist.update_traces(marker_color='#2c3e50') # Dark Slate Grey
        fig_hist.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()
        
        # 4. Per-Month Detailed Breakdown
        st.markdown("#### 📉 Detailed Monthly Breakdown (Daily ATR)")
        st.markdown("Explore daily ATR for every single month individually.")
        
        # Sort months chromologically for the expanders
        for (y, m_name), m_data in filtered_daily.groupby(['Year', 'Month_Name'], sort=False):
            with st.expander(f"📅 Daily Detail: {m_name} {y}", expanded=False):
                fig_m_daily = px.bar(
                    m_data, x='Date', y='Avg_ATR', # Changed x to 'Date' from 'Time' as per original context
                    text_auto='.2f', # Kept original text_auto
                    template='plotly_white'
                )
                fig_m_daily.update_traces(marker_color='#003366') # Kept original marker_color
                # Setting x-axis to category to remove weekend gaps within the month
                fig_m_daily.update_xaxes(type='category')
                fig_m_daily.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_m_daily, use_container_width=True)

with t9:
    st.subheader("🕯️ Directional ATR Analysis")
    st.markdown("ATR values signed by candle direction: **Green (+)** for Bullish expansion, **Red (-)** for Bearish expansion.")
    
    # Independent Controls for this tab
    c_idx1, c_idx2, c_idx3, c_idx4 = st.columns(4)
    with c_idx1:
        tab_tf = st.selectbox("Chart Timeframe", options=list(tf_opt.keys()), index=list(tf_opt.keys()).index(sel_tf), key="tab_tf")
    with c_idx2:
        tab_atr_p = st.number_input("Chart ATR Period", 1, 200, atr_p, key="tab_atr_p")
    with c_idx3:
        max_bars = st.slider("Lookback Bars", 50, 2000, 500, key="tab_max_bars")
    with c_idx4:
        threshold = st.number_input("ATR Intensity Threshold", 0.0, 50.0, 0.5, 0.1, key="tab_thresh")
        
    # Calculation
    # Note: We use 'raw' data loaded globally to re-resample for this specific view
    df_tab = raw.resample(tf_opt[tab_tf], on='Time').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna().tail(max_bars)
    
    df_tab['PrevClose'] = df_tab['Close'].shift(1)
    df_tab['TR'] = np.maximum(df_tab['High'] - df_tab['Low'], 
                        np.maximum(abs(df_tab['High'] - df_tab['PrevClose']), 
                                   abs(df_tab['Low'] - df_tab['PrevClose'])))
    df_tab['ATR'] = df_tab['TR'].rolling(window=tab_atr_p).mean()
    df_tab = df_tab.dropna().reset_index()
    
    # Calculate Stats
    df_tab['Sign'] = np.where(df_tab['Close'] >= df_tab['Open'], 1, -1)
    
    bull_high = len(df_tab[(df_tab['Sign'] == 1) & (df_tab['ATR'] >= threshold)])
    bear_high = len(df_tab[(df_tab['Sign'] == -1) & (df_tab['ATR'] >= threshold)])
    
    # Show Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Bullish Explosions (ATR > Threshold)", f"{bull_high}")
    m2.metric("Bearish Explosions (ATR > Threshold)", f"{bear_high}")
    bias = "Neutral"
    if bull_high > bear_high: bias = "Bullish Intensity"
    elif bear_high > bull_high: bias = "Bearish Intensity"
    m3.metric("Volatility Bias", bias)

    st.divider()

    # Apply Visual Logic
    df_tab['Dir_ATR'] = df_tab['ATR'] * df_tab['Sign']
    df_tab['Color'] = np.where(df_tab['Sign'] == 1, '#2ecc71', '#e74c3c') # Emerald Green / Alizarin Red
    
    # Charting
    fig_dir = go.Figure()
    fig_dir.add_trace(go.Bar(
        x=df_tab['Time'], 
        y=df_tab['Dir_ATR'],
        marker_color=df_tab['Color'],
        name="Directional ATR",
        hovertemplate="Time: %{x}<br>ATR: %{y:.4f}<br><extra></extra>"
    ))
    
    fig_dir.update_layout(
        template='plotly_white',
        height=600,
        yaxis_title="Signed ATR (+ Bullish / - Bearish)",
        xaxis_title="Time",
        showlegend=False,
        xaxis_type='category', # Removes all overnight and weekend gaps
        margin=dict(l=0, r=0, t=20, b=0)
    )
    fig_dir.update_xaxes(rangeslider_visible=True)
    # Add horizontal zero line
    fig_dir.add_hline(y=0, line_dash="dash", line_color="#333", opacity=0.5)
    
    st.plotly_chart(fig_dir, use_container_width=True)
    
    st.info("💡 High bars (positive or negative) indicate strong directional volatility. Falling bars toward the zero-line indicate volatility exhaustion or range-bound behavior.")

with t10:
    st.header("🤖 AI Data Assistant")
    st.markdown("Ask natural language questions about your filtered ATR dataset.")
    
    if not final_gemini_key and not final_groq_key:
        st.warning("⚠️ **No API Key Found**: Please enter a Gemini or Groq key in the sidebar to use the AI Assistant.")
    else:
        # 1. Configuration and Model List
        available_models = get_available_models(final_gemini_key, final_groq_key)
        
        c_ai1, c_ai2 = st.columns([1, 2])
        with c_ai1:
            default_ix = 0
            # Pick a fast default
            if "models/gemini-2.0-flash" in available_models:
                default_ix = available_models.index("models/gemini-2.0-flash")
            elif "groq/llama-3.3-70b-versatile" in available_models:
                default_ix = available_models.index("groq/llama-3.3-70b-versatile")
                
            sel_model_name = st.selectbox("Select AI Model", options=available_models, index=default_ix)
            st.caption(f"Provider: {'Groq' if sel_model_name.startswith('groq') else 'Google Gemini'}")
            
        with c_ai2:
            st.info("💡 **Quant Mode**: Groq models are ultra-fast for code generation.")

        user_question = st.text_input("Ask a question about the dataset (e.g., 'What was the highest ATR in 2024?')")
        
        if user_question:
            # Create unique cache key for this question + model
            cache_key = hashlib.md5((user_question + sel_model_name).encode()).hexdigest()
            
            # Prepare dataframes for AI
            df_for_ai = filtered_daily.copy() # Aggregated
            
            # 1. Fetch Code (from Cache or API)
            if cache_key in st.session_state.ai_cache:
                cached = st.session_state.ai_cache[cache_key]
                generated_code = cached["code"]
                usage_text = cached["usage"]
                st.caption(f"🚀 **Instant Response (Cached)** | {usage_text}")
            else:
                full_prompt = f"""
                You are a senior quantitative analyst and data scientist with deep expertise in multi-asset financial analysis and Python-based analytical workflows. You write production-quality pandas code: precise, vectorized, and institutionally rigorous.

                You have access to 6 DataFrames in memory representing 3 major assets:
                
                **1. Aggregated Trends (Daily/Session Metrics):**
                - `gold_df`, `dxy_df`, `eur_df`
                - Scope: Filtered by the current sidebar date settings.
                - Columns: {df_for_ai.columns.tolist()}

                **2. High-Precision History (Full 5-Year 1-Minute OHLC):**
                - `gold_raw`, `dxy_raw`, `eur_raw`
                - Scope: Unfiltered historical data.
                - Columns: {raw.columns.tolist()}

                **User question:**
                {user_question}

                ## Requirements
                - Return only raw, executable Python code—no markdown fencing.
                - Use the DataFrame names exactly as listed above.
                - For correlation or comparison, merge DataFrames on the 'Time' (raw) or 'Date' (df) columns.
                - Store the final answer in a variable named `result`.
                - You have access to: `pd`, `np`, `px`, `go`, `ta`, `qs`, `stats`, `sk`, `sm`, `arch`, `sns`, `plt`, `pypopt`, `xlsx`, `dt`, `math`.
                - Use `sk` for ML, `sm` for stats, `arch` for volatility, `pypopt` for portfolio optimization, `dt` for datetime operations, and `math` for mathematical functions.
                - For visualizations, assign the plotly figure or a seaborn/matplotlib axis to `result`.

                Write Python code only.
                """
                
                try:
                    if sel_model_name.startswith("groq"):
                        if not final_groq_key:
                            st.error("Groq API Key missing!")
                            st.stop()
                        client = Groq(api_key=final_groq_key)
                        g_model = sel_model_name.split("/")[-1]
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": full_prompt}],
                            model=g_model,
                        )
                        generated_code = chat_completion.choices[0].message.content.strip()
                        usage = chat_completion.usage
                        usage_text = f"Input: `{usage.prompt_tokens}` | Output: `{usage.completion_tokens}` | Total: `{usage.total_tokens}`"
                        
                        # Update session totals
                        st.session_state.total_input_tokens += usage.prompt_tokens
                        st.session_state.total_output_tokens += usage.completion_tokens
                        # Estimate cost (Approximation: $0.15 per 1M tokens)
                        st.session_state.total_cost += (usage.total_tokens / 1_000_000) * 0.15
                    else:
                        genai.configure(api_key=final_gemini_key)
                        model = genai.GenerativeModel(sel_model_name)
                        response = model.generate_content(full_prompt)
                        generated_code = response.text.strip()
                        usage = response.usage_metadata
                        usage_text = f"Input: `{usage.prompt_token_count}` | Output: `{usage.candidates_token_count}` | Total: `{usage.total_token_count}`"
                        
                        # Update session totals
                        st.session_state.total_input_tokens += usage.prompt_token_count
                        st.session_state.total_output_tokens += usage.candidates_token_count
                        # Estimate cost (Approximation: $0.075 per 1M tokens for Flash)
                        st.session_state.total_cost += (usage.total_token_count / 1_000_000) * 0.075
                    
                    st.caption(f"💎 **Token Usage**: {usage_text}")

                    # Cleanup formatting artifacts
                    if "```python" in generated_code:
                        generated_code = generated_code.split("```python")[-1].split("```")[0].strip()
                    elif "```" in generated_code:
                        generated_code = generated_code.split("```")[-1].split("```")[0].strip()
                    
                    # Store in Cache
                    st.session_state.ai_cache[cache_key] = {
                        "code": generated_code,
                        "usage": usage_text
                    }
                except Exception as e:
                    st.error(f"API Error: {e}")
                    st.stop()
            
            # 2. Execution and Rendering (Shared for both Cache and API flows)
            try:
                with st.expander("📝 View AI Quant Code"):
                    st.code(generated_code, language='python')
                
                # Prepare restricted environment
                safe_globals = {
                    "gold_df": df_for_ai,
                    "gold_raw": raw,
                    "dxy_df": resample_and_calculate(raw_dxy, tf_opt[sel_tf], atr_p)[0] if raw_dxy is not None else None,
                    "dxy_raw": raw_dxy,
                    "eur_df": resample_and_calculate(raw_eur, tf_opt[sel_tf], atr_p)[0] if raw_eur is not None else None,
                    "eur_raw": raw_eur,
                    "pd": pd,
                    "np": np,
                    "px": px,
                    "go": go,
                    "ta": ta,
                    "qs": qs,
                    "stats": stats,
                    "sk": sklearn,
                    "sm": sm,
                    "arch": arch_model,
                    "sns": sns,
                    "plt": plt,
                    "pypopt": pypfopt,
                    "xlsx": xlsxwriter,
                    "dt": datetime,
                    "math": math,
                    "result": None
                }
                
                # Execute code
                exec(generated_code, safe_globals)
                
                st.subheader("💡 Analysis Insight")
                final_result = safe_globals.get("result")
                
                # Intelligent Rendering
                if final_result is not None:
                    # Check if it's a plotly figure
                    if hasattr(final_result, 'to_dict') and 'data' in final_result.to_dict():
                        # Professional Touch: Ensure AI charts also have no gaps
                        final_result.update_xaxes(type='category')
                        st.plotly_chart(final_result, use_container_width=True)
                    else:
                        st.write(final_result)
                else:
                    st.info("The AI executed the logic, but no 'result' variable was assigned. Try asking to 'Show the result'.")
            
            except Exception as e:
                st.error(f"Execution Error: {e}")

        # Session Usage Summary (Footer)
        st.divider()
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Session Total Tokens", f"{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}")
        with b2:
            st.metric("Total Input / Output", f"{st.session_state.total_input_tokens:,} / {st.session_state.total_output_tokens:,}")
        with b3:
            st.metric("Est. Session Cost", f"${st.session_state.total_cost:.5f}")
        st.caption("⚠️ Costs are estimates based on standard Tier 1 pricing ($0.075-$0.15 per 1M tokens).")

with t11:
    st.subheader("🔗 Multi-Asset Lead-Lag Engine")
    st.markdown("Discover which asset moves first. This analysis determines if one asset's movements predict another's with a time delay.")
    
    # Configuration
    ll_c1, ll_c2, ll_c3 = st.columns(3)
    with ll_c1:
        base_sym = st.selectbox("Base Asset (The 'Leader'?)", ["DXY", "GOLD", "EURUSD"], index=0)
    with ll_c2:
        target_sym = st.selectbox("Target Asset (The 'Follower'?)", ["GOLD", "DXY", "EURUSD"], index=1)
    with ll_c3:
        lookback_days = st.slider("Lookback Window (Last X Days)", 1, 30, 7)
    
    if base_sym == target_sym:
        st.warning("Please select two different assets to compare.")
    else:
        # Asset Mapping
        asset_map = {
            "GOLD": raw,
            "DXY": raw_dxy,
            "EURUSD": raw_eur
        }
        
        # 1. Fetch and Prepare Data
        b_data = asset_map[base_sym].copy()
        t_data = asset_map[target_sym].copy()
        
        # Filter for recent days to keep CCF fast
        cutoff = b_data['Time'].max() - pd.Timedelta(days=lookback_days)
        b_data = b_data[b_data['Time'] > cutoff]
        t_data = t_data[t_data['Time'] > cutoff]
        
        # Calculate Returns
        b_data['Ret'] = b_data['Close'].pct_change()
        t_data['Ret'] = t_data['Close'].pct_change()
        
        # Merge on exact Time
        merged = pd.merge(b_data[['Time', 'Ret']], t_data[['Time', 'Ret']], on='Time', how='inner', suffixes=('_base', '_target')).dropna()
        
        if len(merged) < 100:
            st.error("Not enough overlapping data points for this period.")
        else:
            # 2. Calculate Cross-Correlation
            max_lag = 60 # 60 minutes
            lags = np.arange(-max_lag, max_lag + 1)
            corrs = []
            
            for lag in lags:
                if lag < 0:
                    # Base leads Target (Target shifted back)
                    c = merged['Ret_base'].shift(abs(lag)).corr(merged['Ret_target'])
                elif lag > 0:
                    # Target leads Base (Base shifted back)
                    c = merged['Ret_target'].shift(lag).corr(merged['Ret_base'])
                else:
                    c = merged['Ret_base'].corr(merged['Ret_target'])
                corrs.append(c)
            
            # 3. Visualization
            ccf_df = pd.DataFrame({'Lag (Minutes)': lags, 'Correlation': corrs})
            
            # Identify critical spikes
            peak_idx = np.argmax(np.abs(corrs))
            peak_lag = lags[peak_idx]
            peak_corr = corrs[peak_idx]
            
            fig_ll = px.bar(ccf_df, x='Lag (Minutes)', y='Correlation',
                          title=f"Cross-Correlation: {base_sym} vs {target_sym}",
                          template='plotly_white',
                          color='Correlation',
                          color_continuous_scale='RdBu_r',
                          range_color=[-1, 1])
            
            # Highlight center line
            fig_ll.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
            
            st.plotly_chart(fig_ll, use_container_width=True)
            
            # 4. Quant Interpretation
            st.divider()
            st.subheader("🧠 Quant Interpretation")
            
            if peak_lag < 0:
                direction = "Inverse" if peak_corr < 0 else "Positive"
                st.success(f"📈 **Predictive Signal Detected**: **{base_sym}** currently LEADS **{target_sym}** by approximately **{abs(peak_lag)} minutes**.")
                st.info(f"A {direction} move in {base_sym} typically shows its maximum impact on {target_sym} volatility {abs(peak_lag)} minutes later. (Strength: {abs(peak_corr):.2f})")
            elif peak_lag > 0:
                st.warning(f"📉 **Reverse Signal**: **{target_sym}** actually LEADS **{base_sym}** by approximately **{peak_lag} minutes**.")
            else:
                st.info(f"⚖️ **Synchronous Correlation**: {base_sym} and {target_sym} move at the exact same time (Lag 0). There is currently no predictive lead-lag relationship.")
                
            st.caption("ℹ️ Note: This calculation uses 1-minute returns to identify high-frequency dependencies. Correlation scores above 0.3 are statistically significant in FX markets.")

st.caption(f"XAUUSD ATR Analysis | Method: Rolling ATR ({atr_p}) | Resampled from 1m to {sel_tf}")
