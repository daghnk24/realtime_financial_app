import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Real-Time Financial Risk Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- CẤU HÌNH REAL-TIME AUTOMATION ---
st.sidebar.title("⚙️ Cấu hình Hệ thống")
refresh_interval = st.sidebar.slider("Tần suất cập nhật (Giây):", min_value=5, max_value=60, value=10)
count = st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# --- LỰA CHỌN MÃ TÀI CHÍNH ---
st.sidebar.subheader("📌 Chọn Mã Tài sản")
ticker_symbol = st.sidebar.text_input("Nhập Ticker (Ví dụ: BTC-USD, AAPL, VCB.VN):", value="BTC-USD")
period = st.sidebar.selectbox("Khoảng thời gian lịch sử:", ["1d", "5d", "1mo", "3mo"], index=1)
interval = st.sidebar.selectbox("Khung thời gian (Interval):", ["1m", "2m", "5m", "15m"], index=0)

st.title("📈 Hệ thống Quản trị Rủi ro & Biến động Giá Tài chính Real-Time")
st.caption(f"Trạng thái: 🟢 Đang chạy Real-time | Lần cập nhật thứ: {count} | Mã: **{ticker_symbol}**")

# --- HÀM TẢI DỮ LIỆU REAL-TIME ---
@st.cache_data(ttl=5)
def load_data(symbol, period, interval):
    data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data

try:
    df = load_data(ticker_symbol, period, interval)
    if df.empty:
        st.error("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã Ticker!")
        st.stop()
except Exception as e:
    st.error(f"Lỗi khi tải dữ liệu: {e}")
    st.stop()

# --- TÍNH TOÁN CÁC CHỈ SỐ KHOA HỌC / TÀI CHÍNH ---
df['Return'] = df['Close'].pct_change()
latest_price = df['Close'].iloc[-1]
price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
pct_change = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0

# 1. Phát hiện bất thường bằng Machine Learning (Isolation Forest)
features = df[['Close', 'Volume']].dropna()
if len(features) > 10:
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    df.loc[features.index, 'Anomaly'] = iso_model.fit_predict(features)
else:
    df['Anomaly'] = 1

# 2. Mô phỏng Monte Carlo để tính Value-at-Risk (VaR 95%)
returns = df['Return'].dropna()
mean_return = returns.mean()
std_return = returns.std()

# Mô phỏng 1,000 kịch bản giá tiếp theo
np.random.seed(42)
simulated_returns = np.random.normal(mean_return, std_return, 1000)
var_95 = np.percentile(simulated_returns, 5) * 100  # Rủi ro thua lỗ tối đa ở độ tin cậy 95%

# --- HIỂN THỊ KPI MAIN ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Giá Hiện Tại", f"${latest_price:,.2f}", f"{pct_change:+.2f}%")
col2.metric("Biến động (Volatility Std)", f"{std_return*100:.2f}%")
col3.metric("Value-at-Risk (VaR 95%)", f"{var_95:.2f}%", help="Mức lỗ tối đa có thể xảy ra trong khung thời gian tới ở độ tin cậy 95%")
col4.metric("Số điểm Anomaly (ML)", f"{(df['Anomaly'] == -1).sum()} điểm")

# --- ĐỒ THỊ CANDLESTICK VÀ MACHINE LEARNING ANOMALY ---
st.subheader("📊 Đồ thị Giá Real-Time & Phát hiện Bất thường (Anomaly Detection)")

fig = go.Figure()

# Nến Nhật
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Giá (Candlestick)"
))

# Đánh dấu các điểm bất thường do Machine Learning phát hiện
anomalies = df[df['Anomaly'] == -1]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies['Close'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Bất thường (ML Anomaly)'
    ))

fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- KHU VỰC PHÂN TÍCH HÀM LƯỢNG KHOA HỌC ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎲 Mô phỏng Monte Carlo (Rủi ro VaR)")
    fig_hist = px.histogram(simulated_returns * 100, nbins=50, 
                            labels={'value': 'Mức sinh lời mô phỏng (%)'},
                            title="Phân phối mô phỏng Monte Carlo 1,000 kịch bản",
                            color_discrete_sequence=['#636EFA'])
    fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text=f"VaR 95% ({var_95:.2f}%)")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("📋 Bảng Dữ liệu Mới nhất (Real-time Stream)")
    st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].tail(8).sort_index(ascending=False))