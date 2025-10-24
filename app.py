import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- USTAWIENIA STRONY ---
st.set_page_config(layout="wide")
st.title("📊 Profesjonalny Dashboard Kryptowalut")

# --- WYBÓR ŹRÓDŁA DANYCH ---
source = st.radio(
    "📡 Wybierz źródło danych:",
    ["Binance", "CoinGecko"],
    horizontal=True
)

# --- POBIERANIE SYMBOLI ---
@st.cache_data(ttl=3600)
def get_all_symbols(source):
    """Pobiera listę kryptowalut z Binance lub CoinGecko"""
    if source == "Binance":
        try:
            url = "https://api.binanceproxy.net/api/v3/exchangeInfo"  # proxy omija 451
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if "symbols" in data:
                return sorted([s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"])
        except Exception as e:
            st.warning(f"❌ Błąd Binance: {e}")
            source = "CoinGecko"  # fallback
    try:
        url = "https://api.coingecko.com/api/v3/coins/list"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        symbols = [item['symbol'].upper() + "USDT" for item in data if 'symbol' in item][:500]
        return sorted(list(set(symbols)))
    except Exception as e:
        st.error(f"❌ Błąd CoinGecko: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

symbols_list = get_all_symbols(source)

# --- SESSION STATE ---
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {}

# --- WYBÓR KRYPTOWALUT ---
with st.form(key='crypto_form'):
    cryptos = st.multiselect("🔎 Wybierz kryptowaluty:", symbols_list, default=["BTCUSDT","ETHUSDT"])
    submit_button = st.form_submit_button("Załaduj dane")

# --- POBIERANIE DANYCH ---
def get_data(symbol, source="Binance", limit=365):
    """Pobiera dane z Binance lub CoinGecko"""
    if source == "Binance":
        try:
            url = f"https://api.binanceproxy.net/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
            data = requests.get(url, timeout=10).json()
            df = pd.DataFrame(data, columns=[
                'Open time','Open','High','Low','Close','Volume','Close time',
                'Quote asset volume','Trades','Taker buy base','Taker buy quote','Ignore'
            ])
            df['Close'] = df['Close'].astype(float)
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df.set_index('Open time', inplace=True)
            return df
        except Exception as e:
            st.warning(f"⚠️ Binance error: {e}. Przełączam na CoinGecko...")
            source = "CoinGecko"

    # Fallback do CoinGecko
    try:
        cg_id = symbol.replace("USDT", "").lower()
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=365"
        r = requests.get(url, timeout=10)
        data = r.json()['prices']
        df = pd.DataFrame(data, columns=['time','Close'])
        df['Open time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('Open time', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ CoinGecko error for {symbol}: {e}")
        return pd.DataFrame()

# --- WSKAŹNIKI TECHNICZNE ---
def add_indicators(df):
    if df.empty: return df
    df['SMA20'] = SMAIndicator(df['Close'], 20).sma_indicator()
    df['EMA20'] = EMAIndicator(df['Close'], 20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], 14).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(df['Close'])
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    return df

# --- PREDYKCJE ---
def linear_prediction(df, days=30):
    if len(df) < 10: return pd.DataFrame()
    df = df.reset_index()
    df['t'] = np.arange(len(df))
    X = df[['t']].values
    y = df['Close'].values
    model = LinearRegression().fit(X, y)
    future = np.arange(len(df), len(df)+days).reshape(-1,1)
    preds = model.predict(future)
    dates = pd.date_range(df['Open time'].iloc[-1], periods=days+1, freq='D')[1:]
    return pd.DataFrame({'Date': dates, 'Predicted': preds})

def prophet_prediction(df, months=1):
    if len(df) < 30: return pd.DataFrame()
    dfp = df.reset_index().rename(columns={'Open time':'ds','Close':'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(dfp)
    fut = m.make_future_dataframe(periods=30*months)
    fc = m.predict(fut)
    return fc[['ds','yhat']].tail(30*months)

def lstm_prediction(df, days=30):
    if len(df) < 100: return pd.DataFrame()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i,0])
        y.append(data[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    last_60 = data[-60:]
    preds = []
    for _ in range(days):
        p = model.predict(last_60.reshape(1,60,1), verbose=0)[0,0]
        preds.append(p)
        last_60 = np.append(last_60[1:], p).reshape(-1,1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    return pd.DataFrame({'Date': future_dates, 'LSTM': preds.flatten()})

# --- SYGNAŁ INWESTYCYJNY ---
def generate_signal(df):
    if df.empty: return "Brak danych"
    last = df.iloc[-1]
    score = 0
    if last['RSI'] < 30: score += 1
    elif last['RSI'] > 70: score -= 1
    if last['Close'] > last['SMA20']: score += 1
    else: score -= 1
    if score >= 1: return "Kup ✅"
    elif score <= -1: return "Sprzedaj ❌"
    return "Zostaw 🤔"

# --- ZAŁADUJ DANE ---
if submit_button:
    for s in cryptos:
        df = get_data(s, source)
        df = add_indicators(df)
        st.session_state['dfs'][s] = df

# --- PORÓWNANIE ---
st.subheader("📈 Porównanie kryptowalut")
fig = go.Figure()
for s in cryptos:
    if s in st.session_state['dfs'] and not st.session_state['dfs'][s].empty:
        df = st.session_state['dfs'][s]
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=s))
fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- SZCZEGÓŁOWA ANALIZA ---
days = st.slider("Dni do prognozy", 7, 180, 30)
for s in cryptos:
    if s not in st.session_state['dfs'] or st.session_state['dfs'][s].empty:
        continue
    df = st.session_state['dfs'][s]
    with st.expander(f"{s} - analiza szczegółowa"):
        st.write(df.tail(10))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_high'], name='BB High', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='BB Low', line=dict(dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Prognozy:**")
        lr = linear_prediction(df, days)
        if not lr.empty:
            st.line_chart(lr.set_index('Date')['Predicted'])
        lstm = lstm_prediction(df, days)
        if not lstm.empty:
            st.line_chart(lstm.set_index('Date')['LSTM'])
        p = prophet_prediction(df, 1)
        if not p.empty:
            st.line_chart(p.set_index('ds')['yhat'])

        st.write(f"**Sygnał inwestycyjny:** {generate_signal(df)}")
