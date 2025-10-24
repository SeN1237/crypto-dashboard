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

# --- Konfiguracja strony ---
st.set_page_config(layout="wide", page_title="Crypto Dashboard", page_icon="💹")
st.title("💹 Profesjonalny Dashboard Kryptowalut 🚀")

# --- Funkcja pobierania symboli ---
@st.cache_data(ttl=3600)
def get_all_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"

    # ✅ Lista awaryjna — najpopularniejsze kryptowaluty
    fallback_symbols = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT",
        "MATICUSDT","LTCUSDT","SHIBUSDT","BCHUSDT","ATOMUSDT","FILUSDT","UNIUSDT","INJUSDT","OPUSDT","APTUSDT",
        "ARBUSDT","NEARUSDT","ETCUSDT","AAVEUSDT","SANDUSDT","RUNEUSDT","IMXUSDT","EGLDUSDT","VETUSDT","THETAUSDT"
    ]

    try:
        with st.spinner("⏳ Pobieranie listy symboli z Binance..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "symbols" in data and isinstance(data["symbols"], list):
                symbols = [s["symbol"] for s in data["symbols"] if s.get("status") == "TRADING"]
                if len(symbols) > 10:
                    return symbols
                else:
                    st.warning("Zwrócono mało symboli z Binance — użyto lokalnej listy.")
                    return fallback_symbols
            else:
                st.warning("Nie udało się pobrać symboli z Binance — użyto lokalnej listy.")
                return fallback_symbols
    except Exception as e:
        st.error(f"Błąd podczas pobierania symboli: {e}")
        return fallback_symbols


# --- Inicjalizacja danych w stanie sesji ---
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {}

# --- Pobranie listy symboli ---
symbols_list = get_all_symbols()
if not symbols_list or len(symbols_list) == 0:
    st.warning("Nie udało się pobrać symboli z Binance — używana lista awaryjna.")
    symbols_list = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]

# --- Wybór kryptowalut ---
default_symbols = ["BTCUSDT", "ETHUSDT"]
valid_defaults = [s for s in default_symbols if s in symbols_list]

cryptos = st.multiselect(
    "🔍 Wybierz kryptowaluty (autocomplete działa podczas wpisywania):",
    options=symbols_list,
    default=valid_defaults
)

# --- Przycisk ładowania danych ---
if st.button("📈 Załaduj dane"):
    for sym in cryptos:
        if sym not in st.session_state['dfs']:
            with st.spinner(f"⏳ Pobieranie danych dla {sym}..."):
                url = f'https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=365'
                data = requests.get(url).json()
                df = pd.DataFrame(data, columns=[
                    'Open time','Open','High','Low','Close','Volume','Close time',
                    'Quote asset volume','Number of trades','Taker buy base',
                    'Taker buy quote','Ignore'
                ])
                df['Close'] = df['Close'].astype(float)
                df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
                df.set_index('Open time', inplace=True)

                # Wskaźniki
                df['SMA20'] = SMAIndicator(df['Close'], 20).sma_indicator()
                df['EMA20'] = EMAIndicator(df['Close'], 20).ema_indicator()
                df['RSI'] = RSIIndicator(df['Close'], 14).rsi()
                stoch = StochRSIIndicator(df['Close'], 14)
                df['StochRSI'] = stoch.stochrsi()
                macd = MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                bb = BollingerBands(df['Close'], 20)
                df['BB_high'] = bb.bollinger_hband()
                df['BB_low'] = bb.bollinger_lband()
                st.session_state['dfs'][sym] = df

# --- Sekcja porównania ---
if len(cryptos) > 0:
    st.subheader("📊 Porównanie cen wybranych kryptowalut")
    fig = go.Figure()
    for sym in cryptos:
        df = st.session_state['dfs'].get(sym)
        if df is not None:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=sym))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Szczegółowa analiza ---
    st.subheader("🔬 Analiza szczegółowa")
    days = st.slider("📅 Dni do prognozy (Liniowa + LSTM):", 7, 180, 30)

    for sym in cryptos:
        df = st.session_state['dfs'][sym]
        with st.expander(f"📈 {sym} - Analiza"):
            st.write(f"### Ostatnie dane ({sym})")
            st.dataframe(df.tail(10))

            # Wykres wskaźników
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_high'], name='BB High', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='BB Low', line=dict(dash='dot')))
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # --- Predykcje ---
            st.write("#### 🔮 Prognozy")
            df_lr = df.reset_index()
            df_lr['t'] = np.arange(len(df_lr))
            X = df_lr[['t']]
            y = df_lr['Close']
            lr = LinearRegression().fit(X, y)
            future_t = np.arange(len(df_lr), len(df_lr)+days).reshape(-1,1)
            pred_lr = lr.predict(future_t)
            future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
            df_pred = pd.DataFrame({'Date': future_dates, 'Predicted': pred_lr})
            st.line_chart(df_pred.set_index('Date'))

            # Prophet
            df_prophet = df.reset_index()[['Open time','Close']].rename(columns={'Open time':'ds','Close':'y'})
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            st.line_chart(forecast.set_index('ds')['yhat'])

            # LSTM
            data = df['Close'].values.reshape(-1,1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            X_train, y_train = [], []
            for i in range(60, len(scaled)):
                X_train.append(scaled[i-60:i,0])
                y_train.append(scaled[i,0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=3, batch_size=16, verbose=0)
            last_60 = scaled[-60:]
            preds = []
            for _ in range(days):
                pred = model.predict(last_60.reshape(1,60,1), verbose=0)[0,0]
                preds.append(pred)
                last_60 = np.append(last_60[1:], pred).reshape(-1,1)
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
            df_lstm = pd.DataFrame({'Date': future_dates, 'LSTM': preds.flatten()})
            st.line_chart(df_lstm.set_index('Date'))

            # --- Sygnał inwestycyjny ---
            last = df.iloc[-1]
            score = 0
            if last['RSI'] < 30: score += 1
            elif last['RSI'] > 70: score -= 1
            if last['StochRSI'] < 0.2: score += 1
            elif last['StochRSI'] > 0.8: score -= 1
            if last['Close'] > last['SMA20']: score += 1
            else: score -= 1

            if score >= 2:
                signal = "Kup ✅"
            elif score <= -2:
                signal = "Sprzedaj ❌"
            else:
                signal = "Zostaw 🤔"

            st.markdown(f"### 💡 Sygnał inwestycyjny: **{signal}**")

            # --- Symulator portfela ---
            amount = st.number_input(f"💰 Kwota w USD dla {sym}:", min_value=10.0, value=1000.0, key=f"amount_{sym}")
            pred_price = df_pred['Predicted'].iloc[-1]
            profit = amount * (pred_price / df['Close'].iloc[-1] - 1)
            st.success(f"💵 Prognozowany zysk/strata po {days} dniach: **${profit:.2f}**")

else:
    st.info("👆 Wybierz kryptowaluty i kliknij 'Załaduj dane', aby rozpocząć analizę.")
