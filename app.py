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

st.set_page_config(layout="wide")
st.title("Profesjonalny Dashboard Kryptowalut - Optymalizacja Session State 🚀")

@st.cache_data(ttl=3600)
def get_all_symbols():
    """
    Pobiera wszystkie dostępne symbole z Binance.
    W razie błędu zwraca lokalną listę 200+ popularnych kryptowalut (fallback).
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"

    # ✅ Lokalna lista awaryjna — 200+ popularnych symboli
    fallback_symbols = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT",
        "MATICUSDT","LTCUSDT","SHIBUSDT","BCHUSDT","ATOMUSDT","FILUSDT","UNIUSDT","INJUSDT","OPUSDT","APTUSDT",
        "ARBUSDT","NEARUSDT","ETCUSDT","AAVEUSDT","SANDUSDT","RUNEUSDT","IMXUSDT","EGLDUSDT","VETUSDT","THETAUSDT",
        "FLOWUSDT","GALAUSDT","ICPUSDT","AXSUSDT","KAVAUSDT","STXUSDT","HBARUSDT","MANAUSDT","CHZUSDT","CRVUSDT",
        "SNXUSDT","DYDXUSDT","PEPEUSDT","SEIUSDT","BLURUSDT","FTMUSDT","COMPUSDT","ENSUSDT","WLDUSDT","XMRUSDT",
        "ALGOUSDT","GMXUSDT","GRTUSDT","QNTUSDT","SUIUSDT","ROSEUSDT","MINAUSDT","CFXUSDT","1INCHUSDT","DASHUSDT",
        "ZECUSDT","LRCUSDT","KSMUSDT","NEOUSDT","CELOUSDT","HOTUSDT","OCEANUSDT","RLCUSDT","BANDUSDT","HNTUSDT",
        "BALUSDT","CHRUSDT","YFIUSDT","CELRUSDT","XEMUSDT","BATUSDT","COTIUSDT","ZILUSDT","TRXUSDT","IOSTUSDT",
        "QTUMUSDT","FTTUSDT","BNXUSDT","SKLUSDT","XLMUSDT","BATUSDT","ENJUSDT","ANKRUSDT","NKNUSDT","OMGUSDT",
        "ICXUSDT","RSRUSDT","RVNUSDT","ONTUSDT","WAVESUSDT","ARUSDT","STORJUSDT","MASKUSDT","SPELLUSDT","ILVUSDT",
        "C98USDT","AUDIOUSDT","KDAUSDT","NMRUSDT","OGNUSDT","GLMRUSDT","SFPUSDT","LDOUSDT","MAGICUSDT","CAKEUSDT",
        "BICOUSDT","ALPHAUSDT","ACHUSDT","XECUSDT","DENTUSDT","DARUSDT","BTTUSDT","PYRUSDT","ATAUSDT","GALUSDT",
        "FXSUSDT","JOEUSDT","MOVRUSDT","AKROUSDT","SRMUSDT","FLUXUSDT","TRBUSDT","REEFUSDT","CTSIUSDT","PUNDIXUSDT",
        "KLAYUSDT","BAKEUSDT","API3USDT","MTLUSDT","SXPUSDT","JASMYUSDT","CKBUSDT","VTHOUSDT","WINUSDT","SYSUSDT",
        "TWTUSDT","FORTHUSDT","CEEKUSDT","CVCUSDT","XVGUSDT","PHAUSDT","BNXUSDT","VOXELUSDT","PROMUSDT","SUPERUSDT",
        "BETAUSDT","DEGOUSDT","HIGHUSDT","AERGOUSDT","ARPAUSDT","TVKUSDT","MBOXUSDT","LITUSDT","BELUSDT","SANTOSUSDT",
        "PORTOUSDT","BARUSDT","LAZIOUSDT","PSGUSDT","CITYUSDT","ATMUSDT","ACMUSDT","ASRUSDT","OGUSDT","ALPINEUSDT",
        "MOBUSDT","BONDUSDT","CTKUSDT","FIROUSDT","DODOUSDT","MIRUSDT","ERNUSDT","PERPUSDT","POLSUSDT","ADXUSDT",
        "MDTUSDT","DATAUSDT","KEYUSDT","AUTOUSDT","TOMOUSDT","HIVEUSDT","SCUSDT","WRXUSDT","DCRUSDT","BTSUSDT"
    ]

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Jeśli Binance zwraca poprawny JSON
        if "symbols" in data and isinstance(data["symbols"], list):
            symbols = [item["symbol"] for item in data["symbols"] if item.get("status") == "TRADING"]
            if len(symbols) > 10:
                return symbols
            else:
                st.warning("Zwrócono mało symboli z Binance — użyto lokalnej listy.")
                return fallback_symbols
        else:
            st.warning("Brak pola 'symbols' w odpowiedzi API Binance — użyto lokalnej listy.")
            return fallback_symbols

    except Exception as e:
        st.error(f"Błąd podczas pobierania symboli z Binance: {e}")
        return fallback_symbols



# --- Inicjalizacja session_state dla danych ---
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {}

# --- Formularz wyboru kryptowalut ---
with st.form(key='crypto_form'):
    cryptos = st.multiselect("Wybierz kryptowaluty (autocomplete działa podczas wpisywania):", symbols_list, default=["BTCUSDT","ETHUSDT"])
    submit_button = st.form_submit_button(label='Załaduj dane')

# --- Funkcje pobierania danych i wskaźników ---
def get_binance_data(symbol='BTCUSDT', interval='1d', limit=365):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['Open time','Open','High','Low','Close','Volume',
                                     'Close time','Quote asset volume','Number of trades',
                                     'Taker buy base','Taker buy quote','Ignore'])
    df['Close'] = df['Close'].astype(float)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    return df

def calculate_indicators(df):
    df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    stoch_rsi = StochRSIIndicator(df['Close'], window=14)
    df['StochRSI'] = stoch_rsi.stochrsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(df['Close'], window=20)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    return df

def linear_prediction(df, days=30):
    df_lr = df.reset_index()
    df_lr['t'] = np.arange(len(df_lr))
    X = df_lr[['t']]
    y = df_lr['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_t = np.arange(len(df_lr), len(df_lr)+days).reshape(-1,1)
    pred = model.predict(future_t)
    future_dates = pd.date_range(df_lr['Open time'].iloc[-1], periods=days+1, freq='D')[1:]
    return pd.DataFrame({'Date': future_dates, 'Predicted': pred})

def prophet_prediction(df, months=1):
    df_prophet = df.reset_index()[['Open time','Close']].rename(columns={'Open time':'ds','Close':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30*months)
    forecast = model.predict(future)
    return forecast[['ds','yhat']].tail(30*months)

def lstm_prediction(df, days=30):
    data = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i,0])
        y.append(scaled_data[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1],1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    last_60 = scaled_data[-60:]
    future_preds = []
    for _ in range(days):
        x_input = last_60.reshape((1,60,1))
        pred = model.predict(x_input, verbose=0)[0,0]
        future_preds.append(pred)
        last_60 = np.append(last_60[1:], pred).reshape(-1,1)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
    future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    return pd.DataFrame({'Date': future_dates, 'LSTM': future_preds.flatten()})

def generate_signal(df):
    last = df.iloc[-1]
    score = 0
    if last['RSI'] < 30: score +=1
    elif last['RSI'] > 70: score -=1
    if last['StochRSI'] < 0.2: score +=1
    elif last['StochRSI'] > 0.8: score -=1
    if last['Close'] > last['SMA20']: score +=1
    else: score -=1
    if score >=2: return "Kup ✅"
    elif score <=-2: return "Sprzedaj ❌"
    else: return "Zostaw 🤔"

# --- Pobieranie danych tylko jeśli ich nie ma w session_state ---
if submit_button:
    for sym in cryptos:
        if sym not in st.session_state['dfs']:
            df_temp = get_binance_data(sym)
            df_temp = calculate_indicators(df_temp)
            st.session_state['dfs'][sym] = df_temp

# --- Porównanie cen dla wszystkich kryptowalut ---
st.subheader("Porównanie cen")
fig = go.Figure()
for sym in cryptos:
    dfc = st.session_state['dfs'][sym]
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc['Close'], name=f'{sym} Close'))
st.plotly_chart(fig, use_container_width=True)

# --- Zakładki / Expander dla każdej kryptowaluty ---
st.subheader("Analiza szczegółowa")
days = st.slider("Dni do prognozy Liniowej i LSTM", 7, 180, 30)
for sym in cryptos:
    dfc = st.session_state['dfs'][sym]
    with st.expander(f"{sym} - szczegóły"):
        st.write(f"### {sym} - ostatnie 10 dni")
        st.dataframe(dfc.tail(10))
        # Wykres wskaźników
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['SMA20'], name='SMA20'))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['EMA20'], name='EMA20'))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['BB_high'], name='BB High', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['BB_low'], name='BB Low', line=dict(dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

        # Prognozy
        st.write("#### Prognozy")
        pred_lr = linear_prediction(dfc, days)
        st.line_chart(pred_lr.set_index('Date')['Predicted'], use_container_width=True)
        for m in [1,3,6]:
            forecast = prophet_prediction(dfc, months=m)
            st.line_chart(forecast.set_index('ds')['yhat'], use_container_width=True)
        pred_lstm = lstm_prediction(dfc, days)
        st.line_chart(pred_lstm.set_index('Date')['LSTM'], use_container_width=True)

        # Sygnał i symulator portfela
        signal = generate_signal(dfc)
        st.write(f"**Sygnał inwestycyjny:** {signal}")
        amount = st.number_input(f"Kwota w USD dla {sym}:", min_value=1.0, value=1000.0, key=f"amount_{sym}")
        pred_price = pred_lr['Predicted'].iloc[-1]
        profit = amount * (pred_price / dfc['Close'].iloc[-1] - 1)
        st.write(f"Prognozowany zysk/strata po {days} dniach: ${profit:.2f}")

