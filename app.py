import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import pandas_ta as ta

st.set_page_config(layout="wide")
st.title("📈 Real-Time Crypto Signal Dashboard")

binance = ccxt.binance({
    'enableRateLimit': True
})

@st.cache_data(ttl=300)
def fetch_ohlcv_df(symbol, timeframe='5m', limit=500):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    return df

def train_predict(symbol):
    df = fetch_ohlcv_df(symbol)
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.obv(append=True)
    macd = df.ta.macd()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bb = df.ta.bbands()
    df['BB_upper'] = bb.iloc[:, 0]
    df['BB_mid'] = bb.iloc[:, 1]
    df['BB_lower'] = bb.iloc[:, 2]
    df['ATR'] = df.ta.atr(length=14)
    stoch = df.ta.stochrsi()
    df['StochRSI_K'] = stoch.iloc[:, 0]
    df['StochRSI_D'] = stoch.iloc[:, 1]
    df.ta.adx(append=True)
    df['momentum_3d'] = df['Close'].pct_change(3)
    df['volatility_5d'] = df['Close'].rolling(5).std()
    df['ema_crossover'] = (df['EMA_50'] > df['EMA_200']).astype(int)
    df['future_return'] = (df['Close'].shift(-10) - df['Close']) / df['Close']
    df.dropna(inplace=True)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = ['RSI_14', 'MACD', 'MACD_signal', 'EMA_50', 'EMA_200', 'OBV',
                'BB_upper', 'BB_mid', 'BB_lower', 'ATR', 'StochRSI_K', 'StochRSI_D',
                'ADX_14', 'momentum_3d', 'volatility_5d', 'ema_crossover']

    if not all(f in df.columns for f in features + ['Close', 'future_return']):
        st.warning(f"⚠️ Missing features for {symbol}")
        return None

    X = df[features]
    y_cls = df['Target']
    y_reg = df['future_return']
    X_train_c, _, y_train_c, _ = train_test_split(X, y_cls, test_size=0.2, shuffle=False)
    X_train_r, _, y_train_r, _ = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train_c, y_train_c)
    reg = XGBRegressor(n_estimators=80)
    reg.fit(X_train_r, y_train_r)
    last_row = X.iloc[[-1]]
    prediction = clf.predict(last_row)[0]
    confidence = clf.predict_proba(last_row)[0][prediction]
    predicted_return = reg.predict(last_row)[0]
    entry = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    tp = entry + 2 * atr
    sl = entry - 1.5 * atr
    return {
        'pair': symbol,
        'signal': "LONG 🚀" if prediction == 1 else "SHORT 🔻",
        'confidence': round(confidence * 100, 2),
        'price': round(entry, 2),
        'TP': round(tp, 2),
        'SL': round(sl, 2),
        'Predicted_10m': round(entry * (1 + predicted_return), 2),
        'Time': df.index[-1].strftime("%Y-%m-%d %H:%M")
    }

if st.button("🔄 Refresh Signals"):
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    results = []
    for s in symbols:
        st.write(f"⏳ {s}...")
        res = train_predict(s)
        if res:
            results.append(res)
            st.success(f"✅ Done {s}")
        else:
            st.warning(f"⚠️ Skipped {s}")
    if results:
        df = pd.DataFrame(results)
        df.sort_values(by='confidence', ascending=False, inplace=True)
        st.dataframe(df)
    else:
        st.info("⚠️ No signals generated.")
