# streamlit_crypto_app_advanced.py

import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import plotly.graph_objects as go
import time

# ----------------------------
# CONFIGURAÇÃO BINANCE
# ----------------------------
client = Client()  # Dados públicos

# ----------------------------
# FUNÇÃO PARA PEGAR DADOS
# ----------------------------
def get_crypto_data(symbol, interval='1h', days='30'):
    klines = client.get_historical_klines(f"{symbol}USDT", Client.KLINE_INTERVAL_1HOUR, f"{days} days ago UTC")
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df[["close","volume"]]
    df["sma10"] = SMAIndicator(df["close"], window=10).sma_indicator()
    df["sma30"] = SMAIndicator(df["close"], window=30).sma_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df.dropna(inplace=True)
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    df.dropna(inplace=True)
    return df

# ----------------------------
# FUNÇÃO TREINA MODELO
# ----------------------------
def train_model(df):
    X = df[["sma10","sma30","rsi","volume"]]
    y = df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
    last_row = X.iloc[-1].values.reshape(1, -1)
    pred = model.predict(last_row)[0]
    prob = model.predict_proba(last_row)[0][1]
    return pred, prob

# ----------------------------
# CRIPTOS
# ----------------------------
cryptos = ["BTC","ETH","BNB","ADA","SOL"]

st.set_page_config(page_title="Cripto AI Predictor", layout="wide")
st.title("📈 Cripto AI Predictor Avançado")
st.write("Previsão de subida ou queda na próxima hora para várias criptos simultaneamente.")

# ----------------------------
# FUNÇÃO PARA RODAR PREVISÕES
# ----------------------------
def run_predictions():
    results = []
    for crypto in cryptos:
        df = get_crypto_data(crypto)
        pred, prob = train_model(df)
        sinal = "🟢 ALTA" if pred==1 else "🔴 BAIXA"
        results.append({
            "Cripto": crypto,
            "Último Preço (USDT)": df["close"].iloc[-1],
            "Sinal": sinal,
            "Probabilidade (%)": f"{prob*100:.1f}%"
        })
    return pd.DataFrame(results)

# ----------------------------
# BOTÃO ATUALIZAR
# ----------------------------
if st.button("🔄 Atualizar Previsões"):
    df_results = run_predictions()
    st.dataframe(df_results, use_container_width=True)

# ----------------------------
# ATUALIZAÇÃO AUTOMÁTICA
# ----------------------------
st.write("Atualização automática a cada 10 minutos.")
auto_placeholder = st.empty()

for i in range(1):  # 1 ciclo inicial, depois pode usar while True localmente
    df_results = run_predictions()
    auto_placeholder.dataframe(df_results, use_container_width=True)
    time.sleep(600)  # Atualiza a cada 10 minutos
