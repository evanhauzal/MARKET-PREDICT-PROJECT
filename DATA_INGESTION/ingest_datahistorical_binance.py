import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from datetime import datetime

# ==============================================================================
# 1. PENGUNDUH DATA HISTORIS (BINANCE VISION)
# ==============================================================================
def download_binance_vision_zip(url: str, col_names: list) -> pd.DataFrame:
    """Mengunduh file ZIP CSV dari Binance Vision, mengekstrak ke memory, dan me-load ke Pandas."""
    print(f"Mengunduh: {url.split('/')[-1]}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"File tidak ditemukan (404) untuk URL: {url}")
        return pd.DataFrame()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f, header=None, low_memory=False)
            
            # Deteksi otomatis: Jika baris pertama bukan angka (mengandung huruf/header), buang baris tersebut
            first_val = str(df.iloc[0, 0])
            if not first_val.replace('.', '', 1).isdigit():
                df = df.iloc[1:].reset_index(drop=True)
            
            df.columns = col_names
            return df

def fetch_historical_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Menarik dan menggabungkan Klines & Premium Index untuk bulan tertentu."""
    month_str = f"{month:02d}"
    
    kline_url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{symbol}-1m-{year}-{month_str}.zip"
    kline_cols = ["open_time", "open", "high", "low", "close", "total_volume", "close_time", 
                  "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    df_kline = download_binance_vision_zip(kline_url, kline_cols)
    
    prem_url = f"https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/{symbol}/1m/{symbol}-1m-{year}-{month_str}.zip"
    prem_cols = ["open_time", "open_prem", "high_prem", "low_prem", "close_prem", "vol_prem", 
                 "close_time_prem", "quote_vol_prem", "count_prem", "taker_base_prem", "taker_quote_prem", "ignore_prem"]
    df_prem = download_binance_vision_zip(prem_url, prem_cols)

    if df_kline.empty:
        return pd.DataFrame()

    df_kline["timestamp"] = pd.to_datetime(pd.to_numeric(df_kline["open_time"]), unit="ms", utc=True)
    df_kline.set_index("timestamp", inplace=True)
    numeric_cols = ["open", "high", "low", "close", "total_volume", "taker_buy_volume"]
    df_kline[numeric_cols] = df_kline[numeric_cols].astype(float)
    
    df_kline["sell_volume"] = (df_kline["total_volume"] - df_kline["taker_buy_volume"]).clip(lower=0.0)
    df_kline["volume_delta"] = df_kline["taker_buy_volume"] - df_kline["sell_volume"]
    
    if not df_prem.empty:
        df_prem["timestamp"] = pd.to_datetime(pd.to_numeric(df_prem["open_time"]), unit="ms", utc=True)
        df_prem.set_index("timestamp", inplace=True)
        df_kline["premium_index_bps"] = df_prem["close_prem"].astype(float) * 10000.0
    else:
        df_kline["premium_index_bps"] = np.nan

    return df_kline

# ==============================================================================
# 2. FEATURE ENGINEERING & 5-MIN TARGET (THE FINAL 13)
# ==============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Memproses Feature Engineering & Target 5-Menit (Stasioner)...")
    eps = 1e-8
    
    # 0. Patch Data Kosong Premium Index dengan Forward Fill
    df["premium_index_bps"] = df["premium_index_bps"].ffill().fillna(0.0)
    
    # 1. Geometri Candle (Hanya Lower Wick Ratio yang lolos audit)
    candle_range = (df["high"] - df["low"]) + eps
    df["lower_wick_ratio"] = (np.minimum(df["open"], df["close"]) - df["low"]) / candle_range
    
    # 2. Volatilitas (Normalized ATR dalam Basis Points)
    prev_c = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()))
    atr_1m = tr.ewm(span=14, adjust=False).mean().bfill()
    df["natr_1m_bps"] = (atr_1m / (df["close"] + eps)) * 10000.0
    
    hl_sq = np.log(df["high"] / (df["low"] + eps)) ** 2
    park_5m = np.sqrt(hl_sq.rolling(5, min_periods=2).mean() / (4.0 * np.log(2.0)))
    park_60m = np.sqrt(hl_sq.rolling(60, min_periods=10).mean() / (4.0 * np.log(2.0))) + eps
    df["volatility_ratio_5m_60m"] = (park_5m / park_60m).fillna(1.0)
    
    # 3. VWAP Z-Score 4H (Raja Fitur)
    typ_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_4h = (typ_price * df["total_volume"]).rolling(240, min_periods=15).sum() / (df["total_volume"].rolling(240, min_periods=15).sum() + eps)
    df["vwap_zscore_4h"] = ((df["close"] - vwap_4h) / (df["close"].rolling(240, min_periods=15).std() + eps)).fillna(0.0)
    
    # 4. Agresi Order Flow
    df["volume_delta_lag1"] = df["volume_delta"].shift(1).fillna(0.0)
    cvd_30m = df["volume_delta"].rolling(30, min_periods=5).sum()
    df["cvd_zscore_30m"] = ((cvd_30m - cvd_30m.rolling(60).mean()) / (cvd_30m.rolling(60).std() + eps)).fillna(0.0)
    
    # 5. Derivatif Stasioner
    df["premium_zscore_60m"] = ((df["premium_index_bps"] - df["premium_index_bps"].rolling(60).mean()) / (df["premium_index_bps"].rolling(60).std() + eps)).fillna(0.0)
    
    # 6. Siklus Waktu Trigonometri
    hour_f = df.index.hour + (df.index.minute / 60.0)
    df["hour_sin"] = np.sin(2 * np.pi * hour_f / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_f / 24.0)
    df["min_sin"] = np.sin(2 * np.pi * df.index.minute / 60.0)
    df["min_cos"] = np.cos(2 * np.pi * df.index.minute / 60.0)
    
    # 7. TARGET 5-MENIT
    future_ret_5m = np.log(df["close"].shift(-5) / df["close"])
    vol_threshold = 0.25 * (df["natr_1m_bps"] / 10000.0)
    
    df["target_return_5m"] = future_ret_5m
    df["target_signal_5m"] = np.where(
        future_ret_5m > vol_threshold, 1,
        np.where(future_ret_5m < -vol_threshold, -1, 0)
    )
    
    return df

# ==============================================================================
# 3. EKSEKUSI UTAMA & PENYIMPANAN MASSAL
# ==============================================================================
if __name__ == "__main__":
    all_raw_dfs = []
    
    start_year = 2021
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    print(f"Memulai unduhan massal dari {start_year} hingga {current_year}-{current_month:02d}...")
    
    for year in range(start_year, current_year + 1):
        for month in range(1, 13):
            if year == current_year and month > current_month:
                break
                
            try:
                monthly_df = fetch_historical_month("BTCUSDT", year, month)
                if not monthly_df.empty:
                    all_raw_dfs.append(monthly_df)
                    print(f"  -> {year}-{month:02d} sukses diunduh ({len(monthly_df):,} baris)")
                else:
                    print(f"  -> {year}-{month:02d} kosong/belum tersedia di server")
            except Exception as e:
                print(f"  -> Gagal mengunduh {year}-{month:02d}: {e}")

    if all_raw_dfs:
        print("\nMenggabungkan seluruh data bulanan...")
        combined_raw_df = pd.concat(all_raw_dfs)
        combined_raw_df = combined_raw_df[~combined_raw_df.index.duplicated(keep='last')]
        combined_raw_df.sort_index(inplace=True)
        
        # Eksekusi Feature Engineering
        features_df = build_features(combined_raw_df)
        
        # Susunan kolom PASTI sesuai dengan target akhir (20 Kolom Total)
        target_columns = [
            "open", "high", "low", "close", "vwap_zscore_4h", "min_cos", 
            "cvd_zscore_30m", "volume_delta", "natr_1m_bps", "volatility_ratio_5m_60m", 
            "premium_zscore_60m", "volume_delta_lag1", "min_sin", "hour_sin", 
            "hour_cos", "lower_wick_ratio", "premium_index_bps", 
            "target_return_5m", "target_signal_5m"
        ]
        
        final_df = features_df[target_columns].copy()
        
        # FINAL CLEANING:
        # 1. Buang 240 baris pertama (Masa warm-up untuk indikator VWAP 4H)
        # 2. Buang 5 baris terakhir (Karena target_return_5m mengambil masa depan, ujungnya pasti NaN)
        final_df = final_df.iloc[240:-5].dropna(subset=["target_return_5m"])
        
        # Konversi target menjadi integer (-1, 0, 1) agar siap masuk model Classifier
        final_df["target_signal_5m"] = final_df["target_signal_5m"].astype(int)
        
        output_file = "btc_1m_final_13_features.csv"
        final_df.to_csv(output_file, index=True, index_label="timestamp")
        
        print(f"\n[+] Selesai! Golden Dataset bersih ({len(final_df):,} baris) disimpan ke '{output_file}'.")
    else:
        print("Tidak ada data yang berhasil dikumpulkan.")