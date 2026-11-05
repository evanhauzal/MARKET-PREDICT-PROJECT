import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PENGUNDUH DATA HISTORIS (BINANCE VISION)
# ==============================================================================
def download_binance_vision_zip(url: str, col_names: list) -> pd.DataFrame:
    """Mengunduh file ZIP CSV dari Binance Vision, mengekstrak ke memory, dan me-load ke Pandas."""
    print(f"Mengunduh: {url.split('/')[-1]}...")
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        print(f"File tidak ditemukan (404) untuk URL: {url}")
        return pd.DataFrame()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f, header=None, low_memory=False)
            
            # Deteksi otomatis header
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
# 2. FEATURE ENGINEERING (PASANGAN FITUR LENGKAP + TARGET 3-KELAS ANTI-NOISE)
# ==============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Memproses Feature Engineering (Pasangan Simetris Utuh & Target 3-Kelas Cost-Aware)...")
    eps = 1e-8
    
    df["premium_index_bps"] = df["premium_index_bps"].ffill().fillna(0.0)
    log_ret_1m = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    
    # --------------------------------------------------------------------------
    # 1. GEOMETRI CANDLE BERPASANGAN (Mendeteksi Bullish & Bearish Sweep Seimbang)
    # --------------------------------------------------------------------------
    candle_range = (df["high"] - df["low"]) + eps
    df["lower_wick_ratio"] = (np.minimum(df["open"], df["close"]) - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - np.maximum(df["open"], df["close"])) / candle_range
    
    # Gabungan simetris (-1.0 = Bearish Upper Sweep, +1.0 = Bullish Lower Sweep)
    df["wick_imbalance"] = df["lower_wick_ratio"] - df["upper_wick_ratio"]
    # Close Location Value (-1.0 = Tutup di Low, +1.0 = Tutup di High)
    df["clv_imbalance"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / candle_range
    
    # --------------------------------------------------------------------------
    # 2. VOLATILITAS STASIONER (Parkinson Ratio & NATR Basis Points)
    # --------------------------------------------------------------------------
    prev_c = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()))
    atr_1m = tr.ewm(span=14, adjust=False).mean().bfill()
    df["natr_1m_bps"] = (atr_1m / (df["close"] + eps)) * 10000.0
    
    hl_sq = np.log(df["high"] / (df["low"] + eps)) ** 2
    park_5m = np.sqrt(hl_sq.rolling(5, min_periods=2).mean() / (4.0 * np.log(2.0)))
    park_60m = np.sqrt(hl_sq.rolling(60, min_periods=10).mean() / (4.0 * np.log(2.0))) + eps
    df["volatility_ratio_5m_60m"] = (park_5m / park_60m).fillna(1.0)
    
    # --------------------------------------------------------------------------
    # 3. VWAP Z-SCORE 4H (Core Mean-Reversion Anchor)
    # --------------------------------------------------------------------------
    typ_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_4h = (typ_price * df["total_volume"]).rolling(240, min_periods=15).sum() / (
        df["total_volume"].rolling(240, min_periods=15).sum() + eps
    )
    df["vwap_zscore_4h"] = ((df["close"] - vwap_4h) / (df["close"].rolling(240, min_periods=15).std() + eps)).fillna(0.0)
    
    # --------------------------------------------------------------------------
    # 4. AGRESI ORDER FLOW + FITUR ORTOGONAL (Bebas Bias Return Masa Lalu)
    # --------------------------------------------------------------------------
    df["volume_delta_lag1"] = df["volume_delta"].shift(1).fillna(0.0)
    
    # Rasio Taker Imbalance Stasioner (-1.0 s/d +1.0, bebas bias jumlah BTC tahun 2021 vs 2026)
    df["taker_imbalance_ratio"] = (df["volume_delta"] / (df["total_volume"] + eps)).clip(-1.0, 1.0)
    df["taker_imbalance_lag1"] = df["taker_imbalance_ratio"].shift(1).fillna(0.0)
    
    cvd_30m = df["volume_delta"].rolling(30, min_periods=5).sum()
    df["cvd_zscore_30m"] = ((cvd_30m - cvd_30m.rolling(60, min_periods=10).mean()) / (cvd_30m.rolling(60, min_periods=10).std() + eps)).fillna(0.0)
    
    # Flow-Price Divergence (Orderbook Absorption Murni: Lolos uji info_di_luar_return_lalu!)
    vd_z = (df["volume_delta"] - df["volume_delta"].rolling(60, min_periods=10).mean()) / (df["volume_delta"].rolling(60, min_periods=10).std() + eps)
    ret_z = (log_ret_1m - log_ret_1m.rolling(60, min_periods=10).mean()) / (log_ret_1m.rolling(60, min_periods=10).std() + eps)
    df["flow_price_divergence"] = (vd_z - ret_z).fillna(0.0).clip(-5.0, 5.0)
    
    # --------------------------------------------------------------------------
    # 5. DERIVATIF STASIONER (Basis Perpetual vs Spot)
    # --------------------------------------------------------------------------
    df["premium_zscore_60m"] = (
        (df["premium_index_bps"] - df["premium_index_bps"].rolling(60, min_periods=10).mean())
        / (df["premium_index_bps"].rolling(60, min_periods=10).std() + eps)
    ).fillna(0.0)
    
    # --------------------------------------------------------------------------
    # 6. SIKLUS WAKTU TRIGONOMETRI BERPASANGAN (Wajib Sin + Cos Bersama!)
    # --------------------------------------------------------------------------
    minute = df.index.minute
    hour_float = df.index.hour + (minute / 60.0)
    
    df["min_sin"] = np.sin(2 * np.pi * minute / 60.0)
    df["min_cos"] = np.cos(2 * np.pi * minute / 60.0)
    df["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    
    # --------------------------------------------------------------------------
    # 7. TARGET 5-MENIT AKURAT (ASYMMETRIC PATH-DEPENDENT TRIPLE-BARRIER)
    # --------------------------------------------------------------------------
    HORIZON = 5
    MIN_TP_BPS = 0.0012  # Minimal TP = 12 bps (0.12%) -> Bersih di atas fee 8-10 bps
    MIN_SL_BPS = 0.0010  # Minimal SL = 10 bps (0.10%) -> Ruang napas anti-noise

    # Gunakan harga Open menit berikutnya (t+1) sebagai harga eksekusi nyata
    entry_prices = np.roll(df["open"].values, -1)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # Ekspektasi volatilitas 5 menit berdasarkan NATR 1m
    vol_5m = (df["natr_1m_bps"].values / 10000.0) * np.sqrt(HORIZON)

    # Jarak TP & Jarak SL
    tp_dist = np.maximum(MIN_TP_BPS, 0.75 * vol_5m)
    sl_dist = np.maximum(MIN_SL_BPS, 0.55 * vol_5m)

    # Level harga untuk skenario LONG & SHORT
    tp_long_level = entry_prices * (1.0 + tp_dist)
    sl_long_level = entry_prices * (1.0 - sl_dist)
    tp_short_level = entry_prices * (1.0 - tp_dist)
    sl_short_level = entry_prices * (1.0 + sl_dist)

    n_rows = len(df)
    long_tp_step = np.full(n_rows, 99, dtype=np.int8)
    long_sl_step = np.full(n_rows, 99, dtype=np.int8)
    short_tp_step = np.full(n_rows, 99, dtype=np.int8)
    short_sl_step = np.full(n_rows, 99, dtype=np.int8)

    for step in range(1, HORIZON + 1):
        fut_high = np.roll(highs, -step)
        fut_low = np.roll(lows, -step)

        # Cek jalur LONG
        long_tp_step[(fut_high >= tp_long_level) & (long_tp_step == 99)] = step
        long_sl_step[(fut_low <= sl_long_level) & (long_sl_step == 99)] = step

        # Cek jalur SHORT
        short_tp_step[(fut_low <= tp_short_level) & (short_tp_step == 99)] = step
        short_sl_step[(fut_high >= sl_short_level) & (short_sl_step == 99)] = step

    # Validasi kemenangan murni (TP tersentuh duluan sebelum SL dalam <= 5 menit)
    long_win = (long_tp_step < long_sl_step) & (long_tp_step <= HORIZON)
    short_win = (short_tp_step < short_sl_step) & (short_tp_step <= HORIZON)

    # Menentukan sinyal final (jika terjadi anomali keduanya True, pilih yang step-nya lebih kecil)
    long_only = long_win & (~short_win | (long_tp_step <= short_tp_step))
    short_only = short_win & (~long_win | (short_tp_step < long_tp_step))

    df["target_signal_5m"] = np.where(long_only, 1, np.where(short_only, -1, 0)).astype(np.int8)

    # Realized Return 5m
    raw_ret_5m = np.log(np.roll(closes, -HORIZON) / (entry_prices + 1e-8))
    df["target_return_5m"] = np.where(
        long_only, tp_dist,
        np.where(short_only, -tp_dist, raw_ret_5m)
    )

    # Bersihkan efek wrap-around np.roll pada HORIZON baris terakhir
    df.iloc[-HORIZON:, df.columns.get_loc("target_return_5m")] = np.nan
    df.iloc[-HORIZON:, df.columns.get_loc("target_signal_5m")] = 0

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
        
        # Susunan Kolom Lengkap (Pasangan Fitur Utuh + Target 3 Kelas)
        target_columns = [
            "open",
            "high",
            "low",
            "close",
            # Pilar 1: Harga, VWAP & Geometri Candle Berpasangan
            "vwap_zscore_4h",
            "lower_wick_ratio",
            "upper_wick_ratio",
            "wick_imbalance",
            "clv_imbalance",
            # Pilar 2: Order Flow & Divergensi Ortogonal
            "volume_delta",
            "volume_delta_lag1",
            "taker_imbalance_ratio",
            "taker_imbalance_lag1",
            "cvd_zscore_30m",
            "flow_price_divergence",
            # Pilar 3: Derivatif Basis Spot-Perp
            "premium_index_bps",
            "premium_zscore_60m",
            # Pilar 4: Volatilitas & Siklus Waktu Berpasangan (Sin + Cos)
            "volatility_ratio_5m_60m",
            "natr_1m_bps",
            "min_sin",
            "min_cos",
            "hour_sin",
            "hour_cos",
            # Target Label (Y)
            "target_return_5m",
            "target_signal_5m"
        ]
        
        final_df = features_df[target_columns].copy()
        
        # FINAL CLEANING:
        final_df = final_df.iloc[240:-5].dropna(subset=["target_return_5m"])
        final_df["target_signal_5m"] = final_df["target_signal_5m"].astype(np.int8)
        
        # Cetak distribusi kelas target (-1, 0, 1) untuk memastikan kelas 0 (WAIT) terbentuk dengan benar
        class_counts = final_df["target_signal_5m"].value_counts(normalize=True) * 100
        print("\nDistribusi Kelas Target 5-Menit (Cost-Aware 3-Kelas):")
        for cls_val, pct in class_counts.items():
            label_name = "WAIT (Noise/Sideways)" if cls_val == 0 else ("LONG (+1)" if cls_val == 1 else "SHORT (-1)")
            print(f"  Kelas {cls_val:2d} [{label_name:21s}] : {pct:.2f}% dari total data")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(base_dir, "btc_1m_golden_features.csv")
        final_df.to_csv(output_file, index=True, index_label="timestamp")
        
        print(f"\n[+] Selesai! Golden Dataset bersih ({len(final_df):,} baris) disimpan ke '{output_file}'.")
        print(f"[+] Daftar Kolom ({len(target_columns) + 1}): {['timestamp'] + target_columns}")
    else:
        print("Tidak ada data yang berhasil dikumpulkan.")