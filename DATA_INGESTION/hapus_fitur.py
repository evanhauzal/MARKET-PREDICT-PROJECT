import time
import pandas as pd
import numpy as np

INPUT_CSV = "btc_1m_3M_stationary_clean.csv"
OUTPUT_CSV = "btc_1m_final_13_features.csv"

t0 = time.perf_counter()
print(f"[1/3] Membaca dataset stasioner {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print("[2/3] Mengeliminasi fitur lemah dan membangun ulang target horizon 5-menit...")

# 1. Daftar 13 Fitur Inti (The Final 13) yang Lulus Audit Ketat
FINAL_FEATURES = [
    "vwap_zscore_4h", 
    "min_cos", 
    "cvd_zscore_30m", 
    "volume_delta", 
    "natr_1m_bps", 
    "volatility_ratio_5m_60m", 
    "premium_zscore_60m", 
    "volume_delta_lag1", 
    "min_sin", 
    "hour_sin", 
    "hour_cos", 
    "lower_wick_ratio", 
    "premium_index_bps"
]

# Kolom dasar yang wajib dipertahankan
BASE_COLS = ["timestamp", "open", "high", "low", "close"]

# 2. Rekonstruksi Kunci Jawaban (Target 5-Menit ke Depan)
# Menghitung return masa depan (shift -5)
future_ret_5m = np.log(df["close"].shift(-5) / df["close"])

# Menghitung batas volatilitas (25% dari Normalized ATR)
vol_threshold = 0.25 * (df["natr_1m_bps"] / 10000.0)

# Membuat klasifikasi: 1 (Naik Kuat), -1 (Turun Kuat), 0 (Sideways/Noise)
df["target_return_5m"] = future_ret_5m
df["target_signal_5m"] = np.where(
    future_ret_5m > vol_threshold, 1,
    np.where(future_ret_5m < -vol_threshold, -1, 0)
)

# 3. Seleksi kolom final dan hapus baris yang tidak memiliki masa depan (NaN di ujung data)
cols_to_keep = BASE_COLS + FINAL_FEATURES + ["target_return_5m", "target_signal_5m"]
df_final = df[cols_to_keep].copy()
df_final = df_final.dropna(subset=["target_return_5m"]).reset_index(drop=True)

print(f"[3/3] Menyimpan Golden Dataset ke {OUTPUT_CSV}...")
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"✅ SELESAI dalam {time.perf_counter() - t0:.1f} detik!")
print(f"📊 Dimensi Final: {df_final.shape[0]:,} baris x {df_final.shape[1]} kolom.")