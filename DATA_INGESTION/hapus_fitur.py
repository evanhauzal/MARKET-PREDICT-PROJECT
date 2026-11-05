import os
import time
import pandas as pd

# ==============================================================================
# SELEKSI FITUR PASCA-AUDIT (ELIMINASI FITUR LEMAH / REDUNDAN)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "btc_1m_elite_12_features.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "btc_1m_golden_features.csv")

t0 = time.perf_counter()
print(f"[1/3] Membaca dataset {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print("[2/3] Menyaring hanya fitur-fitur yang lolos audit kuantitatif ketat...")

# 1. Fitur Utama yang Lolos Audit (The Golden Features):
# - vwap_zscore_4h     : 100% konsisten searah 63 bulan, IC k1 -0.048, OOS logloss gain tertinggi
# - lower_wick_ratio   : Info orisinal murni (retensi IC parsial 62.8%), buyer aggression
# - premium_zscore_60m : Sentimen futures stasioner (retensi IC parsial 56.5%, drift PSI 0.002)
# - volume_delta       : Order flow taker imbalance langsung
# - volume_delta_lag1  : Konfirmasi persistensi order flow, lolos noise OOS
# - cvd_zscore_30m     : Akumulasi order flow 30 menit (92.1% bulan searah)
# - min_sin            : Siklus menit residual yang positif di OOS walk-forward
GOLDEN_FEATURES = [
    "vwap_zscore_4h",
    "lower_wick_ratio",
    "premium_zscore_60m",
    "volume_delta",
    "volume_delta_lag1",
    "cvd_zscore_30m",
    "min_sin"
]

# 2. Fitur yang DIELIMINASI (Dihapus dari dataset ML):
# - natr_1m_bps             : Dieliminasi karena Drift PSI 0.366 (BESAR), outlier z=96.5, korelasi semu mekanis
# - hour_sin, hour_cos      : Dieliminasi karena gain logloss negatif (merusak model / memicu overfitting)
# - volatility_ratio_5m_60m : Dieliminasi karena IC mendekati nol dan tidak tahan delay
# - premium_index_bps (raw) : Dieliminasi karena redundan dengan premium_zscore_60m dan outlier z > 140

BASE_COLS = ["timestamp", "open", "high", "low", "close"]
TARGET_COLS = ["target_return_5m", "target_signal_5m"]

cols_to_keep = BASE_COLS + GOLDEN_FEATURES + TARGET_COLS

missing_cols = [c for c in cols_to_keep if c not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan di {INPUT_CSV}: {missing_cols}")

df_final = df[cols_to_keep].copy()

# Hapus baris kosong jika ada di target
df_final = df_final.dropna(subset=["target_return_5m"]).reset_index(drop=True)

print(f"[3/3] Menyimpan Golden Dataset ke {OUTPUT_CSV}...")
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] SELESAI dalam {time.perf_counter() - t0:.1f} detik!")
print(f"Dimensi Akhir: {df_final.shape[0]:,} baris x {df_final.shape[1]} kolom.")
print(f"Kolom Tersimpan: {list(df_final.columns)}")
