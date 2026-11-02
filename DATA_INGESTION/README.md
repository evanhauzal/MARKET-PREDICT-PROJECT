# 📈 Market Predict: BTCUSDT 1M ML-Driven Trading Bot

Proyek ini adalah *pipeline* *Machine Learning* kuantitatif (berbasis XGBoost & LSTM) yang dirancang untuk memprediksi pergerakan harga 5 menit ke depan pada pasar *Futures* Bitcoin (BTCUSDT). Sistem ini menggunakan arsitektur *Asynchronous Pipeline* yang memisahkan tahap pelatihan model historis (*Batch Training*) dengan tahap eksekusi *real-time* (*Live Signal Maker*).

---

## 💾 1. Sumber Data (Data Ingestion)

Data mentah ditarik secara massal (2021–2026) menggunakan *endpoint* data arsip resmi institusi: **Binance Vision** (`data.binance.vision`). 
Data yang diunduh mencakup lebih dari 3 juta baris (resolusi 1-menit) yang terdiri dari dua komponen utama:

1. **Klines / OHLCV:** Data pergerakan harga dasar dan metrik *Order Flow* (Total Volume & Taker Buy Volume).
2. **Premium Index Klines:** Data yang merekam selisih basis antara pasar *Futures* (Perpetual) dan pasar *Spot*.

**⚠️ Eksklusi Open Interest (OI) Historis:**
Data historis Open Interest secara bawaan disediakan oleh Binance dalam resolusi 5-menit. Menginterpolasi data 5-menit menjadi 1-menit terbukti menghasilkan garis datar palsu (*fake resolution*) yang menyebabkan *Time-Overfitting* pada mesin XGBoost. Oleh karena itu, OI dihapus sepenuhnya dari dataset historis/pelatihan, dan hanya dipanggil secara *real-time* via WebSocket sebagai *Filter Logika (Gatekeeper)* saat eksekusi *live trading* untuk mendeteksi *Liquidity Sweep*.

---

## 🔬 2. Metodologi Rekayasa Fitur (Feature Engineering)

Fitur-fitur dalam dataset ini **tidak** dipilih secara acak. Fitur direkayasa dan disaring melalui **Audit Korelasi Skala Masif (3.022.319 baris)** menggunakan metode *Walk-Forward Validation* (Train: 2021-2024 | Test: 2025-2026) dengan metrik *Spearman Rank*, *Non-Linear Mutual Information (k-NN)*, dan *Permutation Importance* dari algoritma *HistGradientBoosting*.

**Prinsip Utama: 100% Stasioner (Bebas Bias Inflasi)**
Audit membuktikan bahwa fitur yang menggunakan nilai nominal Dolar (seperti ATR) atau jumlah koin mentah (seperti OI) hancur saat memprediksi data masa depan (2025-2026) karena perubahan rezim harga BTC (dari $20k ke $80k+). Oleh karena itu, seluruh fitur diubah menjadi skala stasioner (*Z-Score*, persentase, atau rasio) agar mesin hanya fokus pada momentum dan relativitas, bukan nilai nominal.

---

## 🧬 3. Kamus Fitur (The Final 13)

Dari 25+ fitur mentah, hanya 13 fitur murni (*The Final 13*) yang lulus uji kekuatan prediktif (*Out-of-Sample* positif) dan terhindar dari multikolinearitas.

### Pilar 1: Harga, VWAP & Geometri Candle
*   **`vwap_zscore_4h` (Core Alpha - Skor Tertinggi):** Deviasi harga penutupan terhadap *Volume-Weighted Average Price* (VWAP) 4 jam terakhir, dinormalisasi dengan Z-Score. 
    *   *Tujuan:* Mendeteksi batas *Mean-Reversion*. Mesin menyadari bahwa harga yang terdeviasi terlalu jauh dari rata-rata volume (Z-score > 2 atau < -2) memiliki probabilitas tinggi untuk ditarik kembali ke tengah.
*   **`lower_wick_ratio`:** Rasio panjang ekor bawah terhadap total rentang (jarak High-Low) sebuah *candle*.
    *   *Tujuan:* Mendeteksi *Liquidity Sweep* (Sapu Likuiditas) ke bawah. Rasio yang tinggi menandakan penolakan harga (*price rejection*) tiba-tiba yang sering digunakan *Market Maker* untuk memakan *stop-loss* *trader* ritel sebelum membalikkan tren ke atas.

### Pilar 2: Agresi Order Flow (Jejak Institusi)
*   **`volume_delta`:** Selisih bersih antara *Taker Buy Volume* dan *Taker Sell Volume* pada menit tersebut.
    *   *Tujuan:* Mengukur agresi langsung dari pembeli vs penjual pasar.
*   **`volume_delta_lag1`:** Nilai `volume_delta` dari 1 menit sebelumnya.
    *   *Tujuan:* Memberikan konteks momentum kepada mesin (misal: apakah agresi beli menit lalu berlanjut atau langsung diserap?).
*   **`cvd_zscore_30m`:** *Cumulative Volume Delta* 30 menit yang dinormalisasi dengan Z-Score 60 menit.
    *   *Tujuan:* Mengidentifikasi tekanan tren mikro. Menunjukkan apakah dalam 30 menit terakhir sedang terjadi akumulasi persisten atau distribusi persisten dari *Smart Money*.

### Pilar 3: Volatilitas Stasioner
*   **`natr_1m_bps`:** *Average True Range* 1-menit yang dinormalisasi terhadap harga penutupan (*Close*), diukur dalam *Basis Points* (bps).
    *   *Tujuan:* Indikator rezim volatilitas universal. Membuat model tidak buta saat harga BTC naik tajam, memastikan 10 bps di tahun 2021 dibaca sama dengan 10 bps di tahun 2026.
*   **`volatility_ratio_5m_60m`:** Rasio volatilitas Parkinson jangka sangat pendek (5 menit) dibandingkan jangka menengah (60 menit).
    *   *Tujuan:* Mendeteksi titik kompresi harga (sebelum *breakout* meledak) atau ekspansi ekstrem (sebelum harga terkoreksi).

### Pilar 4: Sentimen Derivatif & Leverage
*   **`premium_index_bps`:** Selisih mentah (dalam *basis points*) antara harga wajar *Futures* dan *Spot*.
    *   *Tujuan:* Mengukur kecenderungan pasar (optimisme vs pesimisme ritel).
*   **`premium_zscore_60m`:** Premium Index dinormalisasi dengan Z-Score rata-rata 1 jam.
    *   *Tujuan:* Mendeteksi kerentanan *Short Squeeze* atau *Long Squeeze*. Jika *trader futures* terlampau serakah/optimis dibanding sejam terakhir (Z-score sangat tinggi), bandar cenderung akan membanting harga untuk melikuidasi mereka.

### Pilar 5: Siklus Waktu Trigonometri (Cyclical Encodings)
*   **`min_sin`, `min_cos`, `hour_sin`, `hour_cos`:** Representasi melingkar (trigonometri) dari menit (0-59) dan jam (0-23) agar model ML tidak mengira bahwa menit 59 dan menit 00 berjarak jauh.
    *   *Tujuan:* Menangkap rutinitas *High-Frequency Trading* (HFT). Ledakan likuiditas selalu konsisten terjadi pada detik-detik pergantian jam, penutupan *candle* (15m/30m), dan waktu pembayaran *Funding Rate* (menit :59).

---

## 🎯 4. Target Pembelajaran (Kunci Jawaban Model)

Model tidak memprediksi harga 1 menit ke depan karena rentangnya terlalu sempit dan habis oleh biaya transaksi (*trading fee* 0.10%). 

*   **`target_return_5m`:** *Log-return* murni harga dari saat ini ke 5 menit ke depan.
*   **`target_signal_5m`:** Variabel biner klasifikasi multi-kelas untuk melatih model classifier:
    *   `1` (Naik): Jika *return* 5 menit ke depan melampaui *threshold* batas biaya dan volatilitas wajar (`0.25 * natr_1m_bps`).
    *   `-1` (Turun): Jika *return* 5 menit ke bawah merosot menembus *threshold*.
    *   `0` (Sideways): Jika harga bergerak datar di dalam *threshold* dan tidak layak untuk dieksekusi.