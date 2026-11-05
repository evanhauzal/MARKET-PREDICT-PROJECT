import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. MUAT DATASET FINAL (15 FITUR + 2 TARGET PRE-CALCULATED)
# ==============================================================================
INPUT_CSV = "btc_1m_final_15_features.csv"
print(f"[1/4] Memuat dataset dari {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# Jika timestamp terbaca sebagai kolom, pastikan masuk ke non_feature_cols
non_feature_cols = [
    "timestamp", "open", "high", "low", "close",
    "target_return_5m", "target_signal_5m"
]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

# Bersihkan nilai inf/NaN pada fitur (jika ada sisa)
X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Gunakan target yang sudah dihitung oleh skrip Data Ingestion
y_reg = df["target_return_5m"].fillna(0.0)
y_cls = df["target_signal_5m"].fillna(0).astype(int)

print(f"      Total sampel diuji : {len(X):,} baris | Jumlah fitur diuji : {len(feature_cols)} fitur\n")

# Untuk efisiensi pada jutaan baris, gunakan subsampling acak/merata
# agar Mutual Info dan Random Forest tidak memakan RAM hingga crash
SAMPLE_SIZE = min(150_000, len(X))
stride = max(1, len(X) // SAMPLE_SIZE)

X_sample = X.iloc[::stride]
y_reg_sample = y_reg.iloc[::stride]
y_cls_sample = y_cls.iloc[::stride]

# ==============================================================================
# 2. UJI 1: DETEKSI FITUR KEMBAR / MULTIKOLINEARITAS (SPEARMAN > 0.88)
# ==============================================================================
print("[2/4] Menjalankan Uji 1: Mencari fitur yang kembar / tumpang-tindih (Redundansi)...")
corr_matrix = X_sample.corr(method="spearman").abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

redundant_pairs = []
for col in upper_tri.columns:
    for row in upper_tri.index:
        val = upper_tri.loc[row, col]
        if val > 0.88:
            redundant_pairs.append((row, col, round(float(val), 4)))

# ==============================================================================
# 3. UJI 2: KORELASI RANK SPEARMAN & MUTUAL INFORMATION (NON-LINIER)
# ==============================================================================
print("[3/4] Menjalankan Uji 2: Mengukur Spearman Correlation & Mutual Information (Non-Linier)...")
spearman_with_target = X_sample.apply(lambda col: col.corr(y_reg_sample, method="spearman")).abs().fillna(0.0)

mi_scores = mutual_info_classif(X_sample, y_cls_sample, discrete_features=False, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols)

# ==============================================================================
# 4. UJI 3: WALK-FORWARD TREE IMPORTANCE (INTERAKSI ANTAR-FITUR)
# ==============================================================================
print("[4/4] Menjalankan Uji 3: Menguji pengaruh interaksi fitur di dalam Pohon Keputusan...")
split_idx = int(len(X_sample) * 0.70)
X_train, X_test = X_sample.iloc[:split_idx], X_sample.iloc[split_idx:]
y_train, y_test = y_cls_sample.iloc[:split_idx], y_cls_sample.iloc[split_idx:]

rf_auditor = RandomForestClassifier(
    n_estimators=150, max_depth=6, min_samples_leaf=15, random_state=42, n_jobs=-1
)
rf_auditor.fit(X_train, y_train)

tree_gain = pd.Series(rf_auditor.feature_importances_, index=feature_cols)

perm = permutation_importance(rf_auditor, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
perm_score = pd.Series(perm.importances_mean, index=feature_cols)

# ==============================================================================
# 5. CETAK RAPOR GABUNGAN & REKOMENDASI SELEKSI FITUR
# ==============================================================================
report = pd.DataFrame({
    "Spearman_Linier": spearman_with_target.round(4),
    "Mutual_Info_NonLinier": mi_series.round(4),
    "Tree_Interaction_Gain": tree_gain.round(4),
    "OutOfSample_Impact": perm_score.round(4),
})

norm_mi = (report["Mutual_Info_NonLinier"] / (report["Mutual_Info_NonLinier"].max() + 1e-8)) * 40
norm_gain = (report["Tree_Interaction_Gain"] / (report["Tree_Interaction_Gain"].max() + 1e-8)) * 40
norm_sp = (report["Spearman_Linier"] / (report["Spearman_Linier"].max() + 1e-8)) * 20

report["Norm_Mutual_Info"] = norm_mi.round(2)
report["Norm_Tree_Gain"] = norm_gain.round(2)
report["Norm_Spearman"] = norm_sp.round(2)
report["Total_Power_Score"] = (norm_mi + norm_gain + norm_sp).round(1)

def assign_verdict(row):
    if row["Total_Power_Score"] >= 35.0:
        return "🔥 SANGAT KUAT (Wajib Dipakai)"
    elif row["Total_Power_Score"] >= 15.0:
        return "✅ BERPENGARUH (Pendukung Interaksi)"
    else:
        return "⚠️ LEMAH / NOISE (Kandidat Dibuang)"

report["Vonis_Audit"] = report.apply(assign_verdict, axis=1)
report = report.sort_values("Total_Power_Score", ascending=False)

display_cols = [
    "Spearman_Linier", "Mutual_Info_NonLinier", "Tree_Interaction_Gain", 
    "OutOfSample_Impact", "Total_Power_Score", "Vonis_Audit"
]

print("\n" + "=" * 105)
print("📊 RAPOR AUDIT PENGARUH 15 FITUR INTI (DIURUTKAN DARI YANG PALING BERPENGARUH)")
print("=" * 105)
print(report[display_cols].to_string())

print("\n" + "=" * 105)
print("🔍 DETEKSI FITUR KEMBAR / MULTIKOLINEARITAS (KORELASI ANTAR-FITUR > 0.88)")
print("=" * 105)
if not redundant_pairs:
    print("✅ Tidak ditemukan fitur yang tumpang-tindih ekstrem. Semua fitur berdiri independen!")
else:
    print("Perhatian! Pasangan fitur di bawah ini memiliki informasi yang hampir sama persis.")
    for f1, f2, corr_val in redundant_pairs:
        s1 = report.loc[f1, "Total_Power_Score"]
        s2 = report.loc[f2, "Total_Power_Score"]
        keep = f1 if s1 >= s2 else f2
        drop = f2 if s1 >= s2 else f1
        print(f"  • [{f1}] vs [{f2}] -> Korelasi: {corr_val*100:.1f}% | Simpan: '{keep}', Hapus: '{drop}'")

report.to_csv("hasil_audit_fitur.csv")
print("\n📁 Rapor lengkap telah disimpan ke: hasil_audit_fitur.csv")

# ==============================================================================
# 6. VISUALISASI HASIL AUDIT & KORELASI
# ==============================================================================
print("\n[Visualisasi] Menyiapkan grafik visualisasi hasil korelasi & audit fitur...")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8

color_map = {
    "🔥 SANGAT KUAT (Wajib Dipakai)": "#10b981",       
    "✅ BERPENGARUH (Pendukung Interaksi)": "#0284c7", 
    "⚠️ LEMAH / NOISE (Kandidat Dibuang)": "#ef4444"    
}

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle(
    "DASHBOARD AUDIT KORELASI & KELAYAKAN FITUR PREDIKSI BTC (1M)", 
    fontsize=18, fontweight="bold", y=0.98, color="#0f172a"
)

# PANEL 1: PERINGKAT TOTAL POWER SCORE
ax1 = axes[0, 0]
df_sorted = report.sort_values("Total_Power_Score", ascending=True)
colors = [color_map.get(v, "#94a3b8") for v in df_sorted["Vonis_Audit"]]
bars = ax1.barh(df_sorted.index, df_sorted["Total_Power_Score"], color=colors, height=0.72, edgecolor="none", alpha=0.9)

for bar in bars:
    w = bar.get_width()
    ax1.text(
        w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", 
        va="center", ha="left", fontsize=8.5, fontweight="semibold", color="#334155"
    )

ax1.axvline(35.0, color="#10b981", linestyle="--", linewidth=1.4, alpha=0.8, label="Batas Sangat Kuat (>= 35)")
ax1.axvline(15.0, color="#0284c7", linestyle=":", linewidth=1.4, alpha=0.8, label="Batas Berpengaruh (>= 15)")
ax1.set_xlim(0, max(df_sorted["Total_Power_Score"]) + 12)
ax1.set_title("1. Peringkat Total Power Score (Kombinasi Linier, MI, & Tree Gain)", fontsize=13, fontweight="bold", pad=10, color="#1e293b")
ax1.set_xlabel("Total Power Score (Skala 0 - 100)", fontsize=10, fontweight="bold", color="#475569")
ax1.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.9)
ax1.grid(True, linestyle="--", alpha=0.4, axis="x")

# PANEL 2: DEKOMPOSISI METRIK FITUR UNGGULAN
ax2 = axes[0, 1]
top_n = min(12, len(report))
top_report = report.head(top_n).sort_values("Total_Power_Score", ascending=True)
y_pos = np.arange(len(top_report))
h = 0.65

ax2.barh(y_pos, top_report["Norm_Mutual_Info"], height=h, label="Mutual Info Non-Linier (Maks 40)", color="#f59e0b", alpha=0.9)
ax2.barh(y_pos, top_report["Norm_Tree_Gain"], height=h, left=top_report["Norm_Mutual_Info"], label="Tree Interaction Gain (Maks 40)", color="#8b5cf6", alpha=0.9)
ax2.barh(y_pos, top_report["Norm_Spearman"], height=h, left=top_report["Norm_Mutual_Info"] + top_report["Norm_Tree_Gain"], label="Spearman Linier (Maks 20)", color="#0284c7", alpha=0.9)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_report.index, fontsize=9.5, fontweight="semibold")
ax2.set_title(f"2. Dekomposisi Bobot Pengaruh (Top {top_n} Fitur Terbaik)", fontsize=13, fontweight="bold", pad=10, color="#1e293b")
ax2.set_xlabel("Komposisi Kontribusi Poin", fontsize=10, fontweight="bold", color="#475569")
ax2.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.9)
ax2.grid(True, linestyle="--", alpha=0.4, axis="x")

for i, total in enumerate(top_report["Total_Power_Score"]):
    ax2.text(total + 1.2, i, f"{total:.1f}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1e293b")

# PANEL 3: PEMETAAN KUADRAN
ax3 = axes[1, 0]
scatter_colors = [color_map.get(v, "#94a3b8") for v in report["Vonis_Audit"]]
sizes = (report["Total_Power_Score"] * 3.5) + 40

scatter = ax3.scatter(
    report["Spearman_Linier"], report["Mutual_Info_NonLinier"], 
    s=sizes, c=scatter_colors, alpha=0.75, edgecolors="#ffffff", linewidths=1.2
)

med_x = report["Spearman_Linier"].median()
med_y = report["Mutual_Info_NonLinier"].median()
ax3.axvline(med_x, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.7)
ax3.axhline(med_y, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.7)

top_to_label = report.head(8).index.tolist()
hidden_gems = report[report["Mutual_Info_NonLinier"] > 0.1].index.tolist()
label_candidates = list(dict.fromkeys(top_to_label + hidden_gems))

for feat in label_candidates:
    x_val = report.loc[feat, "Spearman_Linier"]
    y_val = report.loc[feat, "Mutual_Info_NonLinier"]
    ax3.annotate(
        feat, (x_val, y_val),
        xytext=(6, 6), textcoords="offset points",
        fontsize=8.5, fontweight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.8)
    )

ax3.set_title("3. Pemetaan Kuadran: Korelasi Linier (|Spearman|) vs Non-Linier (MI)", fontsize=13, fontweight="bold", pad=10, color="#1e293b")
ax3.set_xlabel("Korelasi Linier Spearman Terhadap Return Target (|r|)", fontsize=10, fontweight="bold", color="#475569")
ax3.set_ylabel("Skor Ketergantungan Non-Linier (Mutual Information)", fontsize=10, fontweight="bold", color="#475569")
ax3.grid(True, linestyle="--", alpha=0.4)

ax3.text(
    0.03, 0.95, "💎 Fitur Non-Linier Tersembunyi\n(MI Tinggi, Linier Rendah)", 
    transform=ax3.transAxes, fontsize=8.5, va="top", color="#0f766e", 
    bbox=dict(boxstyle="square,pad=0.4", facecolor="#ccfbf1", edgecolor="#2dd4bf", alpha=0.7)
)
ax3.text(
    0.97, 0.95, "⭐ Fitur Sempurna\n(Linier & Non-Linier Kuat)", 
    transform=ax3.transAxes, fontsize=8.5, va="top", ha="right", color="#1e3a8a", 
    bbox=dict(boxstyle="square,pad=0.4", facecolor="#dbeafe", edgecolor="#60a5fa", alpha=0.7)
)

# PANEL 4: HEATMAP MULTIKOLINEARITAS
ax4 = axes[1, 1]
top_feats = report.head(top_n).index.tolist()
corr_top = X_sample[top_feats].corr(method="spearman").abs()

sns.heatmap(
    corr_top, ax=ax4, annot=True, fmt=".2f", cmap="YlGnBu", 
    cbar=True, vmin=0.0, vmax=1.0, square=True,
    linewidths=0.5, linecolor="#f1f5f9",
    annot_kws={"size": 7.5, "weight": "semibold"}
)
ax4.set_title(f"4. Matriks Multikolinearitas Antar-Fitur Unggulan (Top {len(top_feats)})", fontsize=13, fontweight="bold", pad=10, color="#1e293b")
ax4.tick_params(axis="x", rotation=45, labelsize=8.5)
ax4.tick_params(axis="y", rotation=0, labelsize=8.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
dashboard_path = "dashboard_audit_fitur.png"
fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
print(f"✅ [1/2] Dashboard visualisasi komprehensif disimpan ke: {dashboard_path}")

# GRAFIK 2: HEATMAP SEGITIGA LENGKAP
fig_all, ax_all = plt.subplots(figsize=(16, 14))
mask_all = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, ax=ax_all, mask=mask_all, cmap="coolwarm", vmin=0.0, vmax=1.0,
    square=True, linewidths=0.3, linecolor="#ffffff", cbar_kws={"shrink": 0.8}
)
ax_all.set_title("MATRIKS KORELASI LENGKAP SPEARMAN (DETEKSI REDUNDANSI SELURUH FITUR)", fontsize=14, fontweight="bold", pad=12, color="#0f172a")
ax_all.tick_params(axis="x", rotation=90, labelsize=8)
ax_all.tick_params(axis="y", rotation=0, labelsize=8)

heatmap_path = "heatmap_korelasi_seluruh_fitur.png"
fig_all.savefig(heatmap_path, dpi=300, bbox_inches="tight")
print(f"✅ [2/2] Heatmap korelasi lengkap disimpan ke: {heatmap_path}")

print("\n" + "=" * 105)
print("🎉 SELURUH PROSES AUDIT DAN VISUALISASI BERHASIL!")
print("   Buka file gambar berikut untuk analisis visual:")
print(f"   1. {os.path.abspath(dashboard_path)}")
print(f"   2. {os.path.abspath(heatmap_path)}")
print("=" * 105)
print("💡 TIP: Jalankan 'python UJI_KORELASI.py --show' jika ingin memunculkan jendela interaktif matplotlib.")

if "--show" in sys.argv:
    plt.show()
else:
    plt.close("all")