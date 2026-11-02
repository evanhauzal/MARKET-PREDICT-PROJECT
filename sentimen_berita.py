import json
import os
import re
import time
import pandas as pd
import requests
from groq import Groq

# ==========================================
# 1. KONFIGURASI API KEY GROQ & SETUP
# ==========================================

# Menggunakan Llama 3 70B yang sangat cerdas dan super cepat di Groq
MODEL_NAME = "qwen/qwen3.8-27b"

# File untuk mencatat ID berita yang sudah pernah dipakai
SEEN_IDS_FILE = "seen_tree_news_ids.txt"
OUTPUT_CSV = "btc_realtime_clean_sentiment.csv"

# Inisialisasi client Groq
client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# 2. FILTER LOKAL & DEDUPLIKASI CERDAS
# ==========================================
BTC_AND_MACRO_KEYWORDS = [
    r"\bbitcoin\b", r"\bbtc\b", r"\bsatoshi\b", r"\bhalving\b", r"\bibit\b",
    r"\bfbtc\b", r"\bblackrock\b", r"\bfidelity\b", r"\bgrayscale\b",
    r"\bmicrostrategy\b", r"\bsaylor\b", r"\bmt\.?\s?gox\b", r"\bsilk road\b",
    r"\bfed\b", r"\bfomc\b", r"\bpowell\b", r"\bcpi\b", r"\binflation\b",
    r"\bpce\b", r"\bnfp\b", r"\brate cut\b", r"\brate hike\b", r"\bsec\b",
    r"\bcftc\b", r"\btreasury\b", r"\busdt\b", r"\btether\b", r"\bbinance\b",
    r"\bcoinbase\b", r"\bwar\b", r"\bmissile\b", r"\bevacuation\b",
    r"\bsanction\b", r"\btariff\b", r"\bhack\b", r"\bexploit\b",
    r"\bbankrupt\b", r"\bliquidation\b", r"\betf\b", r"\binflow\b", r"\boutflow\b",
]

LOCAL_EXCLUSIONS = [
    r"\bmemecoin\b", r"\blibra\b", r"\bpepe\b", r"\bdoge\b", r"\bshib\b",
    r"\bsolana\b", r"\bcardano\b", r"\bnft\b", r"\bairdrop\b",
    r"\btokenized stock\b", r"quiet week", r"what charts are we watching",
    r"good morning", r"\bgm\b", r"locks in", r"lock in",
]


def normalize_headline(text: str) -> str:
    """Membersihkan awalan sumber/handle Twitter agar judul kembar terdeteksi sama."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^[^:]{2,40}:\s*", "", text)
    text = re.sub(r"^[^:]{2,25}:\s*", "", text)
    clean = re.sub(r"[^a-z0-9\s]", "", text.lower())
    clean = re.sub(r"\blost\b", "lose", clean)
    words = clean.split()
    return " ".join(words[:8])


def passes_prefilter(text: str) -> bool:
    text_lower = text.lower()
    has_btc_directly = bool(re.search(r"\b(bitcoin|btc)\b", text_lower))

    if any(re.search(kw, text_lower) for kw in LOCAL_EXCLUSIONS) and not has_btc_directly:
        return False
    if re.search(r"\$(eth|sol|xrp|bnb|doge|pepe|avax|sui)\b", text_lower) and not has_btc_directly:
        return False

    return any(re.search(kw, text_lower) for kw in BTC_AND_MACRO_KEYWORDS)


# ==========================================
# 3. FUNGSI GROQ DENGAN JSON MODE
# ==========================================
def analyze_ultra_fast_groq(news_batch: list[dict]) -> dict[int, dict]:
    if not news_batch:
        return {}

    formatted_items = "\n".join(
        [f"[{item['id']}] {item['title']}" for item in news_batch]
    )

    # Catatan: Prompt Groq JSON Mode harus menghasilkan object (dict) dengan key tertentu
    prompt = f"""Anda adalah Quantitative BTC Futures Analyst (timeframe 1m).
Analisis daftar berita berikut dan kembalikan HANYA format JSON object dengan key "items".

Format JSON persis seperti ini:
{{
  "items": [
    {{"id": 0, "t": "BREAKING", "d": -0.8, "r": 1.0, "u": 1.0, "note": "alasan maks 4 kata"}}
  ]
}}

Keterangan nilai:
- "t": Pilih "BREAKING" (kejutan baru), "REACTIVE" (harga/likuidasi sudah terjadi), "FLUFF" (opini/basa-basi), atau "ALTCOIN" (koin selain BTC).
- "d": Direction BTC dari -1.0 s/d 1.0 (0.0 jika FLUFF/ALTCOIN/netral).
- "r": Relevansi BTC: 1.0 (BTC langsung), 0.7 (Makro AS/Fed/Perang/ETF), 0.0 (Tidak relevan).
- "u": Urgensi candle 1m: 1.0 (Breaking darurat), 0.3 (Berita REACTIVE terlambat), 0.0 (FLUFF/ALTCOIN).

Daftar Berita:
{formatted_items}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction bot. Output only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL_NAME,
            temperature=0.0,
            response_format={"type": "json_object"}, # Memaksa Groq mengeluarkan JSON bersih
        )

        raw_text = response.choices[0].message.content
        parsed_json = json.loads(raw_text)
        parsed_list = parsed_json.get("items", [])

        lookup = {}
        for res in parsed_list:
            score = round(float(res["d"]) * float(res["r"]) * float(res["u"]), 4)
            lookup[int(res["id"])] = {
                "news_type": res["t"],
                "direction": float(res["d"]),
                "btc_relevance": float(res["r"]),
                "urgency_1m": float(res["u"]),
                "final_catalyst_score": score,
                "reasoning": res.get("note", ""),
            }
        return lookup

    except Exception as e:
        print(f"[!] Error saat memanggil Groq API: {e}")
        return {}


# ==========================================
# 4. EKSEKUSI PIPELINE
# ==========================================
seen_ids = set()
if os.path.exists(SEEN_IDS_FILE):
    with open(SEEN_IDS_FILE, "r", encoding="utf-8") as f:
        seen_ids = set(line.strip() for line in f if line.strip())

print("[1/3] Menarik berita real-time murni dari Tree of Alpha API...")
try:
    raw_api_news = requests.get("https://news.treeofalpha.com/api/news?limit=60", timeout=10).json()
except Exception as e:
    print(f"Gagal mengambil Tree of Alpha: {e}")
    raw_api_news = []

fresh_candidates = []
seen_fingerprints = set()
new_ids_to_save = []

for item in raw_api_news:
    news_uid = str(item.get("_id") or item.get("time") or "")
    if not news_uid or news_uid in seen_ids:
        continue

    text = (item.get("title") or item.get("body") or "").replace("\n", " ").strip()
    if not text:
        continue

    new_ids_to_save.append(news_uid)

    fingerprint = normalize_headline(text)
    if fingerprint in seen_fingerprints:
        continue
    seen_fingerprints.add(fingerprint)

    ts = pd.to_datetime(item.get("time"), unit="ms", utc=True, errors="coerce")
    fresh_candidates.append({
        "news_uid": news_uid,
        "timestamp": ts,
        "source": item.get("source", "TreeNews"),
        "title": text,
    })

if not fresh_candidates:
    print("\n[INFO] Tidak ada berita baru sejak penarikan terakhir.")
    exit()

to_send_to_llm = []
final_records = []

for idx, item in enumerate(fresh_candidates):
    if passes_prefilter(item["title"]):
        to_send_to_llm.append({"id": idx, "title": item["title"][:280]})
        final_records.append({"id": idx, **item, "passed_regex": True})
    else:
        final_records.append({
            "id": idx,
            **item,
            "passed_regex": False,
            "news_type": "LOCAL_FILTERED",
            "direction": 0.0,
            "btc_relevance": 0.0,
            "urgency_1m": 0.0,
            "final_catalyst_score": 0.0,
            "reasoning": "Dibuang filter lokal (Non-BTC/Altcoin/Basa-basi).",
        })

print(f"[2/3] Ditemukan {len(fresh_candidates)} berita unik baru. {len(to_send_to_llm)} berita lolos ke LLM.")

if to_send_to_llm:
    print(f"[3/3] Memanggil Groq API ({MODEL_NAME})...")
    start_time = time.perf_counter()
    llm_results = analyze_ultra_fast_groq(to_send_to_llm)
    elapsed = time.perf_counter() - start_time
    print(f"\n⚡ WAKTU PROSES GROQ: {elapsed:.2f} detik untuk {len(to_send_to_llm)} berita!\n")

    for rec in final_records:
        if rec["passed_regex"] and rec["id"] in llm_results:
            rec.update(llm_results[rec["id"]])

with open(SEEN_IDS_FILE, "a", encoding="utf-8") as f:
    for uid in new_ids_to_save:
        f.write(f"{uid}\n")

df_out = pd.DataFrame(final_records)
df_out.to_csv(OUTPUT_CSV, index=False)

print("=" * 100)
print("=== HASIL BERITA REAL-TIME BARU (MURNI DARI SUMBER & LOLOS FILTER LOKAL) ===")
print("=" * 100)
analyzed_only = [r for r in final_records if r["passed_regex"]]
if not analyzed_only:
    print("Semua berita baru saat ini adalah berita Altcoin/Non-BTC dan sudah dibuang oleh filter lokal (Skor 0.0).")
else:
    for rec in analyzed_only:
        print(f"[{rec['timestamp']}] ({rec['source']}) {rec['title'][:95]}")
        print(f"   -> Tipe: {rec.get('news_type')} | Dir: {rec.get('direction')} | Rel: {rec.get('btc_relevance')} | Urg: {rec.get('urgency_1m')} | SKOR AKHIR: {rec.get('final_catalyst_score')} ({rec.get('reasoning')})")
        print("-" * 100)