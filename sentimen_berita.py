import json
import os
import re
import time
import pandas as pd
import requests
from groq import Groq
from dotenv import load_dotenv
                                                   
load_dotenv(os.path.join(os.path.dirname(__file__), "DATA_INGESTION", ".env"))

# ==========================================
# 1. KONFIGURASI API KEY GROQ & SETUP
# ==========================================

# Menggunakan Llama 3 70B yang sangat cerdas dan super cepat di Groq
MODEL_NAME = "qwen/qwen3.8-27b"

# File untuk mencatat ID berita yang sudah pernah dipakai
SEEN_IDS_FILE = "seen_tree_news_ids.txt"
OUTPUT_CSV = "btc_realtime_clean_sentiment.csv"

# Inisialisasi client Groq
client = Groq()

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
    r"\b(?:bitcoin|btc|xbt|satoshi|satoshis)\b", r"\b(?:ibit|fbtc|gbtc|arkb|bitb|btcusd|btcusdt|xbtusd)\b",
    r"\b(?:microstrategy|mstr|saylor)\b", r"\bstrategy\s+(?:inc\.?|incorporated|bitcoin|btc)\b",
    r"\bmt\.?\s*gox\b", r"\bsilk\s+road\b", r"\b(?:bitcoin|btc)\s+(?:core|cash|sv)\b",
    r"\bfederal\s+reserve\b", r"\b(?:jerome\s+)?powell\b", r"\b(?:core\s+|headline\s+)?(?:cpi|pce|ppi)\b",
    r"\bconsumer\s+price\s+(?:index|indices)\b", r"\bproducer\s+price\s+(?:index|indices)\b",
    r"\bpersonal\s+consumption\s+expenditures\b", r"\b(?:u\.?s\.?|united\s+states)\s+(?:core\s+)?inflation\b",
    r"\b(?:nfp|non[-\s]?farm\s+payrolls?|nonfarm\s+employment)\b",
    r"\b(?:initial|continuing|weekly)\s+(?:jobless|unemployment)\s+claims\b",
    r"\b(?:jolts|job\s+openings|adp\s+(?:employment|payrolls?|jobs))\b",
    r"\b(?:unemployment\s+rate|average\s+hourly\s+earnings|employment\s+cost\s+index)\b",
    r"\b(?:ism|s&p\s+global|s\s+and\s+p\s+global)\s+(?:manufacturing\s+|services\s+|composite\s+)?pmi\b",
    r"\bism\s+(?:manufacturing|services)\b",
    r"\b(?:purchasing\s+managers'?\s+index|retail\s+sales|durable\s+goods\s+orders)\b",
    r"\b(?:consumer\s+confidence|michigan\s+consumer\s+sentiment|inflation\s+expectations)\b",
    r"\b(?:u\.?s\.?|united\s+states)\s+(?:gdp|economic\s+growth)\b", r"\bgross\s+domestic\s+product\b",
    r"\b(?:fed\s+funds|federal\s+funds|interest\s+rate)\s+(?:target|rate|decision|cuts?|hikes?|futures)\b",
    r"\b(?:rate\s+(?:cuts?|hikes?|hold|decision)|dot\s+plot|fed\s+minutes|fomc\s+minutes)\b",
    r"\b(?:quantitative\s+easing|quantitative\s+tightening|balance[-\s]sheet\s+runoff)\b",
    r"\b(?:on\s+rrp|overnight\s+reverse\s+repo|reverse\s+repurchase|standing\s+repo\s+facility)\b",
    r"\b(?:treasury\s+general\s+account|treasury\s+refunding|treasury\s+auction)\b",
    r"\b(?:iorb|sofr|effective\s+federal\s+funds\s+rate)\b",
    r"\b(?:u\.?s\.?\s+)?(?:2|10|30)[-\s]year\s+(?:treasury\s+)?(?:yields?|bonds?)\b",
    r"\b(?:dxy|dollar\s+index|u\.?s\.?\s+dollar\s+index|yield\s+curve)\b",
    r"\b(?:debt\s+ceiling|u\.?s\.?\s+(?:government\s+shutdown|debt\s+default))\b",
    r"\b(?:bank\s+of\s+japan|boj|ecb|bank\s+of\s+england|pboc)\b.{0,100}\b(?:rates?|yields?|stimulus|liquidity|intervention|tightening|easing)\b",
    r"\b(?:yen\s+carry\s+trade|carry[-\s]trade\s+unwind)\b",
    r"\b(?:bitcoin\s+mining|mining\s+difficulty|hash\s*rate|hashprice|block\s+subsidy)\b",
    r"\b(?:bitcoin\s+core|bitcoin\s+network|bitcoin\s+nodes?|bitcoin\s+fork)\b",
    r"\b(?:miner\s+(?:capitulation|reserves?|outflows?|selling)|mining\s+ban)\b",
    r"\b(?:bitcoin|btc)\b.{0,80}\b(?:reorg|double[-\s]spend|vulnerability|bug|outage|congestion)\b",
    r"\b(?:lightning\s+network|bitcoin\s+improvement\s+proposal|taproot|segwit)\b",
    r"\b(?:spot\s+bitcoin\s+(?:etfs?|etps?)|bitcoin\s+trust)\b",
    r"\b(?:exchange[-\s]traded\s+(?:funds?|products?))\b",
    r"\b(?:net\s+(?:inflows?|outflows?)|fund\s+flows?|in[-\s]kind\s+(?:creations?|redemptions?))\b",
    r"\b(?:authorized\s+participants?|cash\s+creations?|cash\s+redemptions?)\b",
    r"\b(?:blackrock|fidelity|grayscale|bitwise|vaneck|wisdomtree|invesco|franklin\s+templeton|21shares)\b",
    r"\b(?:bitcoin|btc|crypto)\b.{0,80}\b(?:pension\s+fund|sovereign\s+wealth|endowment|custody|allocation)\b",
    r"\b(?:hodl|brrr|btco|btcw|ezbc|btcc)\b.{0,60}\b(?:etf|fund|trust|inflows?|outflows?)\b",
    r"\b(?:strategic\s+bitcoin\s+reserve|bitcoin\s+reserve|bitcoin\s+treasury)\b",
    r"\b(?:bitcoin|btc)\b.{0,100}\b(?:purchases?|acquir(?:e|es|ed|ing)|holdings?|sales?|sell(?:s|ing)?|seized|seizures?|forfeiture|auction|repayments?)\b",
    r"\b(?:government|doj|department\s+of\s+justice|u\.?s\.?\s+marshals)\b.{0,100}\b(?:bitcoin|btc|crypto\s+assets?)\b",
    r"\b(?:metaplanet|semler\s+scientific|bitcoin\s+treasury\s+compan(?:y|ies))\b",
    r"\b(?:ftx|celsius|blockfi|genesis)\b.{0,100}\b(?:creditors?|repayments?|distributions?|bankrupt(?:cy)?|liquidations?)\b",
    r"\b(?:usdc|usdt|tether)\b", r"\b(?:stablecoins?|stablecoin\s+reserves?|proof\s+of\s+reserves)\b",
    r"\b(?:de[-\s]?peg(?:s|ged|ging)?|redemption\s+(?:halt|pause|suspension)|reserve\s+shortfall)\b",
    r"\b(?:circle|paxos)\b.{0,80}\b(?:usdc|stablecoins?|reserves?|redemptions?)\b",
    r"\b(?:bank\s+run|banking\s+crisis|bank\s+failure|deposit\s+flight|liquidity\s+crisis|credit\s+crunch)\b",
    r"\b(?:silvergate|signature\s+bank|silicon\s+valley\s+bank)\b",
    r"\b(?:bybit|okx|kraken|bitfinex|bitstamp|gemini|deribit)\b",
    r"\b(?:withdrawals?|deposits?)\s+(?:halted|paused|suspended|disabled|resumed|suspension)\b",
    r"\b(?:halt(?:s|ed)?|paus(?:e|es|ed)|suspend(?:s|ed)?)\s+(?:all\s+)?(?:withdrawals?|deposits?)\b",
    r"\b(?:security\s+breach|private\s+key\s+compromise|exchange\s+outage|exchange\s+insolvency)\b",
    r"\b(?:hacked|hacking|exploited|exploits|bankruptcy|bankruptcies|liquidations)\b",
    r"\b(?:bitcoin|btc|crypto)\b.{0,100}\b(?:funding\s+rates?|open\s+interest|futures\s+basis|options?\s+expir(?:y|ation)|implied\s+volatility|put[-/]call|short\s+squeeze|long\s+squeeze)\b",
    r"\b(?:cme|chicago\s+mercantile\s+exchange|deribit)\b.{0,100}\b(?:bitcoin|btc|crypto)\b",
    r"\b(?:bitcoin|btc)\b.{0,80}\b(?:longs?|shorts?)\s+liquidated\b",
    r"\b(?:sec|cftc)\b.{0,100}\b(?:crypto|bitcoin|btc|etfs?|custody|exchanges?|securities|derivatives)\b",
    r"\b(?:crypto|bitcoin|btc|stablecoin)\b.{0,100}\b(?:bills?|legislation|executive\s+orders?|regulations?|ban(?:ned|s)?|lawsuits?|settlements?|approvals?|rejections?)\b",
    r"\b(?:market\s+structure\s+bill|stablecoin\s+bill|digital\s+asset\s+legislation)\b",
    r"\b(?:mica|micar)\b.{0,80}\b(?:crypto|stablecoins?|exchanges?|regulation)\b",
    r"\b(?:wars?|missiles?|sanctions?|tariffs?)\b",
    r"\b(?:ceasefire|airstrikes?|ballistic\s+missiles?|military\s+strikes?|state\s+of\s+emergency)\b",
    r"\b(?:strait\s+of\s+hormuz|oil\s+embargo|nuclear\s+escalation|trade\s+war)\b",
]

LOCAL_EXCLUSIONS = [
    r"\bmemecoin\b", r"\blibra\b", r"\bpepe\b", r"\bdoge\b", r"\bshib\b",
    r"\bsolana\b", r"\bcardano\b", r"\bnft\b", r"\bairdrop\b",
    r"\btokenized stock\b", r"quiet week", r"what charts are we watching",
    r"good morning", r"\bgm\b", r"locks in", r"lock in",
    r"\bmeme[-\s]?coins?\b", r"\b(?:pepe|doge|shib|bonk|floki|wif|brett)\b",
    r"\b(?:solana|cardano|ethereum|avalanche|polygon|chainlink)\b",
    r"\b(?:nfts?|airdrops?|giveaways?|whitelists?|presales?)\b",
    r"\b(?:pump\.fun|token\s+launch|token\s+unlock|meme\s+token)\b",
    r"\b(?:sponsored\s+post|paid\s+promotion|referral\s+code|join\s+our\s+telegram)\b",
    r"\b(?:daily\s+recap|weekly\s+roundup|price\s+prediction|technical\s+analysis)\b",
    r"\b(?:good\s+morning|good\s+night|happy\s+weekend)\b",
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