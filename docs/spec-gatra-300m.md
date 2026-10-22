---
title: "Spec Gatra-300M: Indonesia-First 300M Dense Transformer for Edge (2026-10-22)"
tags: [gatra, spec, architecture, pretraining, indonesian-lm, edge-deployment]
---

# Spec Gatra-300M (siap eksekusi)

> [!tip] Terkait (Graph)
> - [[pretrain_llms/gatra/docs/wayfinder/MAP-gatra-300m-spec|Wayfinder map]] — 7 keputusan terkunci, T1–T6 resolved.
> - [[pretrain_llms/gatra/docs/research/gatra-300m-arch-2026-10-22|Arch 300M]] · [[pretrain_llms/gatra/docs/research/gatra-300m-data-2026-10-22|Data 300M]] · [[pretrain_llms/gatra/docs/research/gatra-300m-training-2026-10-22|Training 300M]] · [[pretrain_llms/gatra/docs/research/gatra-300m-alternatives-2026-10-22|Alternatif]]
> - [[pretrain_llms/gatra/docs/research/pretrain-best-practices-sub500m-2026-10-22|Best practices]] · [[pretrain_llms/gatra/docs/research/edge-deployment-quant-runtime-2026-10-22|Deployment]] · [[pretrain_llms/gatra/docs/research/gatra-repo-audit-2026-10-22|Repo audit]]

Satu dokumen, tidak ada keputusan tersisa. Pelaksana membaca ini + `configs/gatra-300m.toml` dan mulai bekerja.

---

## 1. Konfigurasi arsitektur (kunci)

Dense decoder-only, pre-RMSNorm, SwiGLU, RoPE, GQA, tied embeddings. Terverifikasi: `param_count()` = **299.885.504** (emb 31.457.280 = 10,49%; body 268.427.264).

| Knob | Nilai |
|---|---|
| Layers $L$ | 28 |
| $d_{model}$ | 960 |
| Q / KV heads | 15 / 3 (GQA 5:1, $d_{head}=64$) |
| $d_{ff}$ | 2560 (SwiGLU 2,67×) |
| Vocab | BPE 32.768, tied |
| RoPE θ | **500.000** dari step-0 (amendemen T6: 16K-ready, tanpa retune) |
| Konteks latih | 4096 (kurikulum §5) |
| Norm | pre-RMSNorm eps 1e-6 + **QK-norm ON** |
| z-loss / soft-cap | OFF (kabel ada, nyalakan saat instabilitas) |
| Sliding window / sink | NO (full attention; varian SWA 5:1+W=1024 hanya 16K, §5) |
| Bias | none; dropout 0.0 |
| KV bf16 | 21,0 KiB/tok → 84 MiB @4K / 168 MiB @8K / 336 MiB @16K |
| Q4_K | ≈168,7 MB; decode ceiling @30 GB/s: ~119 / ~89 / ~59 tok/s @4K/8K/16K (batas atas) |

Keputusan terkunci (tidak dibuka lagi): arch-note/V32k atas training-note/V49k — embedding tax 10,5% vs 14,6%; alternatif (SWA-default, Mamba, conv-hybrid, hybrid-head, MLA, linear, sparse) ditolak sebagai base.

---

## 2. Tokenizer + fertility

- **BPETokenizer V32.768** ID-first (`src/gatra/tokenizer.py`; ByteTokenizer dipertahankan untuk config lama). Latih: `scripts/train_tokenizer_300m.py` (korpus §3.3 data-note: 70% web-ID HQ + EN/code, bukan Aya-MT).
- **Gate G0: fertility ≤1,5 tok/word** di wiki-id + web-id (`scripts/fertility_gate.py`); preseden Aksara wiki-ID 1,357. Byte baseline repo: ~6,5 tok/word.

---

## 3. Data: mix per profil + karakteristik per komponen

Prinsip: single-stage dari step-0; **tidak ada sumber >4,0 epoch** (hard-fail di build); unik-ID terukur ~100–150B vs butuh 750B (3T) — defisit diisi EN/code/math/sintetik.

### Profil A — full 3T (cluster), `[data.mix]`

| Komponen | Share | Token | Unik | Epoch | Karakteristik |
|---|---|---:|---:|---:|---|
| ID web (`id_web`) | 15% | 450B | ~130B | 3,5 | FW2 ind_Latn (~80–100B, tulang punggung) + MADLAD-id (~30–50B) + CulturaX-id (~12B) + KoPI/news/Leipzig/wiki. Porsi ID maksimum aman dalam cap-4. |
| EN web (`en_web`) | 52% | 1.560B | ~5,1T | <1 | FineWeb-Edu/DCLM 60/40. Replay anti-lupa; transfer EN→ID terbukti (Cendol, Blevins). |
| Code | 15% | 450B | 125B | 3,6 | Stack-Edu (threshold 3/5). Menolong penalaran umum, bukan cuma coding. |
| Math | 10% | 300B | ~94B | 3,2 | FineMath (~54B) + InfiMM-WebMath (~40B). Fokus deduksi step-by-step. |
| Sintetik | 7% | 210B | ~50B | ≤4,0 | Cosmopedia-v2 (30B EN) + terjemahan-ID NLLB (~20B, difilter balik FastText-id ≥0,9 + edu ≥3). |
| Instruksi | 1% | 30B | ~1B | — | Aya-id/Cendol (maks 0,3% pretrain; sisanya SFT). |

### Profil B — Spark-full ~1T, `[data.mix_spark]`

30 / 35 / 15 / 12 / 7 / 1 (ID 30% = 300B atas ~150B unik ≈ 2 epoch — ID-heavy, EN dipangkas; keluar rezim overtrain 10k, trade-off tercatat di T5).

### Gate 3T-vs-2T (otomatis, bukan diskusi)

`gatra-build --audit` hitung unik/sumber. **≥750B → 3T jalan. <750B → potong ke 2T** (ID 10% ≈ 3,3 epoch; 6,7k tok/param, masih rezim Sardana). <40B unik-ID → mode CPT (EN penuh → continued-pretrain ID-heavy ala Sailor2).

### Pipeline (diimplementasi T2)

FastText-LID ≥0,5 → edu-classifier ≥3 → KenLM wiki-id (buang tercile-3); exact-hash → MinHash LSH (112/14×8, 75%); bins uint16 + `meta.json` tokenizer; `mix` dari toml; epoch-cap hard-fail.

---

## 4. Recipe training P0–P4

| Fase | Isi | Hyperparam kunci |
|---|---|---|
| P0 proxy μP | ~40M, sweep LR {2,3,4}e-3 + init + wd, ~4B/konfig | Transfer 0-shot ke 300M (biaya ~7% total) |
| P1 pretrain | 3T, single-stage, **WSD + 20% decay** | Peak **3e-3** (dari P0), warmup 2k step (~4B token, dihitung dalam token), batch **2M**, AdamW (0,9;0,95) wd 0,1 clip 1,0, dropout 0,0, **bf16 + DDP**, seq 2048 bulk → 4096 tail |
| P2 distilasi | A/B 50B sebelum commit penuh | Default **MiniPLM data-side** (2,4×); logit-KD hanya bila menang A/B + tokenizer sama; Qwen2.5 ditolak (pajak vocab 45%) |
| P3 post-train | Staged, **NO RL** | SFT 2 epoch (SmolTalk-filter + Aya-id + Cendol-Instruct/Chat + NusaWrites) LR 3e-4 → reasoning-SFT 1 epoch → DPO 2 epoch LR 1e-6 β=0,5. Prioritas **IFEval > MATH** (capacity floor ~300M) |
| P4 QAT+export | 5k step ala Gemma 3 | Target KD dari checkpoint bf16; per-block int4 → GGUF Q4_K_M |

Stability kit: QK-norm ON, z-loss 1e-4 ON, soft-cap ON; spike diantisipasi (ride out, bukan rewind-and-skip).

---

## 5. Kurikulum konteks (T6, kunci)

**2048 bulk (~2,6T) → 4096 (~300B) → 8K (~80B) → 16K (~20B).** θ=500k dari step-0. SWA 5:1+W=1024 **hanya 16K** + gate recall-ID; 4K/8K full attention.

Sumber dokumen panjang ID: `ipfs_indonesia_laws` (105K UU), `ID_REG` (250rb+ dok hukum), `indonesia-regulation` (PERDA/POJK markdown) — `[data.mix_long]`: id_law_long 35 / id_web 10 / en_long 35 / code 10 / math 10. Packing + EN-panjang sebagai fallback.

---

## 6. Profil compute (T5, kunci)

Bound GB10: **15–25 TFLOPS** sustained → full-3T di 1× Spark = **8,6 tahun**; pilot-50B ~1–2 minggu; full-3T cluster 8×A100 ~60–90 hari / 8×H100 ~20–30 hari.

**Spark = P0 + A/B + SFT/DPO/QAT + eval. Cluster = P1 full 3T.** Opsi Spark-full hanya dengan budget ~1T (profil B).

---

## 7. Gates G0–G6 + harness

| Gate | Sinyal | Ambang |
|---|---|---|
| G0 | Fertility ID | ≤1,5 (byte 6,5) |
| G1 | μP sweep bersih | tanpa divergen; RankMe tidak collapse |
| G2 | Pilot 50B | ppl dalam prediksi scaling ±10%; spike irrecoverable = stop |
| G3 | Audit data | ≥750B unik (3T) else fallback §3; **NO-GO full run bila gagal** |
| G4 | A/B distilasi | adopsi bila >2pp ID-eval atau >3% ppl |
| G5 | Post-train | IndoMMLU + NusaX-senti + IFEval-ID monoton vs base; IFEval > MATH |
| G6 | QAT+export | ppl <3% degr, downstream <2pp, **≥30 tok/s sustained + RSS <600MB @4K** |

Harness: `lm-eval-harness` (hellaswag_id, IndoMMLU, NusaX-senti, IFEval-ID) + fertility + tok/s + RSS sebagai merge-gate.

---

## 8. Post-training ID + distilasi

SFT: Aya-id kurasi + Cendol-Instruct/Chat + NusaWrites-budaya (filter kapasitas ala SmolLM2-small: tanpa function-calling/hard). DPO: UltraFeedback-style + pasangan ID (RLAIF bila perlu). Distilasi: MiniPLM default; logit-KD kondisional (T4/P2). **No RL** (SFT 74,0 vs RL 57,0 @950M).

---

## 9. A/B plan (maks 3, ≤250B, <8% budget)

1. **AB1 tokenizer/KD** (2×50B): adopsi-49k vs own-32k → fertility + ppl + delta downstream; 49k menang + buka KD → adopsi, else 32k.
2. **AB2 mix** (4×25B): ID% {5,15} × code/math {off,on} → ID-eval tertinggi tanpa regresi EN.
3. **AB3 distilasi** (≤50B, di arm menang): none vs MiniPLM vs logit-KD → adopsi bila >2pp / >3% ppl.

---

## 10. Risiko + fallback

| Risiko | Mitigasi |
|---|---|
| Unik-ID <750B | Gate G3: 2T → CPT (aturan angka §3) |
| Dok ID panjang kurang | Packing + EN-panjang; 16K jadi varian SWA |
| Spike instabilitas | Stability kit + ride-out; z-loss/soft-cap siap |
| Spark-full dipaksa | Profil B ~1T + justifikasi trade-off (§6) |
| KD gagal | Skip (preseden MobileLLM negatif); compute → reasoning-SFT |
| QAT >3% degr | Ulangi 5k step dengan data campuran pretrain+post; else PTQ + catat delta |

---

## Referensi (primer, diakses 2026-10-22)

Arsitektur/training/data/deployment: 8 catatan `docs/research/` (kutip+sitasi per klaim). Keputusan peta: `docs/wayfinder/MAP-gatra-300m-spec.md` + resolusi T1–T6.
