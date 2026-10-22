---
title: "T5 Result: Compute Profiles Spark vs Cluster (300M, bound terukur)"
tags: [gatra, wayfinder, ticket, compute]
status: done
resolves: T5-compute-profiles
date: 2026-10-22
---

# T5 Result: Profil Compute Spark vs Cluster

Resolves [[T5-compute-profiles]] (research asset, no code — arithmetic checks only).
Baseline compute: `docs/research/gatra-300m-training-2026-10-22.md` §2 (FLOPs $6ND$, 5.4e21 pretrain, 100 TFLOPS/A100, 300 TFLOPS/H100).
Keputusan MAP Q6: Spark = pilot/post-train/inferensi; cluster = full 3T; opsi Spark-full hanya ~1T.

Model: N = 302M. FLOPs/token training $\approx 6N = 1.81\times10^9$. Konversi: tok/s = TFLOPS$\times10^{12} / 6N$.

## 1) Bound GB10 bf16 sustained untuk 300M: 15–25 TFLOPS

**Atap hardware (resmi NVIDIA):**

> "Memory Bandwidth | 273 GB/s" … "Up to 1 PFLOP FP4" … "128 GB LPDDR5x, coherent unified system memory"
> — [NVIDIA DGX Spark specs](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) + footnote "Theoretical FP4 TOPS using the sparsity feature." (ibid.)

FP4 1 PFLOP = inference/quant only (dengan sparsity); **tidak berlaku untuk training bf16**.

**Puncak bf16 (pihak-ketiga, belum diverifikasi NVIDIA):**

> "BF16 62.05 TFLOPS Tensor BF16 62.05 TFLOPS" … "Memory Bandwidth 273.1GB/s" … "48 SMs" / "6,144 Shaders"
> — [HMC NVIDIA GB10 Full Specs](https://hmc-tech.com/gpus/nvidia-gb10) `[THIRD-PARTY — treat as ±20%]`

Konsisten dengan aritmetika inti: $6144 \times 2 \times 2.52\text{GHz} \approx 31$ TFLOPS FP32, bf16 $\approx 2\times$.

**Angka training publikasi NVIDIA (tidak konsisten — jangan jadikan jangkar):**

> "In full fine-tuning of a Llama 3.2B model, we reached a peak of 82,739.2 tokens per second."
> — [NVIDIA Technical Blog: DGX Spark Performance](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)

Tabel yang sama menulis `13,519.54` tok/s untuk run identik (Llama 3.2 3B full-FT, seq 2048, batch 8), dengan rumus "Peak tokens/sec calculated as (batch_size \* steps \* sequence_length)/ Total time to train." (ibid.). Keduanya tidak dapat dipakai: 13.5k tok/s @3.2B $\Rightarrow 13{,}519 \times 6 \times 3.2{\times}10^9 \approx 260$ TFLOPS — **4× di atas puncak bf16 62 TFLOPS**. Metodologi tak dapat direkonsiliasi dengan roofline; kemungkinan presisi campuran FP8 / penghitungan non-standar, tidak dinyatakan.

**Reproduksi komunitas (peringatan ke bawah):**

> "instead of the ~80k tokens per second throughput, i get ~11k tokens per second on the 3B"
> — [NVIDIA Developer Forums: "DGX Spark PyTorch LLM training throughput up to 8x slower than expected"](https://forums.developer.nvidia.com/t/dgx-spark-pytorch-llm-training-throughput-up-to-8x-slower-than-expected/357880)

Bahkan 11k tok/s @3B masih mengimplikasikan >200 TFLOPS — memperkuat kesimpulan: **angka tok/s Spark yang beredar tidak konsisten dengan roofline bf16; bound harus dari first principles.**

**Derivasi bound (roofline + prior MFU):**

Prior MFU model-kecil single-device: baseline §2 memakai ~32% (100/312 TFLOPS A100). Referensi model-besar: "a rapid training speed of 3,700 tokens/sec/GPU, or 40B tokens/day on 128 A100 GPUs… model FLOPS utilization (MFU) and hardware FLOPS utilization (HFU) of 57%" ([PyTorch FSDP blog](https://pytorch.org/blog/maximizing-training/); tabel yang sama: H100 7B MFU 37%). Model 300M single-device lebih rendah (kernel kecil $d{=}896$, 48 SM tak jenuh penuh pada micro-batch wajar; tanpa comm-overlap tapi juga tanpa comm-loss) → MFU 25–40%.

$$62 \times 0.25\text{–}0.40 = \textbf{15–25 TFLOPS sustained (nominal 20)}$$

$$\text{tok/s}_{300M} = 20{\times}10^{12} / 1.81{\times}10^9 \approx \textbf{11.000 tok/s (rentang 8.300–13.800)}}$$

Kelayakan memori: training bf16 @300M ≈ bobot 0.6 + grad 0.6 + AdamW fp32 3.6 + aktivasi (~9 GB @micro 8×2048) ≈ **~15 GB** — muat longgar di 128 GB unified; micro-batch besar dimungkinkan, mendukung ujung atas MFU. Bandwidth 273 GB/s bersama CPU/OS adalah hambatan utama pada batch kecil (rezim bandwidth-sensitive sesuai tiket).

**Gate pengukuran langsung (bila Spark tersedia):** jalankan pilot-50B, ukur tok/s sustained-5min; terima bound ini iff terukur 8k–14k tok/s. Di luar itu → revisi T5.

## 2) Waktu + biaya per profil

FLOPs: P0 $7.7{\times}10^{18}$ (32B @40M) · 50B $9.1{\times}10^{19}$ · 250B $4.5{\times}10^{20}$ · 3T $5.44{\times}10^{21}$ · 1T $1.81{\times}10^{21}$.
Throughput: Spark 20 TFLOPS (1 unit; P0 @40M ≈ 25 TFLOPS → ~104k tok/s) · 2×Spark 1.9× (sinkron grad bf16 0.6 GB/step via 10GbE ≈ 0.5 s vs ~90–180 s compute/step → overhead <1%) · 8×A100 800 TFLOPS · 8×H100 2400 TFLOPS (baseline §2).

Tarif (resmi, Okt 2026, on-demand): Modal — "Nvidia A100, 40 GB $0.000583/sec" (=$2.10/h), "A100, 80 GB $0.000694/sec" (=$2.50/h), "H100 SXM5 $0.001097/sec" (=$3.95/h) ([Modal pricing](https://modal.com/pricing)). RunPod Pods: "H100 SXM $3.99/hr", "A100 SXM $1.79/hr" ([RunPod pricing](https://www.runpod.io/pricing)). Lambda: "8x NVIDIA H100 SXM … $27.92/hr", "8x NVIDIA A100 SXM 40 GB … $10.32/hr" (=$1.29/GPU-h), "1x NVIDIA A100 SXM 40 GB … $1.29/hr" ([Lambda pricing](https://lambda.ai/pricing)). Spark = milik sendiri → marginal ≈ listrik ~240W ≈ **$1–2/hari** (sunk $3.999/unit).

| Beban | 1× Spark (20T) | 2× Spark (1.9×) | 8×A100-40G | 8×H100 |
|---|---|---|---|---|
| P0 proxy 32B | **3–5 hari**, ~$5 | 2–3 hari | 2.7 jam, ~$45 (Modal) | ~1 jam, ~$40 |
| pilot-50B | **52 hari**, ~$75 | ~27 hari | 31 jam, **~$530** (Modal) / ~$320 (Lambda) | 10.5 jam, ~$330 |
| pilot-250B (AB total) | 262 hari (8.6 bln) ✗ | 4.3 bln ✗ | **6.5 hari, ~$2.6k** (Modal) / ~$1.6k (Lambda) | 2.2 hari, ~$1.7k |
| full-3T | 8.6 thn ✗✗ | 4.3 thn ✗✗ | **60–90 hari, ~$24–36k** (Modal) / ~$20k (Lambda 40G) | **20–30 hari, ~$15–23k** (Modal) / ~$18k (Lambda) |
| full-1T | 2.9 thn ✗✗ | 1.4 thn ✗ | 26 hari, ~$10.6k (Modal) | 8.7 hari, ~$6.6k (Modal) |
| SFT+DPO ~10B | **~10–14 hari**, ~$20 ✓ | ~1 mgg | jam-an, ~$50 | jam-an, ~$50 |
| QAT 5k steps | **~3 mgg**, ~$40 ✓ | ~10 hari | ~12 jam, ~$200 | ~4 jam, ~$130 |

✗ = tidak layak wall-clock. §2 (§2 baseline 625d 1×A100 / 60–90d 8×A100 / 20–30d 8×H100) terkonfirmasi konsisten dalam ±5% oleh aritmetika di atas.

## 3) Decision matrix

| Keputusan | Hasil |
|---|---|
| Spark = P0 + A/B + SFT/DPO/QAT + eval | **DIREVISI SEBAGIAN.** Spark = P0 (3–5 hari) ✓ + **satu** pilot-50B verifikasi-bound ✓ + SFT/DPO (~2 mgg) ✓ + QAT (~3 mgg) ✓ + eval ✓. **AB-sweep (250B total, 8.6 bln serial) → cluster** (6.5 hari 8×A100, ~$1.6–2.6k). Pengecualian: AB2 fraksional 4×25B boleh di Spark hanya bila timeline AFK ≥4 bln dan 2 unit paralel (~7 mgg). |
| Cluster = full pretrain | **DIKONFIRMASI.** 3T hanya layak di cluster: 8×A100 60–90 hari / 8×H100 20–30 hari. Pilihan hemat: Lambda 8×A100-40G (~$20k) atau Modal H100 (~$15–23k, tercepat). |
| Opsi Spark-full ~1T | **DITOLAK untuk 1 unit; BERSYARAT untuk 2 unit.** Syarat tak terbantahkan: 1T dalam 90 hari butuh 233 TFLOPS sustained > puncak 62 TFLOPS → **infeasible secara fisik pada 1× Spark apapun hasil ukur.** Ambang terukur satu-satunya yang membuka opsi: pilot-50B di Spark mengukur **≥30 TFLOPS sustained** (ujung atas bound) **DAN** ≥2 unit tersedia **DAN** timeline ≥12 bln (2×Spark ≈ 1.4 thn @20T → ~1 thn @30T). Trade-off kualitas bila diambil: 1T @300M = 3.333 tok/param vs 3T = 10.000 tok/param — keluar rezim overtrain-10k (keduanya masih deep-overtrain vs Chinchilla 20 tok/param; rujuk Sardana/Chinchilla di baseline §6); ekspektasi degradasi mengikuti kurva data-constrained, bukan cliff. |

## 4) Batas kelayakan Spark-full-1T (kondisi terukur)

Ambang kuantitatif (diturunkan, bukan asumsi): $1.81{\times}10^{21} / (T \times 10^{12} \times 86400) \le$ hari-target.
1×Spark butuh 58 TFLOPS untuk 1 thn (93% MFU — mustahil), 233 TFLOPS untuk 90 hari (di atas puncak — mustahil).
**Satu-satunya kondisi penerimaan: ukur pilot-50B ≥30 TFLOPS sustained per unit + 2 unit + budget 12 bln.** Gagal satu → cluster-full, tanpa banding.
