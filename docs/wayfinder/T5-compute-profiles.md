---
title: "T5: Profil compute Spark vs cluster"
tags: [gatra, wayfinder, ticket]
wayfinder: research
status: done
resolves-to: T5-compute-profiles-RESULT.md
blocked-by: []
---

# T5: Profil compute Spark vs cluster

## Question

Kunci pembagian kerja Spark-vs-cluster berbasis angka terukur, bukan asumsi: throughput bf16 sustained GB10 untuk 300M (target ukur langsung bila Spark tersedia; bila belum, bound dari bandwidth 273 GB/s + referensi GB10 training), biaya/waktu 3T di cluster (Modal/dll), dan batas kelayakan Spark-full ~1T.

Keputusan yang harus keluar dari tiket ini:
1. Throughput terukur/ter-bound per profil (TFLOPS sustained, MFU) + waktu P0/pilot-50B/full-3T/full-1T per profil.
2. Matriks pembagian: Spark = P0 + A/B + SFT/DPO/QAT + eval; cluster = P1 full — atau revisi bila angka berkata lain.
3. Opsi Spark-full ~1T: trade-off kualitas (keluar rezim overtrain 10k) + justifikasi kapan diambil.
4. Estimasi biaya per profil (GPU-hours × tarif).

Konteks: `docs/research/gatra-300m-training-2026-10-22.md` §2 (baseline 8×A100/8×H100); spesifikasi GB10 (128GB unified, 273 GB/s, ~1 PFLOP FP4).

Unblocks: T7 (spec). Research AFK — bisa jalan paralel sekarang.

## Resolution (2026-10-22, branch work/t5-compute)

Selesai via `T5-compute-profiles-RESULT.md` (research asset, tanpa kode).

- **Bound GB10 bf16 @300M: 15–25 TFLOPS sustained (nominal 20 → ~11k tok/s).** Dari puncak bf16 62 TFLOPS (pihak-ketiga ±20%) × MFU 25–40% (prior §2 32% + PyTorch FSDP 57%/37%). Angka tok/s NVIDIA (82.739/13.520 @3B) tak konsisten dengan roofline (13.5k×6×3.2B≈260 TFLOPS > puncak) → ditolak sebagai jangkar; forum komunitas melaporkan ~8× lebih lambat dari klaim.
- **Waktu/biaya:** P0 3–5 hari di Spark ✓; pilot-50B 52 hari Spark vs 31 jam 8×A100 (~$320–530) vs 10.5 jam 8×H100 (~$330); full-3T hanya cluster (8×A100 60–90d ~$20–36k; 8×H100 20–30d ~$15–23k); full-1T 1×Spark = 2.9 thn (fisik mustahil).
- **Matrix direvisi:** Spark = P0 + 1× pilot-50B (verifikasi bound) + SFT/DPO/QAT + eval; AB-sweep 250B → cluster (6.5 hari, ~$1.6–2.6k).
- **Spark-full-1T:** DITOLAK 1 unit (butuh 233 TFLOPS untuk 90 hari > puncak 62 TFLOPS). BERSYARAT hanya jika pilot-50B ukur ≥30 TFLOPS/unit + 2 unit + budget 12 bln.
- **Gate ukur:** terima bound iff pilot-50B Spark mengukur 8k–14k tok/s sustained-5min.
