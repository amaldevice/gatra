from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATASET = "uonlp/CulturaX"
DEFAULT_LANG = "id"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 5000
DEFAULT_MIN_CHARS = 80
DEFAULT_OUT = Path("data/culturax-id.jsonl")

# T2 data-audit registry. `expected_unique_tokens` = estimated BPE tokens
# post-dedup (ESTIMATED from dataset cards/papers — see ticket T2
# Resolution; measure with the T1 tokenizer at build time before locking
# the mix). Existing ID sources keep their implicit entries via --dataset.
SOURCES: dict[str, dict] = {
    "culturax-id": {
        "dataset": "uonlp/CulturaX",
        "config": "id",
        "expected_unique_tokens": 12_000_000_000,
    },
    "fineweb2-id": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "ind_Latn",
        "expected_unique_tokens": 90_000_000_000,
    },
    "madlad-id": {
        "dataset": "allenai/MADLAD-400",
        "config": "id",
        "split": "train_clean",
        "expected_unique_tokens": 40_000_000_000,
    },
    "leipzig-id": {
        "dataset": None,  # raw Leipzig files (ind_mixed_2013 + news); local-only
        "config": None,
        "expected_unique_tokens": 1_500_000_000,
    },
    "stack-edu": {
        "dataset": "HuggingFaceTB/stack-edu",
        "config": None,
        "expected_unique_tokens": 125_000_000_000,
    },
    "finemath": {
        "dataset": "HuggingFaceTB/finemath",
        "config": None,
        "expected_unique_tokens": 54_000_000_000,
    },
    "infimm-webmath": {
        "dataset": "HuggingFaceTB/infimm-webmath",
        "config": None,
        "expected_unique_tokens": 40_000_000_000,
    },
    "cosmopedia-v2": {
        "dataset": "HuggingFaceTB/cosmopedia",
        "config": None,
        "expected_unique_tokens": 30_000_000_000,
    },
    "fineweb-edu": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": None,
        "expected_unique_tokens": 1_300_000_000_000,
    },
    "dclm": {
        "dataset": "mlfoundations/dclm-baseline-1.0",
        "config": None,
        "expected_unique_tokens": 3_800_000_000_000,
    },
    "cosmo-id": {
        "dataset": None,  # produced locally: NLLB-3.3B translation of cosmopedia-v2 slice
        "config": None,
        "expected_unique_tokens": 20_000_000_000,
    },
}


def _row_text(row: dict) -> str:
    text = row.get("text")
    if text is None:
        return ""
    return str(text).strip()


def iter_rows(dataset: str, lang: str, split: str, token: str | None = None):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    return load_dataset(dataset, lang, **kwargs)


def write_jsonl(
    rows,
    out_path: Path,
    limit: int,
    min_chars: int,
    score_fn=None,
) -> dict:
    """Stream rows to jsonl. With `score_fn(text) -> dict`, writes
    `{"text", "scores"}` so thresholds stay tunable without re-stream."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = _row_text(row)
            if len(text) < min_chars:
                skipped += 1
                continue
            record: dict = {"text": text}
            if score_fn is not None:
                record["scores"] = dict(score_fn(text))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= limit:
                break
    if kept == 0:
        raise ValueError(f"no documents kept from stream; check auth, language, and min_chars={min_chars}")
    return {"path": str(out_path), "kept": kept, "skipped_short": skipped}


def filter_scored_jsonl(
    src: Path,
    out_path: Path,
    score_key: str,
    threshold: float,
) -> dict:
    """Filter a scored jsonl on `scores[score_key] >= threshold` (no re-stream)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with src.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if float(record.get("scores", {}).get(score_key, float("-inf"))) < threshold:
                dropped += 1
                continue
            fout.write(json.dumps({"text": record["text"]}, ensure_ascii=False) + "\n")
            kept += 1
    return {"path": str(out_path), "kept": kept, "dropped_below_threshold": dropped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream CulturaX Indonesian documents into jsonl")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--lang", default=DEFAULT_LANG, help="CulturaX config name, e.g. id")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max documents to keep")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token", help="Hugging Face token; otherwise uses cached login")
    parser.add_argument("--source", choices=sorted(SOURCES), help="stream a registry source instead of --dataset/--lang")
    parser.add_argument("--edu-classifier", default=None, help="optional edu-classifier model path; scores docs (threshold tuned later via filter_scored_jsonl)")
    parser.add_argument("--kenlm-id", default=None, help="optional KenLM wiki-id model path for perplexity scores")
    parser.add_argument("--fasttext-lid", default=None, help="optional FastText LID model path (enforce id>=0.5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"streaming {args.dataset} lang={args.lang} split={args.split}")
    print("gated dataset: accept the license and run `huggingface-cli login` first")
    stats = write_jsonl(
        iter_rows(args.dataset, args.lang, args.split, token=args.token),
        args.out,
        limit=args.limit,
        min_chars=args.min_chars,
    )
    print(f"wrote {stats['kept']} docs to {stats['path']} (skipped_short={stats['skipped_short']})")


if __name__ == "__main__":
    main()
