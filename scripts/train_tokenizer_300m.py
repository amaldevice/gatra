"""Train the Gatra-300M BPE tokenizer (T1).

Corpus recipe follows docs/research/gatra-300m-data-2026-10-22.md §3.3:
SmolLM2 proportions (70% web-Edu / 15% textbook / 8% math / 5% code /
2% Q&A) with the EN share replaced by ID-first sources.

Full-corpus training streams from HF (needs `datasets` + network);
when those corpora are unavailable, pass local .jsonl files via
--corpora as proof-of-pipeline (see ticket T1).

Usage:
    python scripts/train_tokenizer_300m.py --out out/tokenizer.json
    python scripts/train_tokenizer_300m.py --corpora data/a.jsonl data/b.jsonl \\
        --out out/tokenizer.json --vocab-size 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gatra.tokenizer import BPETokenizer

# (name, HF dataset, config/lang, weight). Weights sum to 1.0.
TOKENIZER_CORPUS_RECIPE = [
    {"name": "web-id", "dataset": "uonlp/CulturaX", "config": "id", "weight": 0.55},
    {"name": "web-id-hq", "dataset": "HuggingFaceFW/fineweb-2", "config": "ind_Latn", "weight": 0.15},
    {"name": "wiki-id", "dataset": "wikimedia/wikipedia", "config": "20231101.id", "weight": 0.10},
    {"name": "textbook-id", "dataset": "HuggingFaceTB/cosmopedia", "config": "default", "weight": 0.05},
    {"name": "math", "dataset": "HuggingFaceTB/finemath", "config": "finemath-4plus", "weight": 0.08},
    {"name": "code", "dataset": "bigcode/starcoderdata", "config": "default", "weight": 0.05},
    {"name": "qa", "dataset": "facebook/natural_reasoning", "config": "default", "weight": 0.02},
]

DEFAULT_OUT = Path("out/tokenizer.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gatra-300M BPE-32k tokenizer")
    parser.add_argument("--corpora", nargs="*", default=[], help="local .jsonl/.txt files (fallback)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--vocab-size", type=int, default=BPETokenizer.VOCAB_SIZE)
    parser.add_argument("--limit-per-source", type=int, default=200_000)
    return parser.parse_args(argv)


def stream_recipe_texts(limit_per_source: int):
    """Yield texts from the §3.3 recipe via HF streaming (needs `datasets`)."""
    from datasets import load_dataset

    for entry in TOKENIZER_CORPUS_RECIPE:
        stream = load_dataset(
            entry["dataset"], entry["config"], split="train", streaming=True
        )
        for i, row in enumerate(stream):
            if i >= limit_per_source:
                break
            text = (row.get("text") or "").strip()
            if text:
                yield text


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    out = Path(args.out)
    if args.corpora:
        files = [Path(p) for p in args.corpora]
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"corpus files not found: {missing}")
        print(f"training on {len(files)} local file(s) (proof-of-pipeline; full §3.3 corpus is follow-up)")
        BPETokenizer.train(files, out, vocab_size=args.vocab_size)
    else:
        import json
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="gatra-tok-")) / "recipe.jsonl"
        kept = 0
        with tmp.open("w", encoding="utf-8") as handle:
            for text in stream_recipe_texts(args.limit_per_source):
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                kept += 1
        print(f"streamed {kept} docs from §3.3 recipe")
        BPETokenizer.train([tmp], out, vocab_size=args.vocab_size)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
