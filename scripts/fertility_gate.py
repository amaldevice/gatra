"""Fertility gate (T1): fail if >threshold tokens/word on ID samples.

Usage:
    python scripts/fertility_gate.py --tokenizer out/tokenizer.json \
        --samples data/wiki-id.sample.jsonl data/web-id.sample.jsonl \
        [--threshold 1.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gatra.tokenizer import BPETokenizer


def fertility(encode: Callable[[str], list[int]], texts: list[str]) -> float:
    total_tokens = sum(len(encode(text)) for text in texts)
    total_words = sum(len(text.split()) for text in texts)
    if total_words == 0:
        raise ValueError("no words in fertility samples")
    return total_tokens / total_words


def gate(value: float, threshold: float = 1.5) -> bool:
    return value <= threshold


def load_sample_texts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    texts.append(json.loads(line)["text"])
                else:
                    texts.append(line)
    return texts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fertility gate: fail if tok/word > threshold")
    parser.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    parser.add_argument("--samples", nargs="+", required=True, help="sample .jsonl/.txt files")
    parser.add_argument("--threshold", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tokenizer = BPETokenizer.from_file(args.tokenizer)
    texts = load_sample_texts([Path(p) for p in args.samples])
    value = fertility(tokenizer.encode, texts)
    status = "PASS" if gate(value, args.threshold) else "FAIL"
    print(f"fertility={value:.3f} tok/word threshold={args.threshold} [{status}]")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
