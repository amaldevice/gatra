from __future__ import annotations

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gatra.config import Config
from gatra.tokenizer import BPETokenizer, ByteTokenizer

EPOCH_CAP = 4.0
BINS_DTYPE = np.uint16


def resolve_data_files(config: Config) -> list[Path]:
    root = config.source.parent.parent if config.source else Path.cwd()
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in config.data.sources():
        path = Path(raw)
        matches = sorted(path.parent.glob(path.name)) if any(ch in path.name for ch in "*?[") else [path]
        if len(matches) == 1 and not matches[0].exists():
            alt = root / path
            matches = sorted(alt.parent.glob(alt.name)) if any(ch in path.name for ch in "*?[") else [alt]
        files = [item.resolve() for item in matches if item.is_file()]
        if not files:
            raise FileNotFoundError(f"dataset not found: {raw}")
        for item in files:
            if item not in seen:
                seen.add(item)
                found.append(item)
    return found


def load_texts_from_file(file_path: Path) -> list[str]:
    if file_path.suffix == ".jsonl":
        records: list[str] = []
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = payload.get("text")
                if not text:
                    raise ValueError(f"missing text field in {file_path}")
                records.append(text)
        return records
    return [file_path.read_text(encoding="utf-8")]


def load_corpus(config: Config) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    texts: list[str] = []
    for file_path in resolve_data_files(config):
        docs = load_texts_from_file(file_path)
        counts[file_path.name] += len(docs)
        texts.extend(docs)
    if config.data.shuffle_docs:
        rng = random.Random(config.seed)
        rng.shuffle(texts)
    return texts, dict(counts)


class TokenDataset:
    def __init__(self, config: Config, tokenizer: ByteTokenizer | BPETokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or ByteTokenizer()
        self.texts, self.source_counts = load_corpus(config)
        encoded = self.tokenizer.encode(config.data.doc_separator.join(self.texts))
        if len(encoded) <= config.model.block_size + 1:
            raise ValueError(
                f"dataset too short: {len(encoded)} tokens, need > {config.model.block_size + 1}"
            )
        split = max(config.model.block_size + 2, int(len(encoded) * (1.0 - config.data.val_ratio)))
        split = min(split, len(encoded) - (config.model.block_size + 1))
        self.train_ids = torch.tensor(encoded[:split], dtype=torch.long)
        self.val_ids = torch.tensor(encoded[split:], dtype=torch.long)

    def get_batch(self, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_ids if split == "train" else self.val_ids
        block = self.config.model.block_size
        if len(source) <= block:
            raise ValueError(f"{split} split too short: {len(source)} tokens")
        ix = torch.randint(len(source) - block, (self.config.train.batch_size,))
        x = torch.stack([source[i : i + block] for i in ix])
        y = torch.stack([source[i + 1 : i + block + 1] for i in ix])
        return x.to(device), y.to(device)


def check_epoch_cap(source_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-source epoch counter. HARD FAILS the build if any source >4.0 epochs."""
    epochs: dict[str, float] = {}
    for name, stats in source_stats.items():
        unique = float(stats["unique_tokens"])
        epoch = float(stats["train_tokens"]) / unique if unique > 0 else float("inf")
        epochs[name] = epoch
        if epoch > EPOCH_CAP:
            raise ValueError(
                f"epoch cap exceeded for source {name!r}: {epoch:.2f} > {EPOCH_CAP} "
                f"(train_tokens={stats['train_tokens']}, unique_tokens={stats['unique_tokens']})"
            )
    return epochs


def build_bins(
    files: list[Path],
    out_dir: Path,
    tokenizer: BPETokenizer,
    name: str = "gatra-300m",
) -> dict:
    """Encode jsonl corpora to uint16 memmap bins (V<65k) + meta.json with tokenizer field."""
    if tokenizer.vocab_size > int(np.iinfo(BINS_DTYPE).max):
        raise ValueError(f"vocab {tokenizer.vocab_size} exceeds uint16 range")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    chunks: list[np.ndarray] = []
    for path in files:
        for text in load_texts_from_file(path):
            ids = tokenizer.encode(text)
            chunks.append(np.asarray(ids, dtype=BINS_DTYPE))
            total += len(ids)
    if total == 0:
        raise ValueError(f"no tokens encoded from {[str(p) for p in files]}")
    flat = np.concatenate(chunks)
    mmap = np.memmap(out_dir / "train.bin", dtype=BINS_DTYPE, mode="w+", shape=flat.shape)
    mmap[:] = flat
    mmap.flush()
    meta = {
        "tokenizer": {"name": name, "vocab": tokenizer.vocab_size, "fertility": None},
        "bins_dtype": "uint16",
        "total_tokens": int(total),
        "num_files": len(files),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_bins(
    bins_dir: Path, tokenizer: BPETokenizer | None = None
) -> tuple[torch.Tensor, dict]:
    """Load uint16 bins. Warns on tokenizer name/vocab mismatch vs meta.json."""
    meta = json.loads((bins_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("bins_dtype") != "uint16":
        raise ValueError(f"expected uint16 bins, meta says {meta.get('bins_dtype')!r}")
    if tokenizer is not None:
        saved = meta.get("tokenizer", {})
        if saved.get("vocab") not in (None, tokenizer.vocab_size):
            warnings.warn(
                f"tokenizer mismatch: bins built with vocab={saved.get('vocab')} "
                f"({saved.get('name')!r}), loading with vocab={tokenizer.vocab_size}",
                UserWarning,
                stacklevel=2,
            )
    flat = np.memmap(bins_dir / "train.bin", dtype=BINS_DTYPE, mode="r")
    return torch.from_numpy(np.asarray(flat, dtype=np.int64)), meta
