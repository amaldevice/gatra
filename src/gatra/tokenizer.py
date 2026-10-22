from __future__ import annotations

from pathlib import Path


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(id_ % 256 for id_ in ids).decode("utf-8", errors="replace")


class BPETokenizer:
    """Byte-level BPE tokenizer (HuggingFace `tokenizers`), V=32768.

    GPT-2-style: ByteLevel pre-tokenizer/decoder, no added prefix space
    (so ``decode(encode(t)) == t`` holds exactly). Special tokens
    ``<pad>``/``<bos>``/``<eos>`` are reserved for the training loop.
    """

    VOCAB_SIZE = 32768
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> BPETokenizer:
        from tokenizers import Tokenizer

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer file not found: {path}")
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def train(
        cls,
        files: list[str | Path],
        out: str | Path,
        vocab_size: int = VOCAB_SIZE,
    ) -> BPETokenizer:
        """Train byte-level BPE on jsonl (``{"text": ...}``) or .txt files."""
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        if not files:
            raise ValueError("empty training corpus: no files given")
        texts = _read_corpus_files(files)
        if not texts:
            raise ValueError("empty training corpus: no documents found")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(iter(texts), trainer=trainer)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out))
        return cls(tokenizer)


def _read_corpus_files(files: list[str | Path]) -> list[str]:
    import json

    texts: list[str] = []
    for raw in files:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"corpus file not found: {path}")
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        texts.append(json.loads(line)["text"])
        else:
            texts.append(path.read_text(encoding="utf-8"))
    return [text for text in texts if text.strip()]
