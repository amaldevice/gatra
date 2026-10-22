"""Tests for BPETokenizer (T1): byte-level BPE, V=32768, alongside ByteTokenizer."""

import json

import pytest

from gatra.tokenizer import ByteTokenizer

SAMPLE_ID = "Pagi itu, Gatra belajar merangkai kata."
LONG_ID = " ".join(
    [
        "Pagi itu matahari terbit di balik pegunungan dan udara terasa sejuk",
        "Raka berjalan ke pasar untuk membeli nasi sayur dan buah segar",
        "Mereka berbincang tentang rencana belajar bersama sore nanti",
    ]
)


def _train_tiny(tmp_path, vocab_size: int = 512):
    from gatra.tokenizer import BPETokenizer

    corpus = tmp_path / "corpus.jsonl"
    # Enough varied docs that `vocab_size` merges are actually reachable.
    subjects = ["Raka", "Siti", "Gatra", "Budi", "Ayah", "Ibu", "Guru", "Petani"]
    verbs = ["membaca", "menulis", "membeli", "menanam", "memasak", "menjual"]
    objects = ["buku", "nasi", "sayur", "mangga", "padi", "ikan", "kain", "kopi"]
    places = ["pasar", "sawah", "sekolah", "rumah", "desa", "kota", "pantai"]
    texts = [
        f"{s} {v} {o} di {p} pada pagi yang cerah"
        for s in subjects
        for v in verbs
        for o in objects
        for p in places
    ]
    corpus.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts))
    out = tmp_path / "tok.json"
    BPETokenizer.train([corpus], out, vocab_size=vocab_size)
    return BPETokenizer.from_file(out)


def test_bpe_importable_alongside_byte() -> None:
    from gatra.tokenizer import BPETokenizer

    assert ByteTokenizer().vocab_size == 256
    assert BPETokenizer.VOCAB_SIZE == 32768


def test_bpe_roundtrip(tmp_path) -> None:
    tok = _train_tiny(tmp_path)
    assert tok.decode(tok.encode(SAMPLE_ID)) == SAMPLE_ID


def test_bpe_vocab_size(tmp_path) -> None:
    # vocab_size is a cap: merges stop when no pair meets min_frequency.
    tok = _train_tiny(tmp_path, vocab_size=512)
    assert tok.vocab_size <= 512
    # Full byte alphabet (256) + specials must always be representable.
    assert tok.vocab_size >= 256 + 4


def test_bpe_fertility_better_than_byte_on_id_fixture(tmp_path) -> None:
    tok = _train_tiny(tmp_path)
    byte_tok = ByteTokenizer()
    words = len(SAMPLE_ID.split())
    byte_fert = len(byte_tok.encode(SAMPLE_ID)) / words
    bpe_fert = len(tok.encode(SAMPLE_ID)) / words
    assert bpe_fert < byte_fert


def test_bpe_train_rejects_empty_corpus(tmp_path) -> None:
    from gatra.tokenizer import BPETokenizer

    with pytest.raises(ValueError, match="empty"):
        BPETokenizer.train([], tmp_path / "tok.json")


def test_bpe_missing_file_raises(tmp_path) -> None:
    from gatra.tokenizer import BPETokenizer

    with pytest.raises(FileNotFoundError):
        BPETokenizer.from_file(tmp_path / "nope.json")


def test_fertility_gate_pass_and_fail(tmp_path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fertility_gate", "scripts/fertility_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fertility, gate = mod.fertility, mod.gate

    texts = ["Pagi itu Gatra belajar merangkai kata", "Raka membeli nasi sayur dan buah"]
    assert fertility(lambda t: t.split(), texts) == pytest.approx(1.0)
    assert gate(1.4, threshold=1.5) is True
    assert gate(1.6, threshold=1.5) is False


def test_train_script_recipe_lists_id_first_sources() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_tokenizer_300m", "scripts/train_tokenizer_300m.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    recipe = mod.TOKENIZER_CORPUS_RECIPE
    names = [entry["name"] for entry in recipe]
    assert "wiki-id" in names and "web-id" in names
    assert abs(sum(entry["weight"] for entry in recipe) - 1.0) < 1e-9


def test_train_script_proof_of_pipeline_on_local_corpora(tmp_path) -> None:
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location(
        "train_tokenizer_300m", "scripts/train_tokenizer_300m.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.copy("data/sample.jsonl", tmp_path / "sample.jsonl")
    out = tmp_path / "tok.json"
    mod.main(["--corpora", str(tmp_path / "sample.jsonl"), "--out", str(out), "--vocab-size", "512"])
    assert out.is_file()
