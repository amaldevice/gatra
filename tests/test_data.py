from pathlib import Path

from gatra.config import Config, DataConfig, ModelConfig
from gatra.data import TokenDataset, load_corpus, resolve_data_files


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text("".join(f'{{"text": "{text}"}}\n' for text in texts), encoding="utf-8")


def test_mixes_multiple_jsonl_files(tmp_path: Path) -> None:
    first = tmp_path / "culturax-id.jsonl"
    second = tmp_path / "wiki-id.jsonl"
    _write_jsonl(first, ["satu dari culturax"] * 4)
    _write_jsonl(second, ["dua dari wikipedia"] * 3)
    config = Config(
        seed=1337,
        data=DataConfig(path="", paths=[str(first), str(second)], shuffle_docs=False),
        model=ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8),
    )
    files = resolve_data_files(config)
    assert [item.name for item in files] == ["culturax-id.jsonl", "wiki-id.jsonl"]
    texts, counts = load_corpus(config)
    assert counts == {"culturax-id.jsonl": 4, "wiki-id.jsonl": 3}
    assert len(texts) == 7
    dataset = TokenDataset(config)
    assert len(dataset.train_ids) > 0
    assert len(dataset.val_ids) > 0


def test_glob_and_shuffle(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", ["dokumen a yang cukup panjang"])
    _write_jsonl(tmp_path / "b.jsonl", ["dokumen b yang juga panjang"])
    config = Config(
        seed=1,
        data=DataConfig(path="", paths=[str(tmp_path / "*.jsonl")], shuffle_docs=True),
        model=ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8),
    )
    texts, counts = load_corpus(config)
    assert set(counts) == {"a.jsonl", "b.jsonl"}
    assert len(texts) == 2


def test_epoch_cap_hard_fails_above_4(tmp_path: Path) -> None:
    import pytest

    from gatra.data import check_epoch_cap

    stats = {"culturax-id": {"unique_tokens": 100, "train_tokens": 350}}
    assert check_epoch_cap(stats) == {"culturax-id": 3.5}
    with pytest.raises(ValueError, match="epoch cap"):
        check_epoch_cap({"culturax-id": {"unique_tokens": 100, "train_tokens": 450}})


def test_mix_weights_must_sum_to_one() -> None:
    import pytest

    from gatra.config import DataConfig

    with pytest.raises(ValueError, match="mix"):
        DataConfig(path="", paths=[], mix={"a": 0.6, "b": 0.5}).validate_mix()
    DataConfig(path="", paths=[], mix={"a": 0.6, "b": 0.4}).validate_mix()


def test_uint16_bins_roundtrip_with_bpe_tokenizer(tmp_path: Path) -> None:
    import json

    import torch

    from gatra.data import build_bins, load_bins

    corpus = tmp_path / "corpus.jsonl"
    subjects = ["Raka", "Siti", "Gatra", "Budi", "Ayah", "Ibu", "Guru", "Petani"]
    verbs = ["membaca", "menulis", "membeli", "menanam", "memasak", "menjual"]
    texts = [f"{s} {v} buku di pasar pada pagi yang cerah" for s in subjects for v in verbs]
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    from gatra.tokenizer import BPETokenizer

    tok = BPETokenizer.train([corpus], tmp_path / "tok.json", vocab_size=512)
    meta = build_bins([corpus], tmp_path / "bins", tok, name="test-bpe")
    assert meta["tokenizer"]["vocab"] == tok.vocab_size
    assert meta["bins_dtype"] == "uint16"
    ids, back = load_bins(tmp_path / "bins", tok)
    assert ids.dtype == torch.int64
    assert back["tokenizer"]["name"] == "test-bpe"
    assert tok.decode(ids[:50].tolist()) == tok.decode(tok.encode(tok.decode(ids[:50].tolist())))


def test_load_bins_warns_on_tokenizer_mismatch(tmp_path: Path) -> None:
    import json

    import pytest

    from gatra.data import build_bins, load_bins

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"text": f"dokumen nomor {i} tentang pasar dan sawah di desa"}) + "\n"
            for i in range(30)
        )
    )
    from gatra.tokenizer import BPETokenizer

    tok = BPETokenizer.train([corpus], tmp_path / "tok.json", vocab_size=512)
    build_bins([corpus], tmp_path / "bins", tok, name="tok-a")
    other = BPETokenizer.train([corpus], tmp_path / "tok2.json", vocab_size=256)
    with pytest.warns(UserWarning, match="[Tt]okenizer"):
        load_bins(tmp_path / "bins", other)


def test_mix_weights_load_from_gatra_300m_toml() -> None:
    from gatra.config import load_config

    config = load_config("configs/gatra-300m.toml")
    assert config.data.validate_mix("mix") == {
        "id_web": 0.15,
        "en_web": 0.52,
        "code": 0.15,
        "math": 0.10,
        "synthetic": 0.07,
        "instruction": 0.01,
    }
    spark = config.data.validate_mix("mix_spark")
    assert abs(sum(spark.values()) - 1.0) < 1e-9
