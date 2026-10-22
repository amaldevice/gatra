from pathlib import Path

import pytest

from gatra.prepare import SOURCES, filter_scored_jsonl, parse_args, write_jsonl


def test_write_jsonl_keeps_long_id_docs(tmp_path: Path) -> None:
    rows = [
        {"text": "pendek"},
        {"text": "Pagi itu matahari terbit di balik pegunungan dan udara terasa sejuk sekali."},
        {"text": "   "},
        {"text": "Raka berjalan ke pasar untuk membeli nasi, sayur, dan buah segar untuk makan siang."},
    ]
    out = tmp_path / "id.jsonl"
    stats = write_jsonl(rows, out, limit=10, min_chars=40)
    assert stats["kept"] == 2
    assert stats["skipped_short"] == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "Raka berjalan" in lines[1]


def test_write_jsonl_respects_limit(tmp_path: Path) -> None:
    rows = ({"text": f"dokumen indonesia yang cukup panjang nomor {i} " * 4} for i in range(20))
    out = tmp_path / "id.jsonl"
    stats = write_jsonl(rows, out, limit=3, min_chars=40)
    assert stats["kept"] == 3
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_sources_registry_has_new_entries_with_unique_token_estimates() -> None:
    for name in (
        "madlad-id",
        "leipzig-id",
        "stack-edu",
        "finemath",
        "infimm-webmath",
        "cosmopedia-v2",
        "fineweb-edu",
        "dclm",
        "cosmo-id",
    ):
        assert name in SOURCES, f"missing SOURCES entry: {name}"
        assert SOURCES[name]["expected_unique_tokens"] > 0


def test_write_jsonl_with_score_fn_writes_scores_column(tmp_path: Path) -> None:
    rows = [
        {"text": "Pagi itu matahari terbit di balik pegunungan dan udara terasa sejuk sekali."},
        {"text": "Raka berjalan ke pasar untuk membeli nasi, sayur, dan buah segar untuk makan siang."},
    ]
    out = tmp_path / "scored.jsonl"
    stats = write_jsonl(
        rows,
        out,
        limit=10,
        min_chars=40,
        score_fn=lambda text: {"edu": 4.0, "lid": 0.99},
    )
    assert stats["kept"] == 2
    import json

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(set(rec["scores"]) == {"edu", "lid"} for rec in records)


def test_filter_scored_jsonl_threshold_tunable_without_restream(tmp_path: Path) -> None:
    import json

    src = tmp_path / "scored.jsonl"
    src.write_text(
        "".join(
            json.dumps({"text": f"dokumen cukup panjang nomor {i} " * 4, "scores": {"edu": float(i)}})
            + "\n"
            for i in range(5)
        ),
        encoding="utf-8",
    )
    loose = tmp_path / "loose.jsonl"
    strict = tmp_path / "strict.jsonl"
    assert filter_scored_jsonl(src, loose, "edu", 1.0)["kept"] == 4
    assert filter_scored_jsonl(src, strict, "edu", 3.0)["kept"] == 2


def test_prepare_cli_accepts_optional_score_flags() -> None:
    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "gatra-prepare",
            "--edu-classifier",
            "models/edu.bin",
            "--kenlm-id",
            "models/wiki-id.arpa",
            "--fasttext-lid",
            "models/lid.bin",
        ]
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.edu_classifier == "models/edu.bin"
    assert args.kenlm_id == "models/wiki-id.arpa"
    assert args.fasttext_lid == "models/lid.bin"
