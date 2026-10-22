"""Reduced-scope T4 tests: wsd_lr(tokens_seen) + TrainConfig new fields.

Follow-ups (not covered here): DDP/bf16 entry, checkpoint scheduler-state,
grad-accum loop wiring, accum-equivalence numerics.
"""

import pytest

from gatra.config import config_from_dict, load_config
from gatra.train import (
    cosine_lr,
    current_lr,
    resolve_accum_steps,
    wsd_lr,
    wsd_planned_tokens,
)


def _wsd_config(**overrides):
    raw = {
        "model": {"block_size": 2048},
        "train": {
            "schedule": "wsd",
            "learning_rate": 3e-3,
            "min_lr": 0.0,
            "warmup_tokens": 4_000_000,
            "total_tokens": 100_000_000,
            "decay_frac": 0.2,
            "max_iters": 1000,
        },
    }
    raw["train"].update(overrides)
    return config_from_dict(raw)


def test_wsd_warmup_shape() -> None:
    config = _wsd_config()
    assert wsd_lr(0, config) == 0.0
    assert wsd_lr(2_000_000, config) == pytest.approx(1.5e-3)
    assert wsd_lr(4_000_000, config) == pytest.approx(3e-3)


def test_wsd_stable_phase() -> None:
    config = _wsd_config()
    # decay starts at total - 0.2*total = 80M
    assert wsd_lr(50_000_000, config) == pytest.approx(3e-3)
    assert wsd_lr(80_000_000, config) == pytest.approx(3e-3)


def test_wsd_decay_shape() -> None:
    config = _wsd_config()
    assert wsd_lr(90_000_000, config) == pytest.approx(1.5e-3)
    assert wsd_lr(100_000_000, config) == pytest.approx(0.0)
    assert wsd_lr(150_000_000, config) == pytest.approx(0.0)


def test_wsd_min_lr_floor() -> None:
    config = _wsd_config(min_lr=3e-4)
    assert wsd_lr(100_000_000, config) == pytest.approx(3e-4)
    assert wsd_lr(200_000_000, config) == pytest.approx(3e-4)


def test_wsd_no_warmup_no_decay() -> None:
    config = _wsd_config(warmup_tokens=0, decay_frac=0.0)
    assert wsd_lr(0, config) == pytest.approx(3e-3)
    assert wsd_lr(99_999_999, config) == pytest.approx(3e-3)


def test_legacy_cosine_is_default() -> None:
    config = config_from_dict({})
    assert config.train.schedule == "cosine"
    assert config.train.accum_steps == 0
    assert config.train.global_tokens == 0
    assert config.train.bf16 is False
    assert config.train.ddp is False


def test_current_lr_dispatch() -> None:
    legacy = config_from_dict({})
    assert current_lr(0, 0, legacy) == cosine_lr(0, legacy)
    wsd = _wsd_config()
    assert current_lr(10, 50_000_000, wsd) == wsd_lr(50_000_000, wsd)


def test_accum_auto_derive() -> None:
    # 8 seqs x 2048 toks x 8 GPUs = 131072/micro-step; 2097152/131072 = 16
    config = _wsd_config(micro_batch_seqs=8, global_tokens=2_097_152)
    assert resolve_accum_steps(config, world_size=8) == 16


def test_accum_explicit_wins() -> None:
    config = _wsd_config(micro_batch_seqs=8, global_tokens=2_097_152, accum_steps=4)
    assert resolve_accum_steps(config, world_size=8) == 4


def test_accum_default_is_one() -> None:
    assert resolve_accum_steps(config_from_dict({})) == 1


def test_accum_not_divisible_raises() -> None:
    config = _wsd_config(micro_batch_seqs=8, global_tokens=2_000_000)
    with pytest.raises(ValueError):
        resolve_accum_steps(config, world_size=8)


def test_planned_tokens_explicit_and_derived() -> None:
    assert wsd_planned_tokens(_wsd_config()) == 100_000_000
    derived = _wsd_config(total_tokens=0, micro_batch_seqs=8, global_tokens=2_097_152)
    # world_size=1 default: 128 accum x 8 seqs x 2048 toks x 1000 iters
    assert wsd_planned_tokens(derived) == 128 * 8 * 2048 * 1000


def test_bad_schedule_raises() -> None:
    with pytest.raises(ValueError):
        config_from_dict({"train": {"schedule": "linear"}})


def test_new_fields_load_from_toml(tmp_path) -> None:
    toml = tmp_path / "wsd.toml"
    toml.write_text(
        "[train]\nschedule = \"wsd\"\nwarmup_tokens = 4000000\n"
        "decay_frac = 0.2\nmin_lr = 0.0003\nglobal_tokens = 2097152\n"
        "micro_batch_seqs = 8\nbf16 = true\nddp = true\n"
    )
    config = load_config(toml)
    assert config.train.schedule == "wsd"
    assert config.train.warmup_tokens == 4_000_000
    assert config.train.decay_frac == pytest.approx(0.2)
    assert config.train.min_lr == pytest.approx(0.0003)
    assert config.train.global_tokens == 2_097_152
    assert config.train.micro_batch_seqs == 8
    assert config.train.bf16 is True
    assert config.train.ddp is True
