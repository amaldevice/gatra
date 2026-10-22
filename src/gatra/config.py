from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


def parse_duration(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    raw = value.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if raw[-1] in units:
        return float(raw[:-1]) * units[raw[-1]]
    return float(raw)


@dataclass
class DataConfig:
    path: str = "data/sample.jsonl"
    paths: list[str] = field(default_factory=list)
    val_ratio: float = 0.1
    shuffle_docs: bool = True
    doc_separator: str = "\n"
    mix: dict[str, float] = field(default_factory=dict)
    mix_spark: dict[str, float] = field(default_factory=dict)

    def sources(self) -> list[str]:
        if self.paths:
            return list(self.paths)
        return [self.path] if self.path else []

    def validate_mix(self, profile: str = "mix") -> dict[str, float]:
        """Mix weights (from [data.mix] / [data.mix_spark]) must sum to 1.0."""
        weights = self.mix if profile == "mix" else self.mix_spark
        if not weights:
            return {}
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"[data.{profile}] weights sum to {total:.6f}, must sum to 1.0")
        return dict(weights)


@dataclass
class ModelConfig:
    n_embd: int = 128
    n_head: int = 4
    n_kv_heads: int | None = None
    n_layer: int = 4
    n_ff: int = 0
    block_size: int = 128
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    vocab_size: int = 256
    use_qk_norm: bool = False
    z_loss_coef: float = 0.0
    attn_logit_cap: float | None = None

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_heads != 0:
            raise ValueError("n_head must be divisible by n_kv_heads")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.n_ff <= 0:
            self.n_ff = 64 * ((int(8 * self.n_embd / 3) + 63) // 64)


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_iters: int = 50
    grad_clip: float = 1.0
    max_time: str | None = None
    out_dir: str = "checkpoints"
    sample_tokens: int = 80
    sample_prompt_chars: int = 24
    sample_count: int = 4
    sample_every: int = 0
    schedule: str = "cosine"
    warmup_tokens: int = 0
    total_tokens: int = 0
    decay_frac: float = 0.2
    min_lr: float = 0.0
    global_tokens: int = 0
    micro_batch_seqs: int = 0
    accum_steps: int = 0
    bf16: bool = False
    ddp: bool = False
    fp16_fallback: bool = True

    def __post_init__(self) -> None:
        if self.schedule not in ("cosine", "wsd"):
            raise ValueError("schedule must be 'cosine' or 'wsd'")
        if self.decay_frac < 0.0 or self.decay_frac > 1.0:
            raise ValueError("decay_frac must be in [0, 1]")


@dataclass
class GenerateConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


@dataclass
class Config:
    name: str = "gatra"
    seed: int = 1337
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    source: Path | None = None

    @property
    def max_time_seconds(self) -> float | None:
        return parse_duration(self.train.max_time)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.train.out_dir) / self.name


def _section(cls, raw: dict):
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in names})


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = Config(
        name=raw.get("name", config_path.stem),
        seed=raw.get("seed", 1337),
        device=raw.get("device", "auto"),
        data=_section(DataConfig, raw.get("data", {})),
        model=_section(ModelConfig, raw.get("model", {})),
        train=_section(TrainConfig, raw.get("train", {})),
        generate=_section(GenerateConfig, raw.get("generate", {})),
        source=config_path,
    )
    root = config_path.parent.parent
    if config.data.paths:
        config.data.paths = [_resolve_existing(item, root) for item in config.data.paths]
    else:
        config.data.path = _resolve_existing(config.data.path, root)
    return config


def _resolve_existing(value: str, root: Path) -> str:
    path = Path(value)
    if path.exists() or "*" in value or "?" in value or "[" in value:
        return value
    alt = root / path
    if alt.exists() or any(alt.parent.glob(alt.name)):
        return str(alt)
    return value


def config_from_dict(raw: dict) -> Config:
    source = raw.get("source")
    return Config(
        name=raw.get("name", "gatra"),
        seed=raw.get("seed", 1337),
        device=raw.get("device", "auto"),
        data=_section(DataConfig, raw.get("data", {})),
        model=_section(ModelConfig, raw.get("model", {})),
        train=_section(TrainConfig, raw.get("train", {})),
        generate=_section(GenerateConfig, raw.get("generate", {})),
        source=Path(source) if source else None,
    )


def apply_overrides(
    config: Config,
    max_time: str | None = None,
    out_dir: str | None = None,
    device: str | None = None,
    max_iters: int | None = None,
) -> Config:
    train = config.train
    if max_time is not None:
        train = replace(train, max_time=max_time)
    if out_dir is not None:
        train = replace(train, out_dir=out_dir)
    if max_iters is not None:
        train = replace(train, max_iters=max_iters)
    return replace(config, train=train, device=device or config.device)
