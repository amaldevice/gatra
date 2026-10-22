"""T3: GQA + QK-norm + 300M config tests.

Seams: ModelConfig public fields, Attention forward numerics, param_count.
"""

import torch
import torch.nn.functional as F

from gatra.config import ModelConfig
from gatra.model import Attention, RotaryEmbedding


def test_config_new_fields_backward_compat_defaults() -> None:
    config = ModelConfig()
    # Unset n_kv_heads resolves to n_head (MHA-preserving default).
    assert config.n_kv_heads == config.n_head
    assert config.use_qk_norm is False
    assert config.z_loss_coef == 0.0
    assert config.attn_logit_cap is None


def test_config_rejects_non_divisible_kv_heads() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_kv_heads"):
        ModelConfig(n_embd=32, n_head=4, n_kv_heads=3)


def _reference_mha(attn: Attention, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Independent MHA reference: split q/k/v from the GQA weights at n_kv == n_head."""
    from gatra.model import apply_rotary

    batch, seq, _ = x.shape
    q = attn.q_proj(x).view(batch, seq, attn.n_head, attn.head_dim)
    k = attn.k_proj(x).view(batch, seq, attn.n_head, attn.head_dim)
    v = attn.v_proj(x).view(batch, seq, attn.n_head, attn.head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn.o_proj(out.transpose(1, 2).contiguous().view(batch, seq, -1))


def test_gqa_matches_mha_when_n_kv_equals_n_head() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        n_embd=32, n_head=4, n_kv_heads=4, n_layer=1, block_size=16, use_qk_norm=False
    )
    attn = Attention(config).eval()
    rope = RotaryEmbedding(8, config.block_size, config.rope_base)
    cos, sin = rope(8)
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        got = attn(x, cos, sin)
        want = _reference_mha(attn, x, cos, sin)
    assert torch.allclose(got, want, atol=1e-5)


def test_gqa_kv_repeat_shape() -> None:
    config = ModelConfig(
        n_embd=32, n_head=4, n_kv_heads=2, n_layer=1, block_size=16, use_qk_norm=False
    )
    attn = Attention(config).eval()
    assert attn.k_proj.out_features == 2 * attn.head_dim
    assert attn.v_proj.out_features == 2 * attn.head_dim
    assert attn.q_proj.out_features == 4 * attn.head_dim


def test_qk_norm_on_off_changes_output_and_adds_params() -> None:
    torch.manual_seed(1)
    base = dict(n_embd=32, n_head=4, n_kv_heads=2, n_layer=1, block_size=16)
    off = Attention(ModelConfig(**base, use_qk_norm=False)).eval()
    on = Attention(ModelConfig(**base, use_qk_norm=True)).eval()
    on.load_state_dict(off.state_dict(), strict=False)
    x = torch.randn(1, 8, 32)
    rope = RotaryEmbedding(8, 16, 10000.0)
    cos, sin = rope(8)
    with torch.no_grad():
        y_off = off(x, cos, sin)
        y_on = on(x, cos, sin)
    assert not torch.allclose(y_off, y_on, atol=1e-5)
    assert sum(p.numel() for p in on.parameters()) > sum(
        p.numel() for p in off.parameters()
    )


def test_z_loss_and_logit_cap_wired_off_by_default() -> None:
    from gatra.model import Gatra

    torch.manual_seed(2)
    config = ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=16)
    model = Gatra(config).eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits_base, loss_base = model(idx, idx)
    assert loss_base is not None
    zl = Gatra(ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=16, z_loss_coef=1e-4))
    zl.load_state_dict(model.state_dict())
    zl.eval()
    with torch.no_grad():
        _, loss_zl = zl(idx, idx)
    assert loss_zl is not None and loss_zl.item() > loss_base.item()
    cap = Attention(
        ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=16, attn_logit_cap=50.0)
    ).eval()
    assert cap.attn_logit_cap == 50.0


def test_existing_tomls_still_load_as_mha() -> None:
    from pathlib import Path

    from gatra.config import load_config

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("gatra-smoke.toml", "gatra-1m.toml", "gatra-10m.toml"):
        config = load_config(root / name)
        assert config.model.n_kv_heads == config.model.n_head
        assert config.model.use_qk_norm is False
        assert config.model.z_loss_coef == 0.0
        assert config.model.attn_logit_cap is None


def test_300m_param_count_within_one_percent() -> None:
    from pathlib import Path

    from gatra.config import load_config
    from gatra.model import Gatra

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "gatra-300m.toml")
    model = Gatra(config.model)
    count = model.param_count()
    assert abs(count - 299_900_000) / 299_900_000 < 0.01, f"param_count={count}"
