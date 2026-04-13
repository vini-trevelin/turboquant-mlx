import mlx.core as mx

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.generate import generate_tokens
from turboquant_mlx.llama_adapter import Model, ModelArgs


def _tiny_model():
    args = ModelArgs(
        model_type="llama",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
    )
    model = Model(args)
    mx.eval(model.parameters())
    model.eval()
    return model


def test_standard_generation_smoke():
    model = _tiny_model()
    tokens = generate_tokens(model, [1, 2, 3], max_tokens=3)
    assert len(tokens) == 3


def test_turboquant_generation_smoke():
    model = _tiny_model()
    model.enable_turboquant(TurboQuantConfig(mode="core", head_dim=8, core_bits=3))
    tokens = generate_tokens(model, [1, 2, 3], max_tokens=3)
    assert len(tokens) == 3


def test_turboquant_qjl_generation_smoke():
    model = _tiny_model()
    model.enable_turboquant(
        TurboQuantConfig(mode="core", head_dim=8, core_bits=3, qjl_enabled=True, qjl_dim=16)
    )
    tokens = generate_tokens(model, [1, 2, 3], max_tokens=3)
    assert len(tokens) == 3
