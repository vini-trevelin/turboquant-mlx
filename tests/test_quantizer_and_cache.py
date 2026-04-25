import mlx.core as mx
import numpy as np

from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantizer import (
    SharedTurboQuantSetup,
    apply_attention_to_values,
    dequantize_tensor,
    quantize_tensor,
    score_queries_against_keys,
)


def _preset_config(qjl_enabled=False, qjl_dim=64):
    return TurboQuantConfig(
        mode="preset",
        head_dim=8,
        preset_name="2.5",
        qjl_enabled=qjl_enabled,
        qjl_dim=qjl_dim,
        outlier_indices=(0, 1),
    )


def test_quantize_dequantize_preserves_shape():
    config = _preset_config()
    setup = SharedTurboQuantSetup.from_config(config)
    tensor = mx.array(np.random.default_rng(0).normal(size=(1, 2, 3, 8)).astype(np.float32))
    compressed = quantize_tensor(tensor, setup)
    restored = dequantize_tensor(compressed, setup)
    assert restored.shape == tensor.shape


def test_cache_streaming_append_and_trim():
    config = TurboQuantConfig(mode="core", head_dim=8, core_bits=3, qjl_enabled=True, qjl_dim=16)
    cache = TurboQuantKVCache(config)
    cache.step = 2
    rng = np.random.default_rng(1)
    first_keys = mx.array(rng.normal(size=(1, 2, 2, 8)).astype(np.float32))
    first_values = mx.array(rng.normal(size=(1, 2, 2, 8)).astype(np.float32))
    second_keys = mx.array(rng.normal(size=(1, 2, 1, 8)).astype(np.float32))
    second_values = mx.array(rng.normal(size=(1, 2, 1, 8)).astype(np.float32))

    cache.update_and_fetch(first_keys, first_values)
    assert cache.offset == 2
    cache.update_and_fetch(second_keys, second_values)
    assert cache.offset == 3
    assert cache.keys.sequence_length == 3
    assert cache.keys.chunk_count == 2
    assert cache.observed_nbytes() > 0
    assert cache.perf_stats["cache_update_seconds"] > 0

    cache.trim(1)
    assert cache.offset == 2
    assert cache.keys.sequence_length == 2
    assert cache.keys.chunk_count == 1


def test_compressed_attention_matches_dense_reference():
    config = TurboQuantConfig(mode="core", head_dim=8, core_bits=3)
    setup = SharedTurboQuantSetup.from_config(config)
    rng = np.random.default_rng(2)
    queries = mx.array(rng.normal(size=(1, 2, 3, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, 2, 5, 8)).astype(np.float32))
    values = mx.array(rng.normal(size=(1, 2, 5, 8)).astype(np.float32))
    compressed_keys = quantize_tensor(keys, setup)
    compressed_values = quantize_tensor(values, setup)

    scores = score_queries_against_keys(
        queries,
        compressed_keys,
        setup,
        scale=0.5,
        apply_qjl=False,
    )
    dense_keys = dequantize_tensor(compressed_keys, setup)
    dense_values = dequantize_tensor(compressed_values, setup)
    dense_scores = mx.matmul(queries, dense_keys.transpose(0, 1, 3, 2)) * 0.5

    assert np.allclose(np.asarray(scores), np.asarray(dense_scores), atol=1e-4)

    weights = mx.softmax(scores, axis=-1, precise=True)
    compressed_out = apply_attention_to_values(weights, compressed_values, setup)
    dense_out = mx.matmul(mx.softmax(dense_scores, axis=-1, precise=True), dense_values)
    assert np.allclose(np.asarray(compressed_out), np.asarray(dense_out), atol=1e-4)


def test_chunked_compressed_attention_matches_dense_reference():
    config = TurboQuantConfig(mode="core", head_dim=8, core_bits=3, qjl_enabled=True, qjl_dim=32)
    cache = TurboQuantKVCache(config)
    cache.step = 2
    setup = cache.setup
    rng = np.random.default_rng(5)
    queries = mx.array(rng.normal(size=(1, 2, 3, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, 2, 5, 8)).astype(np.float32))
    values = mx.array(rng.normal(size=(1, 2, 5, 8)).astype(np.float32))

    compressed_keys, compressed_values = cache.update_and_fetch(keys, values)
    dense_keys = dequantize_tensor(compressed_keys, setup)
    dense_values = dequantize_tensor(compressed_values, setup)

    scores = score_queries_against_keys(
        queries,
        compressed_keys,
        setup,
        scale=0.5,
        apply_qjl=False,
    )
    dense_scores = mx.matmul(queries, dense_keys.transpose(0, 1, 3, 2)) * 0.5
    assert scores.shape == dense_scores.shape

    weights = mx.softmax(scores, axis=-1, precise=True)
    compressed_out = apply_attention_to_values(weights, compressed_values, setup)
    dense_out = mx.matmul(mx.softmax(dense_scores, axis=-1, precise=True), dense_values)
    assert np.allclose(np.asarray(compressed_out), np.asarray(dense_out), atol=1e-4)


def test_teacher_forced_self_parity_zero_kl_and_matched_nll():
    # Standard model compared to itself must produce KL=0, top1=1.0, and
    # identical NLLs for both columns. This guards _compare_logits from being
    # accidentally rewired so the targets never reach the NLL accumulator.
    import mlx.core as mx
    import numpy as np

    from turboquant_mlx.teacher_forcing import _compare_logits

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(8, 11)).astype(np.float32)
    targets = rng.integers(0, 11, size=(8,))
    top1, top5, kl, std_nll, turbo_nll = _compare_logits(logits, logits, targets)
    assert top1 == 8.0
    assert abs(top5 - 8.0) < 1e-6
    assert abs(kl) < 1e-5
    assert abs(std_nll - turbo_nll) < 1e-6
    assert std_nll > 0.0


def test_qjl_oracle_reduces_score_error():
    config = TurboQuantConfig(mode="core", head_dim=16, core_bits=2, qjl_enabled=True, qjl_dim=4096)
    setup = SharedTurboQuantSetup.from_config(config)
    rng = np.random.default_rng(17)
    queries = mx.array(rng.normal(size=(1, 1, 2, 16)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, 1, 4, 16)).astype(np.float32))
    compressed = quantize_tensor(keys, setup, with_qjl=True)

    dense = mx.matmul(queries, keys.transpose(0, 1, 3, 2))
    no_qjl = score_queries_against_keys(queries, compressed, setup, scale=1.0, apply_qjl=False)
    with_qjl = score_queries_against_keys(queries, compressed, setup, scale=1.0, apply_qjl=True)

    err_no_qjl = float(mx.mean(mx.abs(no_qjl - dense)))
    err_with_qjl = float(mx.mean(mx.abs(with_qjl - dense)))
    assert err_with_qjl < err_no_qjl * 0.5, (
        f"QJL must materially reduce score error: no_qjl={err_no_qjl:.4f}, with_qjl={err_with_qjl:.4f}"
    )


def test_qjl_correction_changes_key_scores():
    config = TurboQuantConfig(mode="core", head_dim=8, core_bits=2, qjl_enabled=True, qjl_dim=512)
    setup = SharedTurboQuantSetup.from_config(config)
    query = mx.array(np.array([[[[1.0, 0.4, -0.5, 0.8, 0.3, -0.7, 0.2, 0.1]]]], dtype=np.float32))
    key = mx.array(np.array([[[[0.8, 0.5, -0.2, 0.6, 0.9, -0.3, 0.4, -0.1]]]], dtype=np.float32))
    compressed = quantize_tensor(key, setup, with_qjl=True)

    mse_score = score_queries_against_keys(query, compressed, setup, scale=1.0, apply_qjl=False)
    qjl_score = score_queries_against_keys(query, compressed, setup, scale=1.0, apply_qjl=True)
    assert not np.allclose(np.asarray(mse_score), np.asarray(qjl_score))


def test_gqa_compressed_attention_matches_dense_reference():
    config = TurboQuantConfig(mode="core", head_dim=8, core_bits=3)
    setup = SharedTurboQuantSetup.from_config(config)
    rng = np.random.default_rng(11)
    n_kv, repeats, q_len, kv_len = 2, 4, 3, 6
    queries = mx.array(rng.normal(size=(1, n_kv * repeats, q_len, 8)).astype(np.float32))
    keys = mx.array(rng.normal(size=(1, n_kv, kv_len, 8)).astype(np.float32))
    values = mx.array(rng.normal(size=(1, n_kv, kv_len, 8)).astype(np.float32))
    compressed_keys = quantize_tensor(keys, setup)
    compressed_values = quantize_tensor(values, setup)

    scores = score_queries_against_keys(queries, compressed_keys, setup, scale=0.5, apply_qjl=False)
    assert scores.shape == (1, n_kv * repeats, q_len, kv_len)

    dense_keys = mx.repeat(dequantize_tensor(compressed_keys, setup), repeats, axis=1)
    dense_values = mx.repeat(dequantize_tensor(compressed_values, setup), repeats, axis=1)
    dense_scores = mx.matmul(queries, dense_keys.transpose(0, 1, 3, 2)) * 0.5
    assert np.allclose(np.asarray(scores), np.asarray(dense_scores), atol=1e-4)

    weights = mx.softmax(scores, axis=-1, precise=True)
    compressed_out = apply_attention_to_values(weights, compressed_values, setup)
    dense_out = mx.matmul(mx.softmax(dense_scores, axis=-1, precise=True), dense_values)
    assert compressed_out.shape == (1, n_kv * repeats, q_len, 8)
    assert np.allclose(np.asarray(compressed_out), np.asarray(dense_out), atol=1e-4)


