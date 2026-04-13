import mlx.core as mx
import numpy as np

from turboquant_mlx.generate import RunMetrics, run_generation
from turboquant_mlx.llama_adapter import Model, ModelArgs
from turboquant_mlx.longbench import evaluate_longbench_loaded
from turboquant_mlx.needle import build_needle_case, run_loaded_needle_case
from turboquant_mlx.qa_eval import score_qa_prediction, score_short_answer
from turboquant_mlx.teacher_forcing import evaluate_teacher_forced_loaded


class FakeDetokenizer:
    def __init__(self):
        self._tokens = []

    def add_token(self, token):
        self._tokens.append(str(int(token)))

    def finalize(self):
        return None

    @property
    def text(self):
        return " ".join(self._tokens)


class FakeTokenizer:
    bos_token = None
    eos_token_ids = {999}

    @property
    def detokenizer(self):
        return FakeDetokenizer()

    def encode(self, text, add_special_tokens=True):
        return [idx + 1 for idx, _ in enumerate(text.split())] or [1]

    def decode(self, tokens):
        return " ".join(f"tok{token}" for token in tokens)


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


def test_short_answer_scoring_accepts_numeric_equivalence():
    score = score_short_answer("314159.", ["The launch code is 314159."])
    assert score.correct
    assert score.match_type == "numeric"


def test_qa_scoring_returns_em_and_f1():
    score = score_qa_prediction("Answer: Paris", ["Paris"], dataset_name="triviaqa_e")
    assert score.em == 1.0
    assert score.f1 == 1.0
    assert score.headline_score == 1.0


def test_needle_builder_respects_token_budget():
    tokenizer = FakeTokenizer()
    case = build_needle_case(
        tokenizer,
        context_tokens=32,
        needle="The launch code is 314159.",
        question="What is the launch code?",
        needle_position="back",
        seed=3,
    )
    assert case.context_tokens <= 32
    assert case.needle_position_label == "back"
    assert case.context_tokens > 0


def test_run_generation_reports_memory_deltas(monkeypatch):
    tokenizer = FakeTokenizer()
    memory_state = {"active": [100, 150], "cache": [20, 30]}

    monkeypatch.setattr("turboquant_mlx.generate._ensure_tokenizer", lambda value: value)
    monkeypatch.setattr("turboquant_mlx.generate.make_prompt_cache", lambda model: ["cache"])
    monkeypatch.setattr("turboquant_mlx.generate._reset_peak_memory", lambda: None)
    monkeypatch.setattr("turboquant_mlx.generate._get_active_memory", lambda: memory_state["active"].pop(0))
    monkeypatch.setattr("turboquant_mlx.generate._get_cache_memory", lambda: memory_state["cache"].pop(0))
    monkeypatch.setattr("turboquant_mlx.generate._get_peak_memory", lambda: 80)

    def fake_generate_step(prompt_array, model, max_tokens, sampler, prompt_cache, **kwargs):
        yield mx.array(5), None

    monkeypatch.setattr("turboquant_mlx.generate.generate_step", fake_generate_step)

    _, metrics, _ = run_generation(None, tokenizer, "alpha beta gamma", max_tokens=1)
    assert metrics.context_tokens == 3
    assert metrics.generated_tokens == 1
    assert metrics.active_memory_before_bytes == 100
    assert metrics.active_memory_after_bytes == 150
    assert metrics.peak_memory_delta_bytes == 80
    assert metrics.metal_cache_memory_before_bytes == 20
    assert metrics.metal_cache_memory_after_bytes == 30


def test_teacher_forced_metrics_on_identical_model_are_perfect():
    model = _tiny_model()
    metrics = evaluate_teacher_forced_loaded(model, model, [1, 2, 3, 4], chunk_size=2)
    assert metrics.steps == 3
    assert metrics.top1_agreement == 1.0
    assert metrics.top5_overlap == 1.0
    assert metrics.kl_divergence_mean == 0.0


def test_needle_rows_use_structured_matching(monkeypatch):
    tokenizer = FakeTokenizer()
    case = build_needle_case(
        tokenizer,
        context_tokens=24,
        needle="The launch code is 314159.",
        question="What is the launch code?",
        seed=0,
    )
    metrics = RunMetrics(
        prompt_tokens=case.context_tokens,
        context_tokens=case.context_tokens,
        generated_tokens=1,
        prefill_seconds=0.1,
        decode_seconds=0.1,
        prompt_tps=10.0,
        generation_tps=10.0,
        cache_nbytes=1024,
        active_memory_before_bytes=1,
        active_memory_after_bytes=2,
        peak_memory_delta_bytes=3,
        metal_cache_memory_before_bytes=4,
        metal_cache_memory_after_bytes=5,
    )
    monkeypatch.setattr("turboquant_mlx.needle.run_generation", lambda *args, **kwargs: ("314159.", metrics, None))
    row = run_loaded_needle_case(model=None, tokenizer=tokenizer, case=case, max_tokens=4)
    assert row["correct"] == 1.0
    assert row["match_type"] == "numeric"
    assert row["peak_memory_delta_bytes"] == 3


def test_quality_rows_include_new_metric_schema(monkeypatch):
    examples = [
        {
            "_index": 0,
            "_dataset_name": "triviaqa_e",
            "_prompt": "Question:\nCapital of France?\n\nAnswer:",
            "_prompt_tokens": [1, 2, 3],
            "_prompt_token_target": 16,
            "_prompt_token_count": 3,
            "_context_token_count": 0,
            "answers": ["Paris"],
        }
    ]
    metrics = RunMetrics(
        prompt_tokens=3,
        context_tokens=3,
        generated_tokens=1,
        prefill_seconds=0.2,
        decode_seconds=0.1,
        prompt_tps=15.0,
        generation_tps=10.0,
        cache_nbytes=2048,
        active_memory_before_bytes=10,
        active_memory_after_bytes=20,
        peak_memory_delta_bytes=30,
        metal_cache_memory_before_bytes=40,
        metal_cache_memory_after_bytes=50,
    )
    monkeypatch.setattr("turboquant_mlx.longbench.run_generation", lambda *args, **kwargs: ("Paris", metrics, None))
    result = evaluate_longbench_loaded(
        model=None,
        tokenizer=None,
        examples=examples,
        max_tokens=4,
        turboquant_config=None,
    )
    row = result["rows"][0]
    assert result["mean_em"] == 1.0
    assert result["mean_f1"] == 1.0
    assert row["prompt_token_target"] == 16
    assert row["peak_memory_delta_bytes"] == 30
    assert row["cache_nbytes"] == 2048
