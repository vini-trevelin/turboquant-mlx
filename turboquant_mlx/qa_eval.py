from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional


_ARTICLES = {"a", "an", "the"}
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
_NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]")
_NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)?")
_ANSWER_PREFIX_RE = re.compile(r"^(answer|final answer|response)\s*:\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ShortAnswerScore:
    correct: bool
    normalized_prediction: str
    normalized_gold: str
    match_type: str


@dataclass(frozen=True)
class QAScore:
    dataset: str
    metric: str
    em: float
    f1: float
    headline_score: float
    normalized_prediction: str
    normalized_gold: str
    matched_answer: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _strip_articles(text: str) -> str:
    return " ".join(token for token in text.split() if token not in _ARTICLES)


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = _ANSWER_PREFIX_RE.sub("", text)
    text = text.translate(_PUNCT_TRANSLATION)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _strip_articles(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def canonical_short_answer(text: str) -> str:
    value = text.strip()
    if not value:
        return value
    line = value.splitlines()[0].strip()
    line = re.split(r"\b(?:explanation|because|reasoning)\b\s*[:\-]", line, maxsplit=1, flags=re.IGNORECASE)[0]
    line = _ANSWER_PREFIX_RE.sub("", line)
    line = line.strip().strip("\"'` ")
    sentence = re.split(r"(?<=[.!?])\s+", line, maxsplit=1)[0]
    return sentence.strip()


def extract_numeric_spans(text: str) -> List[str]:
    return [match.group(0).replace(",", "") for match in _NUMERIC_RE.finditer(text)]


def score_short_answer(prediction: str, answers: Iterable[str]) -> ShortAnswerScore:
    canonical_prediction = canonical_short_answer(prediction)
    normalized_prediction = normalize_answer(canonical_prediction)
    numeric_prediction = extract_numeric_spans(canonical_prediction)

    best = ShortAnswerScore(
        correct=False,
        normalized_prediction=normalized_prediction,
        normalized_gold="",
        match_type="none",
    )

    for answer in answers:
        canonical_gold = canonical_short_answer(answer)
        normalized_gold = normalize_answer(canonical_gold)
        numeric_gold = extract_numeric_spans(canonical_gold)
        if numeric_gold and any(value in numeric_prediction for value in numeric_gold):
            return ShortAnswerScore(
                correct=True,
                normalized_prediction=normalized_prediction,
                normalized_gold=normalized_gold,
                match_type="numeric",
            )
        if normalized_prediction == normalized_gold and normalized_gold:
            return ShortAnswerScore(
                correct=True,
                normalized_prediction=normalized_prediction,
                normalized_gold=normalized_gold,
                match_type="exact",
            )
        if normalized_gold and normalized_gold in normalized_prediction:
            return ShortAnswerScore(
                correct=True,
                normalized_prediction=normalized_prediction,
                normalized_gold=normalized_gold,
                match_type="substring",
            )
        if len(normalized_prediction.split()) <= 4 and normalized_prediction and normalized_prediction in normalized_gold:
            return ShortAnswerScore(
                correct=True,
                normalized_prediction=normalized_prediction,
                normalized_gold=normalized_gold,
                match_type="short-span",
            )
        if not best.normalized_gold:
            best = ShortAnswerScore(
                correct=False,
                normalized_prediction=normalized_prediction,
                normalized_gold=normalized_gold,
                match_type="none",
            )
    return best


def token_f1_score(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(canonical_short_answer(prediction)).split()
    gold_tokens = normalize_answer(canonical_short_answer(answer)).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_qa_prediction(prediction: str, answers: Iterable[str], *, dataset_name: str) -> QAScore:
    answers = [answer for answer in answers if answer]
    if not answers:
        return QAScore(
            dataset=dataset_name,
            metric="qa_em_f1",
            em=0.0,
            f1=0.0,
            headline_score=0.0,
            normalized_prediction=normalize_answer(canonical_short_answer(prediction)),
            normalized_gold="",
            matched_answer="",
        )

    normalized_prediction = normalize_answer(canonical_short_answer(prediction))
    best_answer = answers[0]
    best_em = 0.0
    best_f1 = 0.0
    best_gold = normalize_answer(canonical_short_answer(best_answer))

    for answer in answers:
        normalized_gold = normalize_answer(canonical_short_answer(answer))
        em = float(normalized_prediction == normalized_gold and normalized_gold != "")
        f1 = token_f1_score(prediction, answer)
        if (f1, em) > (best_f1, best_em):
            best_answer = answer
            best_em = em
            best_f1 = f1
            best_gold = normalized_gold

    return QAScore(
        dataset=dataset_name,
        metric="qa_em_f1",
        em=best_em,
        f1=best_f1,
        headline_score=best_f1,
        normalized_prediction=normalized_prediction,
        normalized_gold=best_gold,
        matched_answer=best_answer,
    )


def build_prompt(example: Mapping[str, str]) -> str:
    sections = []
    for key in ("instruction", "context", "input", "question"):
        value = example.get(key)
        if value:
            sections.append(f"{key.capitalize()}:\n{value}")
    sections.append("Answer:")
    return "\n\n".join(sections)


def truncate_text_to_token_limit(tokenizer, text: str, token_limit: Optional[int]) -> str:
    if token_limit is None:
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= token_limit:
        return text
    return tokenizer.decode(tokens[:token_limit]).strip()


def truncate_example_to_prompt_tokens(example: Mapping[str, str], tokenizer, prompt_token_limit: int) -> dict:
    context = example.get("context", "") or ""
    prefix_parts = []
    suffix_parts = []
    for key in ("instruction",):
        value = example.get(key)
        if value:
            prefix_parts.append(f"{key.capitalize()}:\n{value}")
    if context:
        prefix_parts.append("Context:\n")
    for key in ("input", "question"):
        value = example.get(key)
        if value:
            suffix_parts.append(f"{key.capitalize()}:\n{value}")
    suffix_parts.append("Answer:")

    prefix = "\n\n".join(prefix_parts)
    suffix = "\n\n".join(suffix_parts)
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    context_tokens = tokenizer.encode(context, add_special_tokens=False) if context else []
    available_context_tokens = max(prompt_token_limit - len(prefix_tokens) - len(suffix_tokens), 0)
    if len(context_tokens) > available_context_tokens:
        context = tokenizer.decode(context_tokens[:available_context_tokens]).strip()

    truncated = dict(example)
    truncated["context"] = context
    prompt = build_prompt(truncated)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) > prompt_token_limit:
        prompt = tokenizer.decode(prompt_tokens[:prompt_token_limit]).strip()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    truncated["_prompt"] = prompt
    truncated["_prompt_tokens"] = prompt_tokens
    truncated["_prompt_token_target"] = prompt_token_limit
    truncated["_prompt_token_count"] = len(prompt_tokens)
    truncated["_context_token_count"] = min(len(context_tokens), available_context_tokens)
    return truncated
