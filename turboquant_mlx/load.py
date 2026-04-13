from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import _download, load_adapters, load_model, load_tokenizer

from .config import TurboQuantConfig
from .llama_adapter import Model, ModelArgs


def _get_turboquant_classes(config: dict):
    if config["model_type"] != "llama":
        raise ValueError(
            f"Only llama model_type is supported by the local TurboQuant adapter, got {config['model_type']}"
        )
    return Model, ModelArgs


def load_turboquant(
    path_or_hf_repo: str,
    *,
    turboquant_config: Optional[TurboQuantConfig] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: Optional[str] = None,
) -> Union[Tuple[Model, TokenizerWrapper], Tuple[Model, TokenizerWrapper, Dict[str, Any]]]:
    model_path = _download(path_or_hf_repo, revision=revision)
    model, config = load_model(
        Path(model_path),
        lazy=lazy,
        model_config=model_config,
        get_model_classes=_get_turboquant_classes,
    )
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    if turboquant_config is not None:
        model.enable_turboquant(turboquant_config)
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config,
        eos_token_ids=config.get("eos_token_id", None),
    )
    if return_config:
        return model, tokenizer, config
    return model, tokenizer

