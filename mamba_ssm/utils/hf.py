import json

import torch

from transformers.utils import (
    WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from transformers.utils.hub import cached_file

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover
    safe_load_file = None


def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    state_dict = None

    resolved_archive_file = cached_file(
        model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved_archive_file is not None:
        state_dict = torch.load(resolved_archive_file, map_location=mapped_device)
    else:
        if safe_load_file is None:
            raise ImportError("safetensors is required to load this checkpoint")
        index_file = cached_file(
            model_name, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False
        )
        if index_file is not None:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            loaded = {}
            for shard in dict.fromkeys(weight_map.values()):
                shard_file = cached_file(
                    model_name, shard, _raise_exceptions_for_missing_entries=False
                )
                if shard_file is None:
                    raise FileNotFoundError(f"Missing shard file {shard} for model {model_name}")
                loaded.update(safe_load_file(shard_file, device=mapped_device))
            state_dict = loaded
        else:
            safe_file = cached_file(
                model_name, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
            )
            if safe_file is None:
                raise FileNotFoundError(
                    f"Could not find weights for {model_name}. Provide PyTorch or safetensors files."
                )
            state_dict = safe_load_file(safe_file, device=mapped_device)

    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    if device is not None:
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
