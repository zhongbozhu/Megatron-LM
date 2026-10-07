# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generic HF chat encoding with unshifted targets independent of packing.

Templates own formatting and assistant supervision boundaries through Jinja
``generation`` blocks. Full supervision includes every rendered token.
"""

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache

import numpy as np

IGNORE_INDEX = -100
_CHAT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_PIPELINE_TEMPLATE_KWARGS = frozenset(
    {
        "add_generation_prompt",
        "add_special_tokens",
        "chat_template",
        "continue_final_message",
        "conversation",
        "max_length",
        "messages",
        "padding",
        "return_assistant_tokens_mask",
        "return_assistant_token_mask",
        "return_dict",
        "return_offsets_mapping",
        "return_tensors",
        "tokenize",
        "tokenizer_kwargs",
        "tools",
        "truncation",
    }
)


def _parse_json(value, field):
    if not isinstance(value, str):
        return deepcopy(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} contains invalid JSON at character {error.pos}") from None


def normalize_chat(messages, tools=None):
    """Validate text records while retaining fields and caller-owned values.

    Native and JSON-serialized arrays are accepted. Tool arguments normalize to
    objects; templates control their serialization. Unknown roles and multimodal
    content are errors, rather than silent remapping or loss of information.
    """
    messages = _parse_json(messages, "messages")
    tools = _parse_json(tools, "tools")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if tools is not None and (
        not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise ValueError("tools must be a list of objects")
    for index, message in enumerate(messages):
        field = f"messages[{index}]"
        if not isinstance(message, dict):
            raise ValueError(f"{field} must be an object")
        if not isinstance(message.get("role"), str) or message["role"] not in _CHAT_ROLES:
            raise ValueError(f"{field}.role must be one of {sorted(_CHAT_ROLES)}")
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message["content"], str):
            raise ValueError(f"{field}.content must be text; multimodal content is unsupported")
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError(f"{field}.tool_calls must be a list")
        for call_index, call in enumerate(calls or []):
            call_field = f"{field}.tool_calls[{call_index}]"
            if not isinstance(call, dict):
                raise ValueError(f"{call_field} must be an object")
            function = call.get("function", call)
            if not isinstance(function, dict):
                raise ValueError(f"{call_field}.function must be an object")
            arguments = _parse_json(function.get("arguments"), f"{call_field}.arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"{call_field}.arguments must be a JSON object")
            function["arguments"] = arguments
    return messages, tools


def _template_controls(chat_template_kwargs):
    if chat_template_kwargs is None:
        controls = {}
    elif isinstance(chat_template_kwargs, Mapping) and all(
        isinstance(key, str) for key in chat_template_kwargs
    ):
        controls = deepcopy(dict(chat_template_kwargs))
    else:
        raise ValueError("chat_template_kwargs must be a string-keyed mapping")
    forbidden = _PIPELINE_TEMPLATE_KWARGS.intersection(controls)
    if forbidden:
        raise ValueError(f"chat_template_kwargs cannot override {sorted(forbidden)}")
    return controls


def _single_sequence(values, name):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Expected a single {name} sequence, got shape {array.shape}")
    return array


def _input_ids(encoded):
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    ids = _single_sequence(encoded, "input_ids")
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError("Chat input_ids must be integers")
    return ids.astype(np.int64, copy=False)


@lru_cache(maxsize=32)
def _validate_generation_template(template):
    """Distinguish missing training-template support from an unsupervised row."""
    from jinja2 import Environment

    in_block = False
    for _, kind, value in Environment().lex(template):
        if kind == "block_begin":
            in_block = True
        elif in_block and kind != "whitespace":
            if kind == "name" and value == "generation":
                return
            in_block = False
    raise ValueError("Assistant loss requires a generation-tagged HF training template")


def tokenize_chat(
    hf,
    messages,
    *,
    tools=None,
    template=None,
    loss_mode="assistant",
    chat_template_kwargs=None,
    add_generation_prompt=False,
):
    """Return complete-conversation IDs and equal-length, unshifted targets.

    Assistant loss follows the HF generation mask exactly. The supplied chat
    template must mark every desired target, including reasoning, tool calls
    and any end tokens, with ``{% generation %}`` blocks. Missing, malformed or
    empty masks fail explicitly; no boundaries are inferred from rendered text.
    Full loss supervises every rendered token without adding an EOS token.
    """
    if loss_mode not in ("assistant", "full"):
        raise ValueError(f"Unknown chat loss mode: {loss_mode}")
    if loss_mode == "assistant" and add_generation_prompt:
        raise ValueError("Assistant supervision expects completed turns, not a generation prompt")
    messages, tools = normalize_chat(messages, tools)
    controls = _template_controls(chat_template_kwargs)
    kwargs = dict(
        tools=tools, chat_template=template, add_generation_prompt=add_generation_prompt, **controls
    )
    if loss_mode == "full":
        ids = _input_ids(hf.apply_chat_template(messages, tokenize=True, **kwargs))
        return ids, ids.copy()
    template = hf.get_chat_template(chat_template=template, tools=tools)
    _validate_generation_template(template)
    kwargs["chat_template"] = template
    encoded = hf.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, **kwargs
    )
    ids = _input_ids(encoded)
    if not isinstance(encoded, Mapping) or "assistant_masks" not in encoded:
        raise ValueError(
            "Chat template returned no assistant mask; assistant loss requires a "
            "generation-tagged HF chat template"
        )
    raw_mask = _single_sequence(encoded["assistant_masks"], "assistant_masks")
    if raw_mask.shape != ids.shape or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("Chat template returned an invalid assistant mask")
    mask = raw_mask.astype(bool)
    if not mask[1:].any():
        raise ValueError("No next-token assistant targets in this conversation")
    return ids, np.where(mask, ids, IGNORE_INDEX)


def truncate_chat(tokens, targets, sequence_length, loss_mode):
    """Right-truncate unshifted chat targets without adding or replacing EOS.

    Shared by online sample assembly and offline preparation. The caller pads
    and applies the single next-token shift after this function returns.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    tokens = _input_ids(tokens)
    targets = _single_sequence(targets, "targets")
    if tokens.shape != targets.shape or (targets.size and targets.dtype.kind not in "iu"):
        raise ValueError("Chat tokens and targets must be equal-length integer sequences")
    if np.any((targets != IGNORE_INDEX) & (targets != tokens)):
        raise ValueError("Each chat target must equal its token ID or IGNORE_INDEX")
    tokens = tokens[: sequence_length + 1]
    targets = targets[: sequence_length + 1].astype(np.int64, copy=False)
    if len(tokens) < 2:
        raise ValueError("Explicit chat requires at least two tokens for next-token prediction")
    if loss_mode == "assistant" and not np.any(targets[1:] != IGNORE_INDEX):
        raise ValueError(
            f"No assistant targets remain after truncation to {sequence_length}; "
            "increase --seq-length or filter the source trajectory"
        )
    return tokens, targets
