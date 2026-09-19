# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Text chat normalization and assistant supervision, independent of packing."""

import inspect
import json
import re
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

IGNORE_INDEX = -100

_CHATML_START = "<|im_start|>assistant\n"
_CHATML_END = "<|im_end|>"
_PIPELINE_TEMPLATE_KWARGS = {
    "add_generation_prompt",
    "chat_template",
    "continue_final_message",
    "max_length",
    "padding",
    "return_assistant_tokens_mask",
    "return_dict",
    "return_tensors",
    "tokenize",
    "tools",
    "truncation",
}
# Match Bridge's Qwen control-token exclusion, not a blanket exclusion of
# special tokens: assistant end tokens must remain supervised.
_CHATML_SKIPPED_TOKENS = {
    "<|im_start|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|AUDIO|>",
    "<|audio_bos|>",
    "<|audio_eos|>",
}


def normalize_chat(messages, tools=None):
    """Preserve tool/reasoning fields without injecting synthetic system turns."""
    messages = json.loads(messages) if isinstance(messages, str) else deepcopy(messages)
    tools = json.loads(tools) if isinstance(tools, str) else deepcopy(tools)
    if not isinstance(messages, list) or not messages:
        raise ValueError("Chat messages must be a nonempty list")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("Chat tools must be a list")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("Each chat message must have a string role")
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message["content"], str):
            raise ValueError("Chat SFT supports text content only")
        if message.get("tool_calls") is not None and not isinstance(message["tool_calls"], list):
            raise ValueError("Tool calls must be a list")
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                raise ValueError("Each tool call must be an object")
            function = call.get("function", call)
            if not isinstance(function, dict):
                raise ValueError("Tool-call function must be an object")
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
            if not isinstance(function.get("arguments"), dict):
                raise ValueError("Tool-call arguments must be a JSON object")
    return messages, tools


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _resolve_template_kwargs(hf, template, chat_template_kwargs):
    """Forward per-row controls without letting them replace pipeline policy."""
    if chat_template_kwargs is None:
        return {}
    if not isinstance(chat_template_kwargs, Mapping) or not all(
        isinstance(key, str) for key in chat_template_kwargs
    ):
        raise ValueError("chat_template_kwargs must be a string-keyed mapping")
    kwargs = dict(chat_template_kwargs)
    forbidden = _PIPELINE_TEMPLATE_KWARGS.intersection(kwargs)
    if forbidden:
        raise ValueError(f"chat_template_kwargs cannot override {sorted(forbidden)}")
    if "truncate_history_thinking" in kwargs:
        truncate = kwargs["truncate_history_thinking"]
        if not isinstance(truncate, bool):
            raise ValueError("chat_template_kwargs.truncate_history_thinking must be a boolean")
        if "truncate_history_thinking" not in template:
            if "preserve_thinking" in template:
                kwargs.pop("truncate_history_thinking")
                kwargs["preserve_thinking"] = not truncate
            elif "clear_thinking" in template:
                kwargs.pop("truncate_history_thinking")
                kwargs["clear_thinking"] = truncate
    if "enable_thinking" in kwargs and "enable_thinking" not in template:
        try:
            parameters = inspect.signature(hf.apply_chat_template).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "thinking" in parameters and "enable_thinking" not in parameters:
            kwargs.setdefault("thinking", kwargs.pop("enable_thinking"))
    return kwargs


def _find_tokens(ids, pattern, start):
    for i in range(start, len(ids) - len(pattern) + 1):
        if ids[i : i + len(pattern)] == pattern:
            return i
    return -1


def _single_sequence(values, *, dtype, name):
    """Normalize an unbatched sequence or a one-row tokenizer tensor."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Expected a single {name} sequence, got shape {array.shape}")
    return array


def _input_ids(encoded):
    # HF versions/custom tokenizers may return a list or a BatchEncoding even
    # without return_dict=True. Converting the mapping itself reads its keys.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    return _single_sequence(encoded, dtype=np.int64, name="input_ids")


def _chatml_masks(hf, ids, messages, tools, template):
    """Bridge-compatible ChatML boundaries in the full rendered token stream.

    Do not tokenize individual turns: reasoning templates can render historical
    turns differently. Ambiguous payload markers still fail closed rather than
    being mistaken for structural assistant boundaries.
    """
    for text in _strings([messages, tools]):
        if _CHATML_START in text or _CHATML_END in text:
            raise ValueError("Chat payload contains an assistant delimiter; mask is ambiguous")

    def encode(text):
        return _input_ids(hf(text, add_special_tokens=False)).tolist()

    start_tokens = encode(_CHATML_START)
    end_tokens = encode(_CHATML_END)
    end_with_newline = encode(_CHATML_END + "\n")
    if not start_tokens or not end_tokens or not end_with_newline:
        raise ValueError("ChatML delimiters must tokenize to nonempty sequences")
    trim_prefixes = []
    if all(marker in template for marker in ("truncate_history_thinking", "<think>", "</think>")):
        # These are exact token prefixes, not a rule to drop reasoning. Match
        # Bridge's boundary fallback, including the opening '<think>\n' prefix.
        trim_prefixes = [encode("<think></think>"), encode("<think>\n")]
    tokens = ids.tolist()
    content_mask = np.zeros(ids.shape, dtype=bool)
    end_mask = np.zeros(ids.shape, dtype=bool)
    cursor, turns = 0, 0
    while (start := _find_tokens(tokens, start_tokens, cursor)) >= 0:
        start += len(start_tokens)
        end = _find_tokens(tokens, end_tokens, start)
        nested = _find_tokens(tokens, start_tokens, start)
        if end < 0 or (nested >= 0 and nested < end):
            raise ValueError("Unterminated or nested assistant span in chat template")
        after_end = end + len(end_tokens)
        # The primary Bridge terminator includes the newline. Its terminal
        # no-newline variant is legal too; never invent a token to complete it.
        if tokens[end : end + len(end_with_newline)] == end_with_newline:
            after_end = end + len(end_with_newline)
        while start < end:
            for prefix in trim_prefixes:
                if (
                    prefix
                    and start + len(prefix) <= end
                    and tokens[start : start + len(prefix)] == prefix
                ):
                    start += len(prefix)
                    break
            else:
                break
        content_mask[start:end] = True
        end_mask[end:after_end] = True
        cursor, turns = after_end, turns + 1
    if turns != sum(message["role"] == "assistant" for message in messages):
        raise ValueError("Assistant delimiters do not match the rendered conversation")
    return content_mask | end_mask, end_mask


def tokenize_chat(
    hf,
    messages,
    *,
    tools=None,
    template=None,
    loss_mode="assistant",
    assistant_start=None,
    assistant_end=None,
    add_generation_prompt=False,
    chat_template_kwargs=None,
):
    """Render once; mask tokens in that SAME rendering, never per-turn lengths.

    Recognized Qwen/ChatML templates follow Bridge's assistant end-token and
    thinking-prefix policy. Generation masks keep their content policy, with
    ChatML end tokens added. Other templates retain explicit delimiter masking.
    Explicit boundary fallback excludes the role prefix and includes the end
    delimiter. It rejects ambiguous payloads rather than training on forged
    role boundaries. Both paths keep the original template's token IDs intact.
    """
    messages, tools = normalize_chat(messages, tools)
    template = hf.get_chat_template(chat_template=template, tools=tools)
    chatml = (
        "<|im_start|>assistant" in template
        and _CHATML_END in template
        and assistant_start in (None, _CHATML_START)
        and assistant_end in (None, _CHATML_END, _CHATML_END + "\n")
    )
    kwargs = dict(
        tools=tools,
        chat_template=template,
        add_generation_prompt=add_generation_prompt,
        **_resolve_template_kwargs(hf, template, chat_template_kwargs),
    )
    if loss_mode == "full":
        ids = _input_ids(hf.apply_chat_template(messages, tokenize=True, **kwargs))
        mask = np.ones(ids.shape, dtype=bool)
        _mask_chatml_control_tokens(hf, ids, mask, chatml)
        return ids, np.where(mask, ids, IGNORE_INDEX)
    if loss_mode != "assistant":
        raise ValueError(f"Unknown chat loss mode: {loss_mode}")
    if add_generation_prompt:
        raise ValueError("Assistant supervision expects completed turns, not a generation prompt")
    generation_mask = re.search(r"\{%[-+]?\s*generation\b", template) is not None
    if generation_mask:
        encoded = hf.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, **kwargs
        )
        ids = _input_ids(encoded)
        raw_mask = encoded.get("assistant_masks")
        mask = _single_sequence(
            [] if raw_mask is None else raw_mask, dtype=bool, name="assistant_masks"
        )
        if mask.shape != ids.shape and not chatml:
            raise ValueError("Chat template returned an invalid assistant mask")
    if chatml:
        if not generation_mask:
            ids = _input_ids(hf.apply_chat_template(messages, tokenize=True, **kwargs))
        fallback, end_mask = _chatml_masks(hf, ids, messages, tools, template)
        # Bridge uses the fallback for absent/empty HF masks and otherwise
        # augments only turn endings; it does not replace a valid content mask.
        if not generation_mask or mask.shape != ids.shape or not mask.any():
            mask = fallback
        else:
            mask |= end_mask
    elif not generation_mask:
        if not assistant_start or not assistant_end:
            raise ValueError(
                "Assistant loss requires a generation-tagged template or explicit "
                "--sft-assistant-start and --sft-assistant-end delimiters"
            )
        if not hf.is_fast:
            raise ValueError("Assistant boundary masking requires a fast tokenizer")
        for text in _strings([messages, tools]):
            if assistant_start in text or assistant_end in text:
                raise ValueError("Chat payload contains an assistant delimiter; mask is ambiguous")
        rendered = hf.apply_chat_template(messages, tokenize=False, **kwargs)
        spans = []
        cursor = 0
        while (start := rendered.find(assistant_start, cursor)) >= 0:
            start += len(assistant_start)
            end = rendered.find(assistant_end, start)
            nested = rendered.find(assistant_start, start)
            if end < 0 or (nested >= 0 and nested < end):
                raise ValueError("Unterminated or nested assistant span in chat template")
            cursor = end + len(assistant_end)
            spans.append((start, cursor))
        if len(spans) != sum(m["role"] == "assistant" for m in messages):
            raise ValueError("Assistant delimiters do not match the rendered conversation")
        encoded = hf(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = _input_ids(encoded)
        mask = np.zeros(ids.shape, dtype=bool)
        span_index = 0
        for i, (start, end) in enumerate(encoded["offset_mapping"]):
            while span_index < len(spans) and start >= spans[span_index][1]:
                span_index += 1
            if span_index < len(spans) and end > start:
                left, right = spans[span_index]
                # Overlap includes a BPE token straddling the body boundary.
                mask[i] = end > left and start < right
    _mask_chatml_control_tokens(hf, ids, mask, chatml)
    if ids.ndim != 1 or not mask[1:].any():
        raise ValueError("No next-token assistant targets in this conversation")
    return ids, np.where(mask, ids, IGNORE_INDEX)


def _mask_chatml_control_tokens(hf, ids, mask, chatml):
    if chatml:
        skipped = [
            token_id
            for token_id, token in getattr(hf, "added_tokens_decoder", {}).items()
            if str(token) in _CHATML_SKIPPED_TOKENS
        ]
        mask[np.isin(ids, skipped)] = False
