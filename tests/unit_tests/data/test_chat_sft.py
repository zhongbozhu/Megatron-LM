# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only contracts; use pytest --noconftest --import-mode=importlib.

Load these dependency-light modules directly to avoid importing CUDA MCore.
FakeHF tests mask alignment, not HF tokenizer or distributed runtime behavior.
"""

import importlib.util
import json
import os
import pickle
import sys
from collections import UserDict
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chat = load_source("_chat_sft_test", "megatron/core/tokenizers/text/libraries/chat_sft.py")
rows = load_source("_jsonl_rows_test", "megatron/training/datasets/jsonl_rows.py")
START, END = "<|im_start|>assistant\n", "<|im_end|>"


class FakeHF:
    """One character per token, preserving trailing template whitespace."""

    is_fast = True
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None
    chat_template = "fake template without generation tags"

    def __len__(self):
        return 256

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or "fake template without generation tags"

    def apply_chat_template(self, messages, tokenize=False, tools=None, **kwargs):
        self.messages, self.tools = messages, tools
        text = ""
        if tools:
            text += "<|im_start|>system\n" + json.dumps(tools) + END + "\n"
        for message in messages:
            text += "<|im_start|>" + message["role"] + "\n"
            text += message.get("reasoning_content", "") + message["content"]
            if message.get("tool_calls"):
                text += json.dumps(message["tool_calls"])
            text += END + "\n"
        self.rendered = text
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def tokenize(hf, messages, **kwargs):
    return chat.tokenize_chat(hf, messages, assistant_start=START, assistant_end=END, **kwargs)


@pytest.mark.parametrize("loss_mode", [None, "assistant", "full"])
def test_sft_tokenizer_public_method_retains_legacy_default(monkeypatch, loss_mode):
    monkeypatch.setitem(sys.modules, "megatron.core.tokenizers.text.libraries.chat_sft", chat)
    module = load_source(
        "_sft_tokenizer_test", "megatron/core/tokenizers/text/libraries/sft_tokenizer.py"
    )
    hf = FakeHF()
    monkeypatch.setattr(module, "HAVE_TRANSFORMERS", True)
    monkeypatch.setattr(
        module,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda **kwargs: hf)),
        raising=False,
    )
    tokenizer = module.SFTTokenizer(
        "unused", "default", loss_mode=loss_mode, assistant_start=START, assistant_end=END
    )
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    ids, targets = tokenizer.tokenize_conversation(messages, True, False)
    if loss_mode == "assistant":
        assert "".join(chr(i) for i in targets if i != -100) == "answer" + END
    else:
        np.testing.assert_array_equal(ids, targets)


def test_sft_hf_named_template_selection_is_not_overridden(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.core.tokenizers.text.libraries.chat_sft", chat)
    module = load_source(
        "_sft_tokenizer_test", "megatron/core/tokenizers/text/libraries/sft_tokenizer.py"
    )

    class NamedHF(FakeHF):
        chat_template = {"default": "no tools", "tool_use": "has tools"}

        def get_chat_template(self, chat_template=None, tools=None):
            assert chat_template is None
            assert tools
            return self.chat_template["tool_use"]

    hf = NamedHF()
    monkeypatch.setattr(module, "HAVE_TRANSFORMERS", True)
    monkeypatch.setattr(
        module,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda **kwargs: hf)),
        raising=False,
    )
    tokenizer = module.SFTTokenizer(
        "unused", "default", loss_mode="assistant", assistant_start=START, assistant_end=END
    )
    tokenizer.tokenize_conversation(
        [{"role": "assistant", "content": "answer"}],
        True,
        False,
        tools=[{"function": {"name": "shell"}}],
    )


def test_assistant_body_eos_not_headers_prompts_or_tool_results():
    hf = FakeHF()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "final"},
    ]
    ids, targets = tokenize(hf, messages)
    assert "".join(map(chr, ids)) == hf.rendered
    supervised = "".join(chr(i) for i in targets if i != -100)
    assert supervised == "answer" + END + "final" + END
    assert ids[-1] == ord("\n") and targets[-1] == -100


def test_tool_only_assistant_and_json_arguments_are_preserved_without_mutation():
    hf = FakeHF()
    messages = [
        {"role": "user", "content": "run a command"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "reason",
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {"role": "tool", "content": "files", "tool_call_id": "x"},
    ]
    tools = [{"type": "function", "function": {"name": "shell"}}]
    _, targets = tokenize(hf, json.dumps(messages), tools=json.dumps(tools))
    assert hf.messages[0]["role"] == "user"  # no synthetic system turn
    assert hf.tools == tools
    assert hf.messages[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert messages[1]["content"] is None
    assert isinstance(messages[1]["tool_calls"][0]["function"]["arguments"], str)
    supervised = "".join(chr(i) for i in targets if i != -100)
    assert "reason" in supervised and '"command": "ls"' in supervised
    assert "files" not in supervised


def test_full_mode_preserves_all_tokens():
    ids, targets = chat.tokenize_chat(
        FakeHF(), [{"role": "user", "content": "hi"}], loss_mode="full"
    )
    np.testing.assert_array_equal(ids, targets)


def test_generation_mask_is_preferred_and_controls_eos_policy():
    class TaggedHF(FakeHF):
        def get_chat_template(self, **kwargs):
            return "{% generation %}answer{% endgeneration %}"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_assistant_tokens_mask"] and kwargs["return_dict"]
            return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 1, 1, 0]}

    _, targets = chat.tokenize_chat(TaggedHF(), [{"role": "assistant", "content": "x"}])
    assert targets.tolist() == [-100, 2, 3, -100]


def test_boundary_straddling_bpe_token_is_supervised():
    class BPEHF(FakeHF):
        def __call__(self, text, **kwargs):
            start = text.index(START) + len(START)
            return {
                "input_ids": [1, 2, 3],
                "offset_mapping": [(0, start - 1), (start - 1, start + 1), (start + 1, len(text))],
            }

    _, targets = tokenize(BPEHF(), [{"role": "assistant", "content": "x"}])
    assert targets.tolist() == [-100, 2, 3]


@pytest.mark.parametrize("payload", [START, END])
def test_ambiguous_payload_fails(payload):
    with pytest.raises(ValueError, match="ambiguous"):
        tokenize(FakeHF(), [{"role": "assistant", "content": payload}])


def test_missing_delimiters_and_no_assistant_fail():
    with pytest.raises(ValueError, match="generation-tagged"):
        chat.tokenize_chat(FakeHF(), [{"role": "assistant", "content": "x"}])
    with pytest.raises(ValueError, match="No next-token"):
        tokenize(FakeHF(), [{"role": "user", "content": "x"}])


def test_slow_tokenizer_and_generation_prompt_fail():
    hf = FakeHF()
    hf.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        tokenize(hf, [{"role": "assistant", "content": "x"}])
    with pytest.raises(ValueError, match="completed turns"):
        tokenize(FakeHF(), [{"role": "assistant", "content": "x"}], add_generation_prompt=True)


@pytest.mark.parametrize("arguments", ["[]", 1, None])
def test_non_object_tool_arguments_fail(arguments):
    with pytest.raises(ValueError, match="JSON object"):
        chat.normalize_chat(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "x", "arguments": arguments}}],
                }
            ]
        )


class ChatMLHF(FakeHF):
    chat_template = "<|im_start|>assistant\n{{ content }}<|im_end|>\n"

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.template_kwargs = kwargs.copy()
        result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
        return {"input_ids": result} if tokenize and kwargs.get("return_dict") else result


@pytest.mark.parametrize("mode", ["assistant", "full", "generation"])
@pytest.mark.parametrize("container", [list, dict, UserDict])
@pytest.mark.parametrize("batched", [False, True])
def test_chat_template_return_containers_preserve_tokens_and_mask(mode, container, batched):
    class ReturnedHF(ChatMLHF):
        def get_chat_template(self, **kwargs):
            template = super().get_chat_template(**kwargs)
            return template + "{% generation %}" if mode == "generation" else template

        def apply_chat_template(self, messages, **kwargs):
            text = FakeHF.apply_chat_template(self, messages, tokenize=False)
            ids = [ord(c) for c in text]
            start = text.index(START) + len(START)
            # Deliberately supervise only the first body token. Flattening must
            # preserve this valid HF mask, not replace it with a full-body fallback.
            mask = [int(i == start) for i in range(len(ids))]
            values = np.array([ids]) if batched else ids
            if mode == "generation":
                result = {
                    "input_ids": values,
                    "assistant_masks": np.array([mask]) if batched else mask,
                }
                return UserDict(result) if container is UserDict else result
            if container is list:
                return values
            return container(input_ids=values, attention_mask=[1] * len(ids))

    hf = ReturnedHF()
    ids, targets = tokenize(
        hf,
        [{"role": "assistant", "content": "answer"}],
        loss_mode="full" if mode == "full" else "assistant",
    )
    assert ids.ndim == targets.ndim == 1
    assert ids.dtype == targets.dtype == np.int64
    assert "".join(map(chr, ids)) == hf.rendered
    expected = (
        hf.rendered if mode == "full" else ("a" if mode == "generation" else "answer") + END + "\n"
    )
    assert "".join(chr(i) for i in targets if i != -100) == expected


@pytest.mark.parametrize("loss_mode", ["assistant", "full"])
def test_chat_template_rejects_multiple_sequences_in_one_row(loss_mode):
    class BatchedHF(ChatMLHF):
        def apply_chat_template(self, messages, **kwargs):
            return UserDict(input_ids=[[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="single"):
        tokenize(BatchedHF(), [{"role": "assistant", "content": "answer"}], loss_mode=loss_mode)


@pytest.mark.parametrize("configured_end", [None, END, END + "\n"])
def test_chatml_matches_bridge_end_newline_policy(configured_end):
    hf = ChatMLHF()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "final"},
    ]
    ids, targets = chat.tokenize_chat(
        hf,
        messages,
        assistant_start=START if configured_end else None,
        assistant_end=configured_end,
    )
    assert "".join(map(chr, ids)) == hf.rendered
    assert "".join(chr(i) for i in targets if i != -100) == "answer" + END + "\nfinal" + END + "\n"


def test_chatml_terminal_turn_without_newline_does_not_invent_one():
    class NoNewlineHF(ChatMLHF):
        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            rendered = super().apply_chat_template(messages, tokenize=False, **kwargs).rstrip("\n")
            return self(rendered)["input_ids"] if tokenize else rendered

    ids, targets = tokenize(NoNewlineHF(), [{"role": "assistant", "content": "answer"}])
    assert "".join(map(chr, ids)).endswith(END)
    assert "".join(chr(i) for i in targets if i != -100) == "answer" + END


@pytest.mark.parametrize("mask_kind", ["content", "empty", "missing", "wrong_length"])
def test_chatml_generation_mask_augments_end_or_falls_back(mask_kind):
    class TaggedHF(ChatMLHF):
        chat_template = START + "{% generation %}{{ content }}{% endgeneration %}" + END + "\n"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_assistant_tokens_mask"]
            encoded = super().apply_chat_template(messages, **kwargs)
            mask = [0] * len(encoded["input_ids"])
            start = self.rendered.index(START) + len(START)
            mask[start : start + len("answer")] = [1] * len("answer")
            if mask_kind == "content":
                encoded["assistant_masks"] = mask
            elif mask_kind == "empty":
                encoded["assistant_masks"] = [0] * len(mask)
            elif mask_kind == "wrong_length":
                encoded["assistant_masks"] = [1]
            return encoded

    _, targets = tokenize(TaggedHF(), [{"role": "assistant", "content": "answer"}])
    assert "".join(chr(i) for i in targets if i != -100) == "answer" + END + "\n"


def test_chatml_bridge_token_boundary_regression():
    """Same token/mask fixture as Bridge's no-generation-keyword chat test."""

    class BoundaryHF(ChatMLHF):
        def __call__(self, text, **kwargs):
            return {"input_ids": {START: [101], END: [102], END + "\n": [102, 103]}[text]}

        def apply_chat_template(self, messages, **kwargs):
            return [100, 10, 102, 103, 101, 21, 22, 102, 103]

    ids, targets = tokenize(
        BoundaryHF(),
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    )
    assert ids.tolist() == [100, 10, 102, 103, 101, 21, 22, 102, 103]
    assert (targets != -100).tolist() == [False] * 5 + [True] * 4


@pytest.mark.parametrize(
    "prefix,supervised_prefix",
    [
        ("<think></think>", ""),
        ("<think>\nreason</think>\n", "reason</think>\n"),
        ("<think>reason</think>", "<think>reason</think>"),
        ("<think></think><think>\n", ""),
    ],
)
def test_chatml_thinking_fallback_matches_bridge_exact_prefix_trimming(prefix, supervised_prefix):
    hf = ChatMLHF()
    hf.chat_template += " truncate_history_thinking <think> </think>"
    ids, targets = tokenize(hf, [{"role": "assistant", "content": prefix + "answer"}])
    assert prefix + "answer" in "".join(map(chr, ids))  # trim the mask, never the input
    assert (
        "".join(chr(i) for i in targets if i != -100) == supervised_prefix + "answer" + END + "\n"
    )


def test_chatml_thinking_trim_is_not_applied_to_other_templates():
    _, targets = tokenize(ChatMLHF(), [{"role": "assistant", "content": "<think>\nanswer"}])
    assert "".join(chr(i) for i in targets if i != -100) == "<think>\nanswer" + END + "\n"


def test_chatml_generation_mask_keeps_its_thinking_content_policy():
    class TaggedHF(ChatMLHF):
        chat_template = (
            ChatMLHF.chat_template + " {% generation %} truncate_history_thinking <think> </think>"
        )

        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            start = self.rendered.index(START) + len(START)
            end = self.rendered.index(END, start)
            encoded["assistant_masks"] = [int(start <= i < end) for i in range(len(self.rendered))]
            return encoded

    _, targets = tokenize(TaggedHF(), [{"role": "assistant", "content": "<think>\nanswer"}])
    assert "".join(chr(i) for i in targets if i != -100) == "<think>\nanswer" + END + "\n"


@pytest.mark.parametrize("loss_mode", ["assistant", "full"])
def test_chatml_skips_control_tokens_but_not_assistant_end(loss_mode):
    class ControlHF(ChatMLHF):
        added_tokens_decoder = {200: "<|image_pad|>", 102: END}

        def __call__(self, text, **kwargs):
            return {"input_ids": {START: [101], END: [102], END + "\n": [102, 103]}[text]}

        def apply_chat_template(self, messages, **kwargs):
            return [101, 21, 200, 22, 102, 103]

    _, targets = tokenize(ControlHF(), [{"role": "assistant", "content": "x"}], loss_mode=loss_mode)
    assert targets[2] == -100 and targets[-2:].tolist() == [102, 103]


@pytest.mark.parametrize(
    "template_control,expected",
    [
        ("truncate_history_thinking", {"truncate_history_thinking": False}),
        ("preserve_thinking", {"preserve_thinking": True}),
        ("clear_thinking", {"clear_thinking": False}),
    ],
)
def test_template_controls_forwarded_and_aliases_match_bridge(template_control, expected):
    hf = ChatMLHF()
    hf.chat_template += " " + template_control
    controls = {"truncate_history_thinking": False}
    tools = [{"function": {"name": "shell"}}]
    tokenize(
        hf, [{"role": "assistant", "content": "x"}], tools=tools, chat_template_kwargs=controls
    )
    assert {k: hf.template_kwargs[k] for k in expected} == expected
    assert hf.tools == tools and controls == {"truncate_history_thinking": False}


def test_enable_thinking_maps_to_custom_tokenizer_parameter():
    class ThinkingHF(ChatMLHF):
        def apply_chat_template(self, messages, thinking=None, **kwargs):
            self.thinking = thinking
            return super().apply_chat_template(messages, **kwargs)

    hf = ThinkingHF()
    tokenize(
        hf, [{"role": "assistant", "content": "x"}], chat_template_kwargs={"enable_thinking": False}
    )
    assert hf.thinking is False


@pytest.mark.parametrize(
    "controls",
    [
        [],
        {1: True},
        {"truncate_history_thinking": "false"},
        *[{key: True} for key in sorted(chat._PIPELINE_TEMPLATE_KWARGS)],
    ],
)
def test_template_controls_cannot_override_pipeline_policy(controls):
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        tokenize(ChatMLHF(), [{"role": "assistant", "content": "x"}], chat_template_kwargs=controls)


def test_sft_tokenizer_forwards_template_controls(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.core.tokenizers.text.libraries.chat_sft", chat)
    module = load_source(
        "_sft_controls_test", "megatron/core/tokenizers/text/libraries/sft_tokenizer.py"
    )
    hf = ChatMLHF()
    monkeypatch.setattr(module, "HAVE_TRANSFORMERS", True)
    monkeypatch.setattr(
        module,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda **kwargs: hf)),
        raising=False,
    )
    tokenizer = module.SFTTokenizer("unused", "default", loss_mode="assistant")
    tokenizer.tokenize_conversation(
        [{"role": "assistant", "content": "x"}],
        True,
        False,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert hf.template_kwargs["enable_thinking"] is False


@pytest.fixture
def bridge_chat_parity():
    """Opt-in CPU parity using installed Bridge and an existing HF snapshot.

    Set BRIDGE_CHAT_PARITY_TOKENIZER to the training tokenizer directory. No
    downloads, GPU or model weights are needed; missing deps fail if opted in.
    """
    path = os.environ.get("BRIDGE_CHAT_PARITY_TOKENIZER")
    if not path:
        pytest.skip("Set BRIDGE_CHAT_PARITY_TOKENIZER for real tokenizer/Bridge parity")
    from transformers import AutoTokenizer

    from megatron.bridge.data.conversation_processing import tokenize_chat_example
    from megatron.bridge.data.token_utils import extract_skipped_token_ids

    hf = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def compare(row, loss_mode):
        expected = tokenize_chat_example(
            row, hf, loss_mode=loss_mode, skipped_tokens=extract_skipped_token_ids(hf)
        )
        ids, targets = tokenize(
            hf,
            row.get("conversation", row.get("messages")),
            tools=row.get("tools"),
            chat_template_kwargs=row.get("chat_template_kwargs"),
            loss_mode=loss_mode,
        )
        np.testing.assert_array_equal(ids, expected.input_ids.numpy())
        np.testing.assert_array_equal(targets != -100, expected.assistant_mask.numpy())
        active = targets[1:] != -100
        np.testing.assert_array_equal(targets[1:][active], expected.input_ids.numpy()[1:][active])

    return compare


@pytest.mark.parametrize("loss_mode", ["assistant", "full"])
@pytest.mark.parametrize(
    "controls", [None, {"enable_thinking": False}, {"truncate_history_thinking": False}]
)
def test_real_tokenizer_matches_bridge_chat_preprocessing(bridge_chat_parity, loss_mode, controls):
    row = {
        "conversation": [
            {"role": "user", "content": "Inspect the current directory."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should list files.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "main.py"},
            {"role": "assistant", "content": "There is one Python file."},
            {"role": "user", "content": "What is its name?"},
            {
                "role": "assistant",
                "content": "main.py",
                "reasoning_content": "Use the tool result.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ],
        "chat_template_kwargs": controls,
    }
    bridge_chat_parity(row, loss_mode)


def test_real_coderforge_rows_match_bridge_chat_preprocessing(bridge_chat_parity):
    path = os.environ.get("BRIDGE_CHAT_PARITY_JSONL")
    if not path:
        pytest.skip("Set BRIDGE_CHAT_PARITY_JSONL to materialized training.jsonl")
    with Path(path).open(encoding="utf-8") as source:
        examples = list(islice((json.loads(line) for line in source if line.strip()), 32))
    assert examples, "No rows in the parity dataset"
    for index, row in enumerate(examples):
        try:
            bridge_chat_parity(row, "assistant")
        except Exception as error:
            raise AssertionError(
                f"Bridge parity failed for row {index}, trajectory_id={row.get('trajectory_id')}"
            ) from error


def test_jsonl_preserves_heterogeneous_fields_and_worker_pickle(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "\u4f60\u597d",
                    "tool_calls": [{"function": {"arguments": '{"x":1}'}}],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"arguments": {"other": [1, 2]}}}],
                }
            ],
            "tools": [{"function": {"name": "x"}}],
        },
    ]
    path.write_text(
        "\n" + "\n\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )
    dataset = rows.JsonlRows(path)
    assert len(dataset) == 2 and dataset[0] == records[0]
    assert dataset[-1] == records[1]
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone._file is None and clone[0] == records[0]
    assert clone._file is not dataset._file
    stream = dataset._file
    dataset._pid = -1
    assert dataset[1] == records[1] and stream.closed


@pytest.mark.parametrize("text", ["", "\n\n", "[]\n"])
def test_invalid_jsonl_fails(tmp_path, text):
    path = tmp_path / "data.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        rows.JsonlRows(path)
