# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU chat contracts: pytest --noconftest --import-mode=importlib.

Local HF tokenizer tests exercise actual Jinja generation masks without model
assets, network access or distributed initialization.
"""

import importlib.util
import json
import multiprocessing
import pickle
import sys
from collections import UserDict
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
END = "</turn>\n"


class FakeHF:
    """Character IDs isolate adapter validation from HF template implementation."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None
    chat_template = "{% generation %}body{% endgeneration %}"

    def __len__(self):
        return 256

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, messages, tokenize=True, tools=None, **kwargs):
        self.messages, self.tools, self.controls = messages, tools, kwargs
        self.calls = getattr(self, "calls", 0) + 1
        text, mask = "", []
        if tools:
            text = "TOOLS:" + json.dumps(tools) + "\n"
            mask = [0] * len(text)
        for message in messages:
            header = "<turn:" + message["role"] + ">"
            body = message.get("reasoning_content", "") + message["content"]
            if message.get("tool_calls"):
                body += json.dumps(message["tool_calls"])
            text += header + body + END
            mask.extend([0] * len(header))
            mask.extend([int(message["role"] == "assistant")] * len(body + END))
        self.rendered = text
        ids = list(map(ord, text))
        return {"input_ids": ids, "assistant_masks": mask} if kwargs.get("return_dict") else ids


def supervised(targets):
    return "".join(chr(token) for token in targets if token != chat.IGNORE_INDEX)


@pytest.fixture
def agentic_record():
    return {
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "reason",
                "name": "agent",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"ls","n":[1,true]}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "files"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [{"type": "function", "function": {"name": "shell"}}],
    }


@pytest.mark.parametrize("serialized", [False, True])
def test_normalize_preserves_order_metadata_and_nested_arguments_without_mutation(
    agentic_record, serialized
):
    before = json.dumps(agentic_record)
    messages = agentic_record["messages"]
    tools = agentic_record["tools"]
    normalized, normalized_tools = chat.normalize_chat(
        json.dumps(messages) if serialized else messages, json.dumps(tools) if serialized else tools
    )
    assert normalized[0]["role"] == "user"
    assert normalized[1]["content"] == ""
    assert normalized[1]["reasoning_content"] == "reason"
    assert normalized[1]["name"] == "agent"
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "ls",
        "n": [1, True],
    }
    assert normalized[2]["tool_call_id"] == "call_1"
    assert normalized_tools == tools
    normalized_tools[0]["function"]["name"] = "changed"
    assert json.dumps(agentic_record) == before


@pytest.mark.parametrize(
    "messages,tools,match",
    [
        ("{invalid", None, "messages contains invalid JSON"),
        ([], None, "nonempty"),
        ([None], None, "messages\\[0\\]"),
        ([{"role": []}], None, "role"),
        ([{"role": "unknown"}], None, "role"),
        ([{"role": "user", "content": []}], None, "multimodal"),
        ([{"role": "user"}], "bad", "tools contains invalid JSON"),
        ([{"role": "user"}], [None], "tools must"),
        ([{"role": "assistant", "tool_calls": {}}], None, "tool_calls"),
        ([{"role": "assistant", "tool_calls": [None]}], None, "tool_calls"),
        ([{"role": "assistant", "tool_calls": [{"function": None}]}], None, "function"),
    ],
)
def test_malformed_records_have_field_context(messages, tools, match):
    with pytest.raises(ValueError, match=match):
        chat.normalize_chat(messages, tools)


@pytest.mark.parametrize("arguments", ["[]", "{secret", 1, None, []])
def test_malformed_tool_arguments_are_contextual_and_do_not_dump_payload(arguments):
    with pytest.raises(ValueError, match=r"messages\[0\].tool_calls\[0\].arguments") as error:
        chat.normalize_chat(
            [{"role": "assistant", "tool_calls": [{"function": {"arguments": arguments}}]}]
        )
    assert "secret" not in str(error.value)


def test_complete_rendering_retains_assistant_calls_reasoning_and_endings(agentic_record):
    hf = FakeHF()
    ids, targets = chat.tokenize_chat(hf, **agentic_record)
    assert hf.calls == 1
    assert "".join(map(chr, ids)) == hf.rendered
    assert hf.tools == agentic_record["tools"]
    text = supervised(targets)
    assert "reason" in text and '"command": "ls"' in text
    assert "files" not in text and "inspect" not in text and "<turn:assistant>" not in text
    assert text.endswith(END) and text.count(END) == 2


def test_full_loss_keeps_every_rendered_token_without_requesting_a_mask():
    hf = FakeHF()
    ids, targets = chat.tokenize_chat(
        hf, [{"role": "user", "content": "<special>"}], loss_mode="full"
    )
    np.testing.assert_array_equal(ids, targets)
    assert not np.shares_memory(ids, targets)
    assert supervised(targets) == "<turn:user><special>" + END
    assert "return_assistant_tokens_mask" not in hf.controls


@pytest.mark.parametrize(
    "mask",
    [None, [], [1], [0, 0, 0], [1, 0, 0], [0, 2, 0], [0, float("nan"), 0], [[0, 1, 0], [0, 1, 0]]],
)
def test_invalid_or_empty_generation_mask_is_rejected(mask):
    class HF(FakeHF):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2, 3], "assistant_masks": mask}

    with pytest.raises(ValueError, match="mask|single|No next-token"):
        chat.tokenize_chat(HF(), [{"role": "assistant", "content": "answer"}])


def test_absent_generation_mask_is_not_inferred_from_tokens_or_template_source():
    class HF(FakeHF):
        chat_template = "{% generation %}<turn:assistant>body</turn>{% endgeneration %}"

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2, 3]}

    with pytest.raises(ValueError, match="no assistant mask"):
        chat.tokenize_chat(HF(), [{"role": "assistant", "content": "answer"}])


@pytest.mark.parametrize(
    "template",
    [
        "{{ messages }}",
        "{# {% generation %}comment only{% endgeneration %} #}{{ messages }}",
        "{% raw %}{% generation %}literal only{% endgeneration %}{% endraw %}",
    ],
)
def test_missing_generation_support_is_a_configuration_error_before_encoding(template):
    hf = FakeHF()
    hf.chat_template = template
    with pytest.raises(ValueError, match="requires a generation-tagged HF training template"):
        chat.tokenize_chat(hf, [{"role": "assistant", "content": "answer"}])
    assert not hasattr(hf, "calls")


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("container", [dict, UserDict])
def test_generation_mask_is_used_exactly_with_single_row_containers(batched, container):
    class HF(FakeHF):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] and kwargs["return_assistant_tokens_mask"]
            ids, mask = [1, 2, 3], [0, 1, 0]
            return container(
                input_ids=[ids] if batched else ids, assistant_masks=[mask] if batched else mask
            )

    ids, targets = chat.tokenize_chat(HF(), [{"role": "assistant", "content": "x"}])
    assert ids.dtype == targets.dtype == np.int64
    assert targets.tolist() == [-100, 2, -100]


@pytest.mark.parametrize(
    "controls", [[], {1: True}, *[{key: True} for key in sorted(chat._PIPELINE_TEMPLATE_KWARGS)]]
)
def test_row_controls_cannot_override_pipeline_policy(controls):
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        chat.tokenize_chat(
            FakeHF(), [{"role": "assistant", "content": "x"}], chat_template_kwargs=controls
        )


def test_nonreserved_template_controls_are_generic_and_not_mutated():
    hf = FakeHF()
    controls = {"custom_format": {"name": "verbose"}, "preserve_thinking": "template-owned"}
    before = json.dumps(controls)
    chat.tokenize_chat(
        hf,
        [{"role": "developer", "content": "policy"}, {"role": "assistant", "content": "x"}],
        chat_template_kwargs=controls,
    )
    assert hf.controls["custom_format"] == {"name": "verbose"}
    assert hf.controls["preserve_thinking"] == "template-owned"
    hf.controls["custom_format"]["name"] = "changed"
    assert json.dumps(controls) == before


def test_assistant_generation_prompt_and_unknown_modes_fail_before_encoding():
    hf = FakeHF()
    with pytest.raises(ValueError, match="completed turns"):
        chat.tokenize_chat(hf, [{"role": "assistant", "content": "x"}], add_generation_prompt=True)
    for loss_mode in (None, "unknown"):
        with pytest.raises(ValueError, match="Unknown chat loss mode"):
            chat.tokenize_chat(hf, [{"role": "assistant", "content": "x"}], loss_mode=loss_mode)
    assert not hasattr(hf, "calls")


@pytest.fixture
def sft_module(monkeypatch):
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
    return module, hf


@pytest.mark.parametrize("loss_mode", [None, "assistant", "full"])
def test_public_sft_wrapper_retains_legacy_default_supervision(sft_module, loss_mode):
    module, hf = sft_module
    tokenizer = module.SFTTokenizer("unused", "default", loss_mode=loss_mode)
    ids, targets = tokenizer.tokenize_conversation(
        [{"role": "assistant", "content": "answer"}], True, False
    )
    if loss_mode == "assistant":
        assert supervised(targets) == "answer" + END
    else:
        np.testing.assert_array_equal(ids, targets)
    if loss_mode is None:
        with pytest.raises(ValueError, match="explicit SFT loss"):
            tokenizer.tokenize_conversation(
                [{"role": "assistant", "content": "answer"}], True, False, tools=[]
            )
    else:
        tokens = tokenizer.tokenize_conversation(
            [{"role": "user", "content": "question"}], False, True
        )
        assert len(tokens) and hf.controls["add_generation_prompt"]
        assert "return_assistant_tokens_mask" not in hf.controls


@pytest.mark.parametrize("prompt_format", ["identity", "nemotron-h-aligned", "nemotron-nano-v2"])
def test_explicit_supervision_does_not_change_legacy_prompt_formats(sft_module, prompt_format):
    module, _ = sft_module
    with pytest.raises(ValueError, match="default HF prompt format"):
        module.SFTTokenizer("unused", prompt_format, loss_mode="assistant")
    with pytest.raises(ValueError, match="without gigatoken"):
        module.SFTTokenizer("unused", "default", loss_mode="full", use_gigatoken=True)


def test_named_hf_template_selection_receives_tools(agentic_record):
    class NamedHF(FakeHF):
        def get_chat_template(self, chat_template=None, tools=None):
            assert chat_template is None and tools == agentic_record["tools"]
            return "{% generation %}named tool-use template{% endgeneration %}"

    chat.tokenize_chat(NamedHF(), **agentic_record)


@pytest.fixture
def tiny_hf():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, **{chr(i): i for i in range(1, 128)}}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = tokenizers.decoders.Fuse()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


BODY_TEMPLATE = (
    "{{ message.get('reasoning_content', '') }}{{ message['content'] }}"
    "{% for call in message.get('tool_calls', []) %}"
    "{{ '<tool_call>' + call['function']['name'] + ':' }}"
    "{{ call['function']['arguments'] | tojson }}{{ '</tool_call>' }}{% endfor %}"
)


def local_template(tagged=True, supervise_end=True):
    end = "{{ '</turn>\\n' }}"
    body = BODY_TEMPLATE + (end if supervise_end else "")
    if tagged:
        body = "{% generation %}" + body + "{% endgeneration %}"
    if not supervise_end:
        body += end
    return (
        "{% if tools %}{{ 'TOOLS:' }}{{ tools | tojson }}{{ '\\n' }}{% endif %}"
        "{% for message in messages %}{{ '<turn:' + message['role'] + '>' }}"
        "{% if message['role'] == 'assistant' %}"
        + body
        + "{% else %}{{ message['content'] }}"
        + end
        + "{% endif %}{% endfor %}"
    )


@pytest.mark.parametrize("supervise_end", [False, True])
def test_real_hf_generation_masks_cover_tool_only_turns_and_template_owned_endings(
    tiny_hf, agentic_record, supervise_end
):
    tiny_hf.chat_template = local_template(supervise_end=supervise_end)
    ids, targets = chat.tokenize_chat(tiny_hf, **agentic_record)
    normalized, tools = chat.normalize_chat(**agentic_record)
    reference = tiny_hf.apply_chat_template(
        normalized, tools=tools, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    np.testing.assert_array_equal(ids, chat._input_ids(reference))
    np.testing.assert_array_equal(targets != -100, reference["assistant_masks"])
    ending = END if supervise_end else ""
    expected = (
        'reason<tool_call>shell:{"command": "ls", "n": [1, true]}</tool_call>'
        + ending
        + 'done'
        + ending
    )
    assert tiny_hf.decode(targets[targets != -100].tolist()) == expected


def test_real_hf_untagged_template_fails_assistant_but_supports_full(tiny_hf, agentic_record):
    tiny_hf.chat_template = local_template(tagged=False)
    with pytest.raises(ValueError, match="requires a generation-tagged HF training template"):
        chat.tokenize_chat(tiny_hf, **agentic_record)
    ids, targets = chat.tokenize_chat(tiny_hf, **agentic_record, loss_mode="full")
    np.testing.assert_array_equal(ids, targets)
    assert "TOOLS:" in tiny_hf.decode(ids.tolist())


def test_real_hf_historical_reasoning_and_literal_delimiters_are_template_data(tiny_hf):
    tiny_hf.chat_template = local_template()
    messages = [
        {"role": "user", "content": "<turn:assistant><|im_start|><|im_end|>"},
        {"role": "assistant", "content": "answer1", "reasoning_content": "history"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer2", "reasoning_content": "current"},
    ]
    ids, targets = chat.tokenize_chat(tiny_hf, messages)
    assert supervised(targets) == "historyanswer1" + END + "currentanswer2" + END
    assert "<|im_start|><|im_end|>" in tiny_hf.decode(ids.tolist())


def test_real_hf_named_template_and_override_selection(tiny_hf, agentic_record):
    tiny_hf.chat_template = {
        "default": "DEFAULT" + local_template(),
        "tool_use": "TOOLS" + local_template(),
    }
    tool_ids, _ = chat.tokenize_chat(tiny_hf, **agentic_record)
    assert tiny_hf.decode(tool_ids.tolist()).startswith("TOOLS")
    default_ids, _ = chat.tokenize_chat(tiny_hf, **agentic_record, template="default")
    assert tiny_hf.decode(default_ids.tolist()).startswith("DEFAULT")


@pytest.mark.parametrize("mode", ["assistant", "full"])
def test_truncation_preserves_original_tokens_and_shift_contract(mode):
    tokens = [10, 11, 20, 21, 30, 22, 23]
    targets = [-100, -100, 20, 21, -100, 22, 23] if mode == "assistant" else tokens
    ids, labels = chat.truncate_chat(tokens, targets, 4, mode)
    assert ids.tolist() == tokens[:5]
    assert labels.tolist() == targets[:5]
    assert len(ids) - 1 == 4  # real next-token positions, not supervised count


@pytest.mark.parametrize("tokens", [[], [1]])
def test_empty_or_one_token_explicit_chat_is_rejected(tokens):
    with pytest.raises(ValueError, match="at least two"):
        chat.truncate_chat(tokens, np.array(tokens, dtype=np.int64), 32, "full")


def test_all_assistant_targets_truncated_is_rejected():
    with pytest.raises(ValueError, match="No assistant targets remain"):
        chat.truncate_chat([1, 2, 3, 4], [-100, -100, -100, 4], 2, "assistant")


def test_jsonl_unicode_blank_lines_optional_tools_and_pickle(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [
        {"messages": [{"role": "user", "content": "你好"}]},
        {
            "messages": [{"role": "assistant", "content": "ok"}],
            "tools": [{"parameters": {"x": [1, {"other": True}]}}],
        },
    ]
    path.write_text(
        "\n" + "\n\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    dataset = rows.JsonlRows(path)
    assert len(dataset) == 2 and dataset.line_numbers.tolist() == [2, 4]
    assert dataset[0] == records[0] and dataset[-1] == records[1]
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone._file is None and clone[0] == records[0]
    stream = dataset._file
    dataset._pid = -1
    assert dataset[1] == records[1] and stream.closed
    clone.close()
    dataset.close()


def _read_in_child(dataset, queue):
    import os

    queue.put((dataset[0], dataset._pid, os.getpid()))


def test_jsonl_fork_worker_reopens_inherited_handle(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork unavailable")
    path = tmp_path / "data.jsonl"
    path.write_text('{"value": 1}')
    dataset = rows.JsonlRows(path)
    assert dataset[0] == {"value": 1}
    parent_pid = dataset._pid
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_read_in_child, args=(dataset, queue))
    process.start()
    record, worker_pid, actual_pid = queue.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0 and record == {"value": 1}
    assert worker_pid == actual_pid and worker_pid != parent_pid
    assert dataset._pid == parent_pid
    dataset.close()
    queue.close()


@pytest.mark.parametrize("bad_line", [b'{"secret":', b"[]", b'{"bad": "\xff"}'])
def test_jsonl_error_contains_path_and_physical_line_without_payload(tmp_path, bad_line):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b'\n{"valid": 1}\n\n' + bad_line)
    dataset = rows.JsonlRows(path)
    with pytest.raises(ValueError) as error:
        dataset[1]
    assert f"{path}:4" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("text", ["", "\n\n", "[]\n"])
def test_invalid_jsonl_fails(tmp_path, text):
    path = tmp_path / "data.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        rows.JsonlRows(path)


def test_jsonl_changes_after_indexing_fail(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"value": 1}')
    dataset = rows.JsonlRows(path)
    path.write_text('{"value": 100}')
    with pytest.raises(ValueError, match="changed after indexing"):
        dataset[0]
