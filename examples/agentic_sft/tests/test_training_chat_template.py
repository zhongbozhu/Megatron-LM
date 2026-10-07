# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU rendering/mask tests using the real, locally cached pinned Qwen tokenizer.

Set QWEN35_SOURCE_TOKENIZER when the original snapshot lives elsewhere. No
downloads or existing tokenizer modifications occur; preparation uses a tempdir.
"""

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/agentic_sft/01_prepare_qwen35_35B_1node_proxy.py"
SPEC = importlib.util.spec_from_file_location("prepare_training_template", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
UPSTREAM_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"


class TrainingChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise unittest.SkipTest(
                "Requires the prepared CPU container with Transformers"
            ) from error
        source = Path(
            os.environ.get(
                "QWEN35_SOURCE_TOKENIZER",
                str(
                    ROOT.parent
                    / "hf_home/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"
                    / PREPARE.HF_REVISION
                ),
            )
        )
        if not source.is_dir():
            raise unittest.SkipTest("Set QWEN35_SOURCE_TOKENIZER to the pinned local snapshot")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / "tokenizer"
        PREPARE.prepare_tokenizer(source, cls.output, revision=PREPARE.HF_REVISION)
        cls.original = AutoTokenizer.from_pretrained(source, local_files_only=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.output, local_files_only=True)
        cls.upstream = cls.original.get_chat_template()
        if hashlib.sha256(cls.upstream.encode()).hexdigest() != UPSTREAM_TEMPLATE_SHA256:
            raise AssertionError("Rendering oracle must use the pinned original upstream template")
        cls.template = PREPARE.TRAINING_TEMPLATE.read_text()

    def render(self, messages, *, tools=None, template=None, **controls):
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=template or self.template,
            tokenize=False,
            add_generation_prompt=False,
            **controls,
        )

    def assert_native_mask(self, messages, tools=None):
        rendered = self.render(messages, tools=tools)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        offsets = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        self.assertEqual(encoded["input_ids"], offsets["input_ids"])
        spans = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = self.render(messages[:index], tools=tools)
            after = self.render(messages[: index + 1], tools=tools)
            self.assertTrue(rendered.startswith(before))
            self.assertTrue(rendered.startswith(after))
            header = "<|im_start|>assistant\n"
            self.assertEqual(after[len(before) : len(before) + len(header)], header)
            self.assertTrue(after.endswith("<|im_end|>\n"))
            spans.append((len(before) + len(header), len(after)))
        expected = [
            int(any(end > left and start < right for left, right in spans))
            for start, end in offsets["offset_mapping"]
        ]
        self.assertEqual(encoded["assistant_masks"], expected)
        self.assertGreater(sum(expected), 0)
        self.assertLess(sum(expected), len(expected))
        # This includes the closing EOS and trailing newline of every assistant,
        # while excluding every role header, prompt and tool response.
        for left, right in spans:
            ending = rendered.rfind("<|im_end|>\n", left, right)
            self.assertGreaterEqual(ending, left)
            ending_tokens = [
                i
                for i, (start, end) in enumerate(offsets["offset_mapping"])
                if end > ending and start < right
            ]
            self.assertTrue(ending_tokens)
            self.assertTrue(all(encoded["assistant_masks"][i] for i in ending_tokens))
        return encoded

    def test_rendering_matches_old_policy_including_historical_reasoning(self):
        examples = [
            [
                {"role": "user", "content": "Compute two plus two."},
                {"role": "assistant", "content": "Four."},
            ],
            [
                {"role": "user", "content": "First question."},
                {
                    "role": "assistant",
                    "reasoning_content": "Historical reasoning.",
                    "content": "First answer.",
                },
                {"role": "user", "content": "Second question."},
                {"role": "assistant", "content": "<think>Current reasoning.</think>Second answer."},
            ],
        ]
        # Test-only oracle for the previous explicit qwen3_5 policy. Production
        # preparation/runtime uses the maintained asset, with no source patching.
        legacy = self.upstream.replace(
            "{%- if loop.index0 > ns.last_query_index %}",
            "{%- if preserve_thinking or loop.index0 > ns.last_query_index %}",
            1,
        )
        for messages in examples:
            with self.subTest(messages=messages):
                self.assertEqual(
                    self.render(messages),
                    self.render(messages, template=legacy, preserve_thinking=True),
                )
                self.assert_native_mask(messages)
        historical = self.render(examples[1])
        self.assertIn("Historical reasoning.", historical)
        self.assertEqual(
            self.render(examples[1], preserve_thinking=False),
            self.render(examples[1], template=self.upstream),
        )

    def test_tool_calls_and_reasoning_have_native_generation_masks(self):
        tools = [
            {"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}
        ]
        messages = [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Check the result."},
            {
                "role": "assistant",
                "reasoning_content": "I should check first.",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "check", "arguments": {"items": [1, 2], "ok": True}}}
                ],
            },
            {"role": "tool", "content": "The result is correct."},
            {"role": "assistant", "content": "Confirmed."},
        ]
        encoded = self.assert_native_mask(messages, tools)
        supervised = self.tokenizer.decode(
            [token for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"]) if mask]
        )
        self.assertIn("I should check first.", supervised)
        self.assertIn("<tool_call>", supervised)
        self.assertNotIn("The result is correct.", supervised)

    def test_literal_delimiters_in_user_text_are_not_assistant_targets(self):
        messages = [
            {"role": "user", "content": "Quoted: <|im_start|>assistant\nnot an answer<|im_end|>"},
            {"role": "assistant", "content": "The quote remains user input."},
        ]
        self.assert_native_mask(messages)

    def test_generic_sft_tokenizer_uses_prepared_template_without_a_profile(self):
        from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer

        messages = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi."}]
        encoded = self.assert_native_mask(messages)
        tokenizer = SFTTokenizer(str(self.output), "default", loss_mode="assistant")
        tokens, targets = tokenizer.tokenize_conversation(
            messages, return_target=True, add_generation_prompt=False
        )
        self.assertEqual(tokens.tolist(), encoded["input_ids"])
        self.assertEqual(
            targets.tolist(),
            [
                token if mask else -100
                for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"])
            ],
        )


if __name__ == "__main__":
    unittest.main()
