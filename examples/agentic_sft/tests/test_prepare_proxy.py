# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only tests for tokenizer/config preparation; no model weights needed."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "01_prepare_qwen35_35B_1node_proxy.py"
SPEC = importlib.util.spec_from_file_location("prepare_proxy", SCRIPT)
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)


class PrepareProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.config = {
            "model_type": "qwen3_5_moe",
            "transformers_version": "4.57.0.dev0",
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 40,
                "hidden_size": 2048,
                "num_experts": 256,
                "moe_intermediate_size": 512,
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
            },
        }
        (self.source / "config.json").write_text(json.dumps(self.config))
        (self.source / "tokenizer_config.json").write_text('{"chat_template": "tool 世界"}')
        (self.source / "tokenizer.json").write_text('{"version": "1.0"}')
        (self.source / "chat_templates").mkdir()
        (self.source / "chat_templates/default.jinja").write_text("{{ tools }} {{ messages }}")

    def test_only_assets_with_full_config_and_explicit_training_template(self):
        (self.source / "model.safetensors").write_bytes(b"not a model: must never be read")
        output = self.root / "tokenizer"
        PROXY.prepare_tokenizer(self.source, output)
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 40)
        self.assertEqual(len(config["layer_types"]), 40)
        self.assertEqual(config["mtp_num_hidden_layers"], 1)
        self.assertEqual(config["hidden_size"], 2048)
        self.assertEqual(config["transformers_version"], "4.57.0.dev0")
        self.assertFalse(config["tie_word_embeddings"])
        self.assertFalse(list(output.glob("*.safetensors*")))
        self.assertEqual(
            (output / "tokenizer.json").read_bytes(), (self.source / "tokenizer.json").read_bytes()
        )
        template = PROXY.TRAINING_TEMPLATE.read_text()
        self.assertEqual((output / "chat_template.jinja").read_text(), template)
        self.assertEqual(
            json.loads((output / "tokenizer_config.json").read_text())["chat_template"], template
        )
        self.assertFalse((output / "chat_templates").exists())
        manifest = json.loads((output / "proxy_manifest.json").read_text())
        self.assertEqual(manifest["format_version"], 2)
        self.assertEqual(manifest["retained_decoder_layers"], list(range(40)))
        self.assertEqual(
            manifest["effective_chat_template_sha256"], PROXY.sha256(PROXY.TRAINING_TEMPLATE)
        )
        self.assertIn(
            "tokenizer_config.json:chat_template", manifest["original_chat_templates_sha256"]
        )
        self.assertEqual(
            (self.source / "tokenizer_config.json").read_text(), '{"chat_template": "tool 世界"}'
        )
        PROXY.prepare_tokenizer(self.source, output)
        (output / "tokenizer.json").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            PROXY.prepare_tokenizer(self.source, output)

    def test_different_layer_count_cannot_reuse_assets(self):
        output = self.root / "tokenizer"
        PROXY.prepare_tokenizer(self.source, output)
        with self.assertRaisesRegex(ValueError, "different provenance/config"):
            PROXY.prepare_tokenizer(self.source, output, num_layers=4)

    def test_explicit_eight_layer_config_remains_available(self):
        output = self.root / "tokenizer_l8"
        PROXY.prepare_tokenizer(self.source, output, num_layers=8)
        config = json.loads((output / "config.json").read_text())
        manifest = json.loads((output / "proxy_manifest.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 8)
        self.assertEqual(len(config["layer_types"]), 8)
        self.assertEqual(manifest["retained_decoder_layers"], list(range(8)))
        with self.assertRaisesRegex(ValueError, "different provenance/config"):
            PROXY.prepare_tokenizer(self.source, output)

    def test_bad_layer_count_and_missing_assets_do_not_publish(self):
        for count in (0, 6, 41):
            with self.subTest(count=count), self.assertRaises(ValueError):
                PROXY.proxy_config(self.config, count)
        (self.source / "tokenizer_config.json").unlink()
        output = self.root / "tokenizer"
        with self.assertRaisesRegex(ValueError, "tokenizer_config.json"):
            PROXY.prepare_tokenizer(self.source, output)
        self.assertFalse(output.exists())

    def test_same_config_with_changed_vocabulary_is_not_reused(self):
        output = self.root / "tokenizer"
        PROXY.prepare_tokenizer(self.source, output)
        (self.source / "tokenizer.json").write_text('{"version": "changed"}')
        with self.assertRaisesRegex(ValueError, "differs from the selected source"):
            PROXY.prepare_tokenizer(self.source, output)

    def test_stale_external_template_is_not_reused(self):
        output = self.root / "tokenizer"
        PROXY.prepare_tokenizer(self.source, output)
        (output / "chat_templates").mkdir()
        (output / "chat_templates/tool_use.jinja").write_text("stale source template")
        with self.assertRaisesRegex(ValueError, "unexpected template assets"):
            PROXY.prepare_tokenizer(self.source, output)

    def test_legacy_artifact_is_not_mutated(self):
        output = self.root / "tokenizer"
        PROXY.prepare_tokenizer(self.source, output)
        manifest = json.loads((output / "proxy_manifest.json").read_text())
        manifest["format_version"] = 1
        (output / "proxy_manifest.json").write_text(json.dumps(manifest))
        before = {
            str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        with self.assertRaisesRegex(ValueError, "new --output-dir"):
            PROXY.prepare_tokenizer(self.source, output)
        self.assertEqual(
            before,
            {
                str(path.relative_to(output)): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            },
        )


if __name__ == "__main__":
    unittest.main()
