# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only launch contracts; DRY_RUN never invokes torchrun or loads model weights."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "04_train_qwen35_35B_1node_proxy.sh"
MODES = ("online_static_cp", "online_dynamic_cp", "offline_dynamic_cp")


class TrainingLauncherTests(unittest.TestCase):
    def launch(self, mode, **overrides):
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "PATH": os.environ["PATH"],
                "DRY_RUN": "1",
                "WORKSPACE": temporary,
                "SAVE_DIR": str(Path(temporary) / "never_created"),
                **overrides,
            }
            result = subprocess.run(
                ["bash", str(SCRIPT), mode],
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertFalse(Path(environment["SAVE_DIR"]).exists())
            return shlex.split(result.stdout.splitlines()[-1]), result.stdout

    def value(self, argv, flag):
        self.assertEqual(argv.count(flag), 1)
        return argv[argv.index(flag) + 1]

    def test_all_modes_default_to_full_model_with_fp32_cpu_optimizer(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                argv, output = self.launch(mode)
                self.assertEqual(self.value(argv, "--num-layers"), "40")
                self.assertEqual(self.value(argv, "--mtp-num-layers"), "1")
                self.assertEqual(self.value(argv, "--optimizer-offload-fraction"), "1.0")
                self.assertIn("--optimizer-cpu-offload", argv)
                self.assertIn("--use-precision-aware-optimizer", argv)
                self.assertNotIn("--fp8-param-gather", argv)
                self.assertNotIn("--reuse-grad-buf-for-mxfp8-param-ag", argv)
                for flag in (
                    "--main-params-dtype",
                    "--main-grads-dtype",
                    "--exp-avg-dtype",
                    "--exp-avg-sq-dtype",
                ):
                    self.assertEqual(self.value(argv, flag), "fp32")
                self.assertEqual(
                    self.value(argv, "--global-batch-size"),
                    "8" if mode == "offline_dynamic_cp" else "32",
                )
                self.assertEqual(self.value(argv, "--recompute-granularity"), "full")
                self.assertIn("OMP threads/rank=12", output)
                self.assertIn(
                    "/models/qwen35_35B_1node/tokenizer", self.value(argv, "--tokenizer-model")
                )

    def test_mode_precision_offload_matrix_has_no_incompatible_fp8_flags(self):
        for mode in MODES:
            for precision in ("mxfp8", "bf16"):
                for offload in ("0", "1"):
                    with self.subTest(mode=mode, precision=precision, offload=offload):
                        argv, output = self.launch(
                            mode, PRECISION=precision, OPTIMIZER_CPU_OFFLOAD=offload
                        )
                        self.assertEqual("--optimizer-cpu-offload" in argv, offload == "1")
                        self.assertEqual("--use-precision-aware-optimizer" in argv, offload == "1")
                        for flag in ("--fp8-param-gather", "--reuse-grad-buf-for-mxfp8-param-ag"):
                            self.assertEqual(flag in argv, precision == "mxfp8" and offload == "0")
                        for flag in (
                            "--fp8-format",
                            "--moe-use-grouped-tensor",
                            "--use-transformer-engine-op-fuser",
                        ):
                            self.assertEqual(flag in argv, precision == "mxfp8")
                        self.assertIn(f"OMP threads/rank={'12' if offload == '1' else '1'}", output)

    def test_explicit_layers_and_thread_count_are_preserved(self):
        argv, output = self.launch(
            "online_dynamic_cp", NUM_LAYERS="8", OPTIMIZER_CPU_OFFLOAD="0", OMP_NUM_THREADS="3"
        )
        self.assertEqual(self.value(argv, "--num-layers"), "8")
        self.assertIn("--fp8-param-gather", argv)
        self.assertIn("OMP threads/rank=3", output)

    def test_invalid_offload_toggle_fails_before_launch(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.launch("online_static_cp", OPTIMIZER_CPU_OFFLOAD="yes")

    def test_direct_master_checkpoint_loading_is_only_needed_for_fp8_parameters(self):
        for offload in ("0", "1"):
            with self.subTest(offload=offload):
                argv, _ = self.launch(
                    "online_dynamic_cp",
                    INIT_MODE="checkpoint",
                    PRETRAINED_CHECKPOINT="/unused/shape-compatible",
                    OPTIMIZER_CPU_OFFLOAD=offload,
                )
                self.assertEqual("--load-main-params-from-ckpt" in argv, offload == "0")


if __name__ == "__main__":
    unittest.main()
