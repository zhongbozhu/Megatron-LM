# Data Pipeline

## FIM dataset

`GPTFIMDataset` extends Megatron-Core’s `GPTDataset` to support **Fill-in-the-Middle (FIM)** data augmentation.
It probabilistically converts samples into FIM format using configurable rates, with support for both PSM and SPM patterns, fragment-level splitting, and length-preserving output.

`GPTFIMDatasetConfig` provides the configuration needed to enable this behavior.
`GPTFIMDatasetConfig` configuration object extending `GPTDatasetConfig` to enable FIM preprocessing.

**Attributes**

- `rate`: Probability of converting a sample into a FIM example. A value of `1.0` means FIM is always applied. a value of `0.0` means FIM is never applied.
- `spm_rate`: Probability of using the SPM FIM pattern (vs PSM). The remaining probability (`1 - spm_rate`) selects the PSM (prefix-suffix-middle) pattern instead. For example, if `spm_rate = 0.3`: 30% SPM, 70% PSM.
- `extra_tokens`: Dictionary containing the FIM special tokens: {"prefix", "middle", "suffix", "pad", "eod"}.
- `split_sample`: Optional token around which samples are split before applying FIM. If provided, the input sequence is divided at every occurrence of this token, and FIM is applied independently to each fragment. `A B C <SPLI_SAMPLE> D E F <SPLIT_SAMPLE> G H` -> `FIM(Fragment 1) <SPLI_SAMPLE> FIM(Fragment 2) <SPLI_SAMPLE> FIM(Fragment 3)`.
- `fragment_rate`: Probability of applying FIM to each fragment when split_sample is used.
- `no_prefix`: If the decoded sequence starts with this prefix, FIM is skipped.
`GPTFIMDataset` dataset class that loads token sequences from an `IndexedDataset` and applies FIM transformations before returning each sample.

**PSM Format**
```
[prefix_tok] prefix [suffix_tok] suffix [middle_tok] middle
```

**SPM Format**
```
[prefix_tok, suffix_tok] suffix [middle_tok] prefix middle
```

**Special cases:**

- If the sequence starts with no_prefix, FIM is skipped.
- If FIM is not applied, the sample is returned unchanged.
## Varlen dataset

`VarlenDataset` packs SFT-style instruction data of widely varying lengths into
THD (variable-length) format. It extends the `SFTDataset` family, so it reuses
the same packing / `cu_seqlens` / context-parallel padding logic, and is
selected independently of `--sft`.

This section documents the **input schemas** it accepts. Loading and packing are
described with the dataset classes themselves.

### Schema auto-detection

The input layout is validated for each record, most explicit first. The first
match wins; optional fields such as `tools` may be absent from the first record:

| Schema | Detected by | Normalized to |
|---|---|---|
| `openai-messages` | a `messages` column | passed through |
| `bridge-conversation` | a `conversation` column | renamed to messages |
| `sharegpt` | a `conversations` column | messages list |
| `alpaca` / `dolly` | an instruction column **and** an output column | 3-turn messages list |
| `pretrain-text` | a `text` column | raw string (no chat template) |

Column names accepted for the alpaca/dolly layout:

- instruction: `instruction`, `prompt`, `query`, `question`
- output: `output`, `response`, `completion`, `answer`
- optional extra user-turn context: `input` (Stanford Alpaca), `context` (Dolly)

If no layout matches, construction or record access raises a `ValueError`
listing the columns and supported schemas.

### Legacy normalization rules

Without an explicit `--sft-loss-mode`, the instruction-tuning layouts are converted to the messages list the
parent `SFTDataset` expects:

- **A leading `system` turn is guaranteed.** An empty one is prepended when the
  sample does not already start with a system turn, so
  `SFTDataset._split_conversations` treats each sample as one conversation.
- **ShareGPT speakers are mapped to roles** via the `from` field:
  `human`/`user` → `user`; `gpt`/`assistant`/`model`/`chatgpt`/`bing`/`bard` →
  `assistant`; `system` → `system`; `tool`/`function`/`observation` → `tool`.
  Unrecognized speakers fall back to `user` rather than failing.
- **Alpaca/Dolly context is folded into the user turn**, joined to the
  instruction by a blank line when present.
- **Non-`role`/`content` keys are dropped** from ordinary legacy messages.
  Supplying tool calls, tool IDs, row-level tools or template controls to this
  unsupported legacy path raises an error instead of silently discarding them.

`pretrain-text` is the exception: it returns the `text` column unchanged as a
plain string, and the dataset dispatches on that to skip chat templating and
prompt masking. This supports long-context pretraining corpora (Dolma, OLMo
midtraining) packed through the same THD path as SFT.

### Limitations

- Turn content must be a plain string. Multi-modal samples that carry content as
  a list of image/text parts raise a `ValueError`.
- Non-string values in instruction/output fields raise a `ValueError` rather
  than being coerced.


### Explicit tool-aware chat SFT

Local JSONL is the guaranteed agentic input format. A record contains `messages`
and optional `tools` and `chat_template_kwargs`. Native arrays and JSON-serialized
arrays are accepted for messages/tools. Tool-call argument strings must encode
JSON objects; normalization retains nested values, names, IDs, reasoning fields,
message ordering, and other message metadata without mutating the source record.
Top-level metadata is not passed to the template as arbitrary keyword arguments.

Use `--use-varlen-dataset`, `--tokenizer-type SFTTokenizer`,
`--sft-tokenizer-prompt-format default`, and an explicit `--sft-loss-mode`.
The new modes support the HF backend; gigatoken and legacy custom prompt formats
are rejected. One whole trajectory remains one logical causal attention sequence,
including repeated user/assistant/tool turns. No system message is synthesized,
and unsupported roles are errors. Existing HF Hub and Parquet sources remain
available, but arbitrary heterogeneous nested Arrow schemas are not guaranteed;
materialize canonical JSONL first when necessary.

- `assistant` supervises positions marked by the tokenizer's HF `{% generation %}`
  blocks. The template defines whether reasoning, tool calls and ending tokens
  belong to those regions. Missing, malformed or empty masks are errors; provide
  a training template with generation annotations. There is no delimiter-guessing
  fallback or model-specific profile in the encoder.
- `full` supervises every retained next-token target, including role/control
  tokens. Physical padding is excluded from loss.
- Omitting `--sft-loss-mode` retains legacy tokenizer defaults and raw-text
  pretraining behavior.

Load a prepared training template through `--tokenizer-model`. For Qwen3.5,
keep assistant headers outside generation blocks and reasoning, tool calls,
body, existing `<|im_end|>` and its newline inside. Preserve historical reasoning
when preparing the template; training never rewrites templates. Template-specific controls may
be supplied via each record's `chat_template_kwargs`. They cannot override tools,
template selection, tokenization, truncation, padding, masks, generation prompts
or return formats. Switching templates or loss policy requires repacking offline
files, because their supervision is already materialized.

The encoder returns unshifted IDs/targets from one complete rendering. Sample
assembly right-truncates both to `sequence_length + 1`, never appends/substitutes
EOS, then shifts exactly once. Samples with fewer than two tokens or no remaining
assistant targets fail with an input location. The six scheduler-facing tensors
retain `original_seq_len = N` real next-token positions and `padded_seq_len = P`
physical positions; padding uses the existing divisor and has zero loss. Real
length is not inferred from a zero loss mask, since unsupervised context is real.
`--varlen-sbhd-validation` uses the same target policy with fixed physical padding.

`JsonlRows` scans immutable files once to index nonblank byte offsets and physical
line numbers. It opens a file per worker/process, omits handles from pickle state,
and reports parse/schema errors as path plus physical line without dumping the
trajectory. UTF-8 and a final line without a newline are supported. It is an
indexed map-style reader, not streaming: offsets consume memory, startup needs a
scan, and tokens are not cached. Offline packed Parquet input uses a separate
packed-row counting convention: one dataset item is one stored row, which may
contain multiple trajectories.
