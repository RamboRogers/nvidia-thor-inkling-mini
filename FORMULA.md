# Formula and provenance

## Exact inputs

- Checkpoint: `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`. The [fetcher](fetch_model.py) verifies the original tokenizer/template, model config, quantization metadata, target weights and bundled DSpark weights file-by-file. No remote model code is enabled.[1]
- Public base: `vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648`; `linux/arm64`, Docker image/config ID `sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de`. This is the CUDA 13.0 base with PyTorch `2.14.0+cu130` and vLLM `0.1.dev25855+g7cb5dae22`.[3]
- CUDA donor: `cqingwang/Inkling-2-Mini@54e2d9456738591ecd057a1a29586b0a69bfbac6`. The selected donor file bytes match the corresponding inputs from the benchmarked build. The upstream DGX Spark launcher is **not** used on Thor.[2]
- Original measured `v6-thor-opt` image ID: `sha256:11add92a9d85905a3c990ef6ce99c1a9656a367e715d324ee5675fde727bec75`. This is historical provenance, not a public pull reference. The all-in-one patch reproduces the 16 modified or added runtime Python files byte-for-byte; full input/output SHA-256s are in [the manifest](patches/manifest.json). The source build consolidates layers, rebuilds the CUDA extensions, and adds public metadata, so its image ID differs.

## What changed

| Optimization | What to keep when porting |
|---|---|
| Dynamic verification cap in the DSpark CUDA graph | `cost:1=0.30,2=0.22,3=0.12,4=0.0,min=1,max=8`. The padding hook skips unused Humming MoE rows; unused draft positions are masked before rejection sampling. This is not setting the DSpark proposal count below seven. |
| INT8 draft heads/layers | Use the 65,536-entry hot vocabulary and narrow W8 tiles. The drafter scatters into the fixed full-vocabulary buffer for probabilistic verification; the target checkpoint and full target vocabulary are unchanged. |
| Packed BF16x on the LM head | Compile for `sm_110a`. Startup checks that unpacking restores the original BF16 weight bits and rejects a mismatch. Leave dense QKVR and WO projections on the stock path; enabling BF16x there was not the winning configuration. |
| Attention | Use `tml_fa4` for **global** prefill layers, the SM110 decode split cap, BF16 KV, and the qualified warmup configuration. See [the launch arguments](serve.py) for the exact switches. |
| OpenAI interface | Use the `inkling` reasoning and tool parsers with `--enable-auto-tool-choice`. A named-tool finish correction and a model-specific explicit `none` shim are included in the patch. |

For deterministic short jobs, `reasoning_effort="none"` was tested at `temperature=0`. For sampled chat or tools, prefer `low` or `medium`. Combining `none` with a high temperature can sample unwanted opening tokens in this implementation. The project qualifies text and tool use, not vision or a different GPU architecture. A stock DGX Spark image requiring CUDA 13.2 was rejected by the tested R580 host before execution; don't disable the NVIDIA driver check.[4]

The 28 GiB cache is shared, not one allocation per request. At the context boundary, `prompt_tokens + reserved_output_tokens` must fit the 262,144-token limit. A 65,536-token allowance is not a claim that a test generated that many tokens. Start a clean benchmark on an idle server; changing cache precision, power mode, build/runtime version, or competing work can change both speed and available context.

## Attribution and licenses

- The Thor wrapper and patch use Apache-2.0. The patch modifies upstream vLLM and includes compatibility work from trufae / Mjolnir and sxuff; see [the original INT8 notice](patches/NOTICE).[3][4][5]
- The builder preserves the MIT donor licenses and credits cqingwang and Andy Shu for Pico.[2]
- The Thinking Machines Lab checkpoint remains a separate Apache-2.0 upstream download; no weights or upstream container layers are redistributed here.[1]

## What was verified

The public source recipe built on Jetson AGX Thor from the pinned base and donor, with both freshly compiled SM110a extensions. A GPU canary loaded both from the rebuilt image, round-tripped a 1,024 × 4,096 BF16 weight matrix bit-exactly, and matched a reference projection with each kernel. The smoke client also passed against the unchanged running benchmark profile. The rebuilt image was not cold-loaded with the full checkpoint for this packaging test; the published speed and capacity figures are measurements of the original image, not the rebuild.

## Sources

[1] https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195 — Thinking Machines Lab: pinned Inkling 2 Mini checkpoint
[2] https://github.com/cqingwang/Inkling-2-Mini/tree/54e2d9456738591ecd057a1a29586b0a69bfbac6 — cqingwang: pinned DGX Spark CUDA overlay sources
[3] https://github.com/vllm-project/vllm — vLLM upstream
[4] https://github.com/GoBrrr-ai/mjolnir/releases/tag/release-2026.11.25 — Mjolnir upstream release
[5] https://github.com/sxuff/inkling-2-mini-spark — sxuff: original INT8 DSpark/Marlin-style draft projection adaptation
