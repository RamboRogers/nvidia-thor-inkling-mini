# Thor benchmark results · 2026-11-26

The data comes from a single Jetson AGX Thor in MAXN mode. The default result to cite is the **baked v6** run, not the fastest tuning run or one unusually fast prompt.

## Controlled loopback results

| Profile | C1 decode · none | C1 decode · medium | Cold-prefix input rate · ~32K target | C4 aggregate output |
|---|---:|---:|---:|---:|
| [v5 baseline · fast](results/baseline-v5-fast.json) | 56.91 tok/s | 59.40 tok/s | 2,322.2 tok/s | 100.35 tok/s |
| **[v6 baked · fast](results/v6-baked-fast.json)** | **75.44 tok/s** | **70.34 tok/s** | **3,278.8 tok/s** | **104.67 tok/s** |
| [v6 tuning · full](results/v6-tuning-full.json), different repetition counts | 76.75 tok/s | 70.40 tok/s | 3,232.2 tok/s | 95.35 tok/s |

The baked v6 cold sample processed **34,932 measured prompt tokens** in **10.654 s** to first streamed activity. Input rate is native prompt tokens divided by that wall time; it is not an isolated GPU kernel rate. The C4 result used **2,048 completion tokens** over **19.566 s** and observed 4 running requests in vLLM metrics. That column is a batch rate, **not** a per-user rate. These are profile comparisons, not one-variable ablations. The full tuning run did not improve every metric; it produced 1,920 completion tokens in its C4 batch.

## LAN qualification of the production v6 profile

[Sanitized timing record](results/v6-lan.json). A Qwen3.5 4B sidecar and Nemotron embedding service remained resident for this test. The **70.46 tok/s** LAN figure is a mean across four separate 512-completion-token samples with `reasoning_effort=medium`, not a per-task minimum. Native completion counts can include reasoning tokens.

| Task | First streamed activity | Decode output | End-to-end output |
|---|---:|---:|---:|
| Code | 0.259 s | 61.33 tok/s | 59.83 tok/s |
| Text | 0.182 s | 62.50 tok/s | 61.49 tok/s |
| Prose | 0.206 s | 62.72 tok/s | 61.76 tok/s |
| Math | 0.241 s | 95.29 tok/s | 92.07 tok/s |

Separate **behavior** checks: short smoke median first-visible latency **0.922 s**; functional code passed; short C4 tools **4/4** correct (**3.952 s** median to first visible final answer); mixed C4 **4/4** correct (**13.392 s** median for the short lanes). Both batches reached four scheduler residents. The single long request reserved **196,608 input + 65,536 output = 262,144 tokens**, recovered **3/3 needles**, used **492 completion tokens**, and took **64.68 s** to first reasoning and **76.78 s** to first visible answer. First reasoning is not a pure prefill time.

## Reproduce the speed tests

```bash
mkdir -p local-results
python3 benchmarks/bench.py --mode fast --label my-thor --out local-results/my-thor.json
```

`fast` runs code, text, prose and math in series (C1). For each task there is **one excluded warmup** plus **two** measured generations with `none`/512-token cap and **one** with `medium`/1,024-token cap. `full` uses three and two measured repetitions, respectively, and adds a ~128K-target cold-prefix test. Both use `temperature=0`, `top_p=1` and continuous native usage. The client reports per-task medians and their mean, then runs the cold-prefix and C4 phases. This is a speed probe, not a model-quality evaluation or a reproduction of the original long/mixed retrieval checks.

- **Decode** times from the first reasoning-or-content SSE activity to the last, subtracting the first chunk's token count when available (one-token fallback in the original harness). Untimable single-chunk samples use end-to-end rate. Native output includes reasoning and can include structural tokens. These are client-side estimates, not vendor GPU counters.
- **End-to-end** includes the request, prefill, streaming, and `[DONE]`. **First visible** waits for non-whitespace answer text, not reasoning.
- **C4 aggregate** is all four completion counts divided by batch wall time. Check `peak_running`, not simply the number of client threads.
- **Historical receipts:** bench JSON preserves recorded numbers, finish reasons and hashes; generated previews and private deployment metadata are omitted. The public client uses the same synthetic speed prompts and does not write reasoning or answer text. Its completeness guard rejects a stream without a finish reason, usage, or `[DONE]`.
