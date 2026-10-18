# DeepSeek V4.1 Flash 1017 Dual Spark benchmark method

Each engine uses the same pinned 1017 checkpoint, the same packed 1017 Engrams, TP2, a 262144-token request limit, eight request slots and DSpark depth up to three. TensorFold uses full exact prefill and no expert pruning. Thinking is tested both disabled and at reasoning effort 75. The budget is a model prompt setting; a 64-token output cap can end a thinking response before its final answer.

The TensorFold text matrix was collected before explicitly enabling `TF_DSV41_BIAS_VL=/model`. The recipe now sets that flag for native image routing. It affects image positions, not these text-only requests; image understanding is not part of the benchmark.

The published quick matrix uses 27 deterministic ledger lines: 520 input tokens with thinking disabled and 546 with reasoning effort 75. Decode runs use a 64-token output cap. C1 through C8 means that many simultaneous requests released from a barrier. Each point is one measured pass, after the engine's startup and earlier generation checks. This is a quick operational comparison; it does not estimate run-to-run variance.

Separate prefill-only runs use the same input and a one-token output limit at C1 through C8. Both thinking modes apply. Actual token usage is retained rather than assuming that thinking headers have equal lengths. Earlier longer runs informed initial diagnostics and are excluded from this quick matrix.

TTFT is measured from HTTP submission to the first streamed content or reasoning delta. The client accepts both `reasoning_content` and `reasoning` delta fields. Effective aggregate prefill throughput is total prompt tokens divided by the slowest TTFT in a prefill-only batch. This includes scheduler, network and first-token work. It is an endpoint measurement, not a claim about pure GPU kernel time. Mixed decode runs report effective prefill separately because earlier streams may decode while later prompts wait.

Per-stream observed decode throughput is (completion_tokens - 1) divided by first-to-last delta time. Speculative decoding can deliver several tokens in one delta, so this remains an observed streaming rate. Aggregate observed decode uses the first-to-last delta window across the whole batch. Aggregate wall throughput uses all output tokens divided by batch wall time, including prefill. EOS can shorten an output; output counts and finish reasons are retained.

The runner saves per-request JSONL, CSV summaries, prompt hashes, token counts, a run manifest and Prometheus snapshots before and after every measured batch. TensorFold request-log extracts are included where available. Its published Python batcher does not populate separate prefill durations in those logs; null values remain null.

Clients were paused for isolated measurements. Startup and compilation are excluded. A preliminary vLLM run overlapped its automatic post-ready warmup; those samples are excluded. The quick matrix does not run an additional per-case warmup; first-use scheduling or specialization overhead may remain. Weight reads use a read-only NFS export over the fabric; active Engram lookup uses rank-local NVMe. Prepared-cache writes and session reuse are disabled for TensorFold. No model weights, credentials or private user prompts are distributed with these results.

## Exact tuning on 2026-10-18

The [retained comparison](../results/exact-tuning-20261018/comparison.json)
uses the same native 1017 prompt IDs and full output token IDs across two
TensorFold launches. Each profile completed 44 timed requests: C1/C8 varied
prose with 128 input tokens and 64 output tokens, plus C1 cold prefill with
1,536/3,072 input tokens and one output token, thinking off/effort75. Each
cohort has two repeats; one untimed warmup per geometry is archived separately.

| Setting | Original exact profile | Retained exact profile |
| --- | ---: | ---: |
| `TF_DSV41_EXPERT_BLOCK` | 1024 | 1536 |
| `TF_DSV41_DRAFT_GRAPH_SLOTS` | 4 | 8 |
| `TF_DSV41_GRAPH_ROWS_MAX` | 32 | 64 |
| `TF_DSV41_GRAPH_WARM` | decode | all |

The 1,536-row prompt chunk now fits one expert pass instead of 1,024+512.
The eight-slot draft graph is captured on first use; target graphs are warmed
through 64 rows. All settings use existing upstream kernels. Exact full prefill,
the checkpoint, precision, eight request slots, 262,144-token request limit and
524,288-token shared pool stay fixed. The 5 GiB admission target, 4 GiB hard
floor and graph-memory guards are unchanged. The shared expert scratch grew
from about 0.55 to 0.82 GiB; this is a speed/capacity choice, not a memory saving.
The candidate reported 8.26/9.59 GiB host MemAvailable on head/worker at ready.

The optional [host observer and client snapshot](../scripts/tuning_support/)
records exact original token-list queue emissions, server monotonic time,
prefill/finalization counters, graph-cache metadata and speculative-draft
counters. A C8 decode window starts after the latest first emission and excludes
that entire emission. It requires no subsequent prefill/finalization changes.
It is a finite-batch aggregate with finishing tails, not a constant eight-stream
steady-state rate. Cold prefill requires disabled session storage, zero cache
hits and matching executed-input/finalization counts; its rate includes request
scheduling and first-token work.

Median decode-only C8 rates were 122.1/125.4 tok/s with thinking off and
204.2/238.2 at effort75, original/retained. End-to-end C8 output rates, including
prefill and response delivery, were 78.4/80.3 and 104.6/112.5 tok/s respectively.
Cold prefill improved 2.4–6.9%; C1 decode changes were within 1.5%. These are
descriptive medians of two repeats from one boot per profile, with no confidence
interval or sustained-throughput claim. The settings were tested together;
individual gains are not isolated. vLLM was not rerun.

All 2,312 timed output tokens per profile, their full text and all prompt IDs
matched. Long-prefill checks capped output at one token; the 64-token decode
checks used short prompts. This bounded reproducibility check is not a general
quality evaluation. A separate exact EXL3 NT2/MTL2 kernel candidate passed actual
layer word/graph checks but increased MoE latency 31.8–50.6%, and was rejected.

For the observed launch, mount `scripts/tuning_support` read-only at `/tuning`
on both ranks, set `PYTHONPATH=/tuning/src:/tuning`, and replace the ordinary
`python -m tensorfold` module with
`python -m silyx.adapters.deepseek_phase_observer`, retaining all other normal
launch arguments. This wrapper changes host telemetry only. The four retained
settings work without the observer. Its source check is:

```bash
PYTHONPATH=/tuning/src:/tuning python -m silyx.adapters.deepseek_phase_observer \
  --check-sources /usr/local/lib/python3.12/dist-packages/tensorfold
```

Replay the exact archived plan against an idle observed service:

```bash
PYTHONPATH=scripts/tuning_support/src:scripts/tuning_support \
  python -m benchmarks.deepseek_serving \
  --plan results/exact-tuning-20261018/prepared/plan.json \
  --warmup --out results/raw-local/exact-recheck
```

Add `--baseline results/exact-tuning-20261018/baseline-exact1536/requests.jsonl`
to require matching complete output IDs and text. The runner stops on an
isolation, telemetry, output-cap or comparison failure. The archived
`compare.py` recomputes the summary from both retained runs and the prepared plan.

## Credits

The workload and measurement scripts accompany this recipe. Runtime implementation and hardware techniques come from [Jay Leaton's dual-Spark DeepSeek recipe](https://github.com/jayleaton/deepseek-v41-tensorfold-spark), [TensorFold](https://github.com/ashhart/TensorFold) and [MiaAI's vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks).

The retained Silyx host observer/client snapshot and combined-profile screening
apply lessons from the GLM experiments; inference and all four tuning controls
remain Jay Leaton's TensorFold DeepSeek implementation, with its existing notices.
