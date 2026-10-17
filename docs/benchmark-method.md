# DeepSeek V4.1 Flash Dual Spark benchmark method

Each engine uses the same pinned 1017 checkpoint, the same packed 1017 Engrams, TP2, a 262144-token request limit, eight request slots and DSpark depth up to three. TensorFold uses full exact prefill and no expert pruning. Thinking is tested both disabled and at reasoning effort 75. The budget is a model prompt setting; a 64-token output cap can end a thinking response before its final answer.

The TensorFold text matrix was collected before explicitly enabling `TF_DSV41_BIAS_VL=/model`. The recipe now sets that flag for native image routing. It affects image positions, not these text-only requests; image understanding is not part of the benchmark.

The published quick matrix uses 27 deterministic ledger lines: 520 input tokens with thinking disabled and 546 with reasoning effort 75. Decode runs use a 64-token output cap. C1 through C8 means that many simultaneous requests released from a barrier. Each point is one measured pass, after the engine's startup and earlier generation checks. This is a quick operational comparison; it does not estimate run-to-run variance.

Separate prefill-only runs use the same input and a one-token output limit at C1 through C8. Both thinking modes apply. Actual token usage is retained rather than assuming that thinking headers have equal lengths. Earlier longer runs informed initial diagnostics and are excluded from this quick matrix.

TTFT is measured from HTTP submission to the first streamed content or reasoning delta. The client accepts both `reasoning_content` and `reasoning` delta fields. Effective aggregate prefill throughput is total prompt tokens divided by the slowest TTFT in a prefill-only batch. This includes scheduler, network and first-token work. It is an endpoint measurement, not a claim about pure GPU kernel time. Mixed decode runs report effective prefill separately because earlier streams may decode while later prompts wait.

Per-stream observed decode throughput is (completion_tokens - 1) divided by first-to-last delta time. Speculative decoding can deliver several tokens in one delta, so this remains an observed streaming rate. Aggregate observed decode uses the first-to-last delta window across the whole batch. Aggregate wall throughput uses all output tokens divided by batch wall time, including prefill. EOS can shorten an output; output counts and finish reasons are retained.

The runner saves per-request JSONL, CSV summaries, prompt hashes, token counts, a run manifest and Prometheus snapshots before and after every measured batch. TensorFold request-log extracts are included where available. Its published Python batcher does not populate separate prefill durations in those logs; null values remain null.

Clients were paused for isolated measurements. Startup and compilation are excluded. A preliminary vLLM run overlapped its automatic post-ready warmup; those samples are excluded. The quick matrix does not run an additional per-case warmup; first-use scheduling or specialization overhead may remain. Weight reads use a read-only NFS export over the fabric; active Engram lookup uses rank-local NVMe. Prepared-cache writes and session reuse are disabled for TensorFold. No model weights, credentials or private user prompts are distributed with these results.

## Credits

The workload and measurement scripts accompany this recipe. Runtime implementation and hardware techniques come from [Jay Leaton's dual-Spark DeepSeek recipe](https://github.com/jayleaton/deepseek-v41-tensorfold-spark), [TensorFold](https://github.com/ashhart/TensorFold) and [MiaAI's vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks).
