# DeepSeek V4.1 Flash 1017 Dual Spark benchmarks

**TensorFold is faster for single-request decode. vLLM is faster for aggregate throughput at C8 and for effective prefill in these quick runs.**

TensorFold and vLLM use the same pinned `DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` weights and packed 1017 Engrams on TP2 dual Spark. Each engine completed 144 requests successfully: C1–C8, thinking disabled and effort 75, with separate decode and prefill runs. Inputs are 520 tokens with thinking disabled and 546 with effort 75. Prompt hashes, input counts and output counts match across engines.

Decode uses a 64-token cap; prefill uses one token. Each point is one pass. These short runs do not estimate variance or steady-state long-response performance. Startup and automatic warmup are excluded; the preliminary run that overlapped vLLM startup warmup is excluded.

[TensorFold CSV](tensorfold-quick.csv), [vLLM CSV](vllm-quick.csv), [TensorFold request records](tensorfold/), [vLLM request records](vllm/), [method](../docs/benchmark-method.md).

## Thinking disabled

All throughput columns below are aggregate tok/s. Prefill is effective endpoint throughput, including scheduling and first-token work.

| Concurrency | TensorFold decode | vLLM decode | TensorFold prefill | vLLM prefill |
| --- | ---: | ---: | ---: | ---: |
| 1 | 58.1 | 36.9 | 423.1 | 615.9 |
| 2 | 43.4 | 51.4 | 451.2 | 621.9 |
| 3 | 44.6 | 59.2 | 442.1 | 759.5 |
| 4 | 42.4 | 47.9 | 431.7 | 780.9 |
| 5 | 41.1 | 42.5 | 440.5 | 784.6 |
| 6 | 48.0 | 57.8 | 444.2 | 850.8 |
| 7 | 38.9 | 55.7 | 438.1 | 805.6 |
| 8 | 41.3 | 60.1 | 444.7 | 842.0 |

## Reasoning effort 75

All throughput columns below are aggregate tok/s. Prefill is effective endpoint throughput, including scheduling and first-token work.

| Concurrency | TensorFold decode | vLLM decode | TensorFold prefill | vLLM prefill |
| --- | ---: | ---: | ---: | ---: |
| 1 | 77.5 | 41.4 | 415.0 | 603.4 |
| 2 | 53.5 | 56.4 | 413.3 | 607.8 |
| 3 | 51.5 | 54.7 | 435.7 | 608.4 |
| 4 | 66.5 | 44.9 | 422.1 | 708.8 |
| 5 | 34.6 | 58.7 | 436.4 | 779.6 |
| 6 | 42.6 | 58.4 | 433.8 | 840.5 |
| 7 | 42.3 | 62.5 | 444.0 | 797.4 |
| 8 | 40.4 | 76.4 | 440.9 | 839.7 |

## End-to-end output throughput

This rate includes prefill and the complete batch wall time; it is often more useful for estimating short-request capacity than the decode window alone. Values are aggregate output tok/s.

| Concurrency | TensorFold off | vLLM off | TensorFold 75 | vLLM 75 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 26.7 | 24.9 | 29.9 | 26.4 |
| 4 | 35.6 | 41.8 | 40.9 | 39.4 |
| 8 | 35.2 | 51.0 | 36.8 | 61.8 |

Decode aggregate uses the first-to-last streamed delta window across the batch. With staggered prefill, some requests decode while others wait; it is an observed endpoint rate. CSV files retain per-stream rates and TTFT. C8 aggregate is not the speed of each of eight streams.

The TensorFold measurements use the published Python 0.6.0 path, exact full prefill and no expert pruning. They do not measure the newer Zig path. The two engines use their documented scheduler and cache settings, so these are deployment measurements rather than isolated kernel comparisons.

## Credits

Measured with [Jay Leaton’s TensorFold Spark engine](https://github.com/jayleaton/deepseek-v41-tensorfold-spark), [TensorFold](https://github.com/ashhart/TensorFold), [MiaAI’s vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks), the [LibertAIDAI 1017 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), and [DeepSeek 1017 Engrams](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
