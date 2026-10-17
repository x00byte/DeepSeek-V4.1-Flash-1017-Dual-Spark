# Quick TensorFold results

One pass per C1–C8 point, thinking disabled and effort 75. Decode uses a 64-token cap; separate prefill runs use a one-token cap. Inputs are 520 tokens with thinking disabled and 546 with effort 75. These short runs do not estimate variance or steady-state long-response performance.

[Combined CSV](tensorfold-quick.csv), [per-request records and run manifests](tensorfold/), [method](../docs/benchmark-method.md). The vLLM comparison is pending.

| Concurrency | Decode off, aggregate tok/s | Decode 75, aggregate tok/s | Prefill off, effective aggregate tok/s | Prefill 75, effective aggregate tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 58.1 | 77.5 | 423.1 | 415.0 |
| 2 | 43.4 | 53.5 | 451.2 | 413.3 |
| 3 | 44.6 | 51.5 | 442.1 | 435.7 |
| 4 | 42.4 | 66.5 | 431.7 | 422.1 |
| 5 | 41.1 | 34.6 | 440.5 | 436.4 |
| 6 | 48.0 | 42.6 | 444.2 | 433.8 |
| 7 | 38.9 | 42.3 | 438.1 | 444.0 |
| 8 | 41.3 | 40.4 | 444.7 | 440.9 |

Decode aggregate uses the first-to-last streamed delta window across the batch. With staggered prefill, some requests decode while others wait; it is an observed endpoint rate. Prefill includes admission, network and first-token work. CSV files also retain per-stream rates, TTFT and end-to-end throughput.

## Credits

Measured with the pinned [TensorFold Spark engine](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) and [LibertAIDAI 1017 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), using [DeepSeek 1017 Engrams](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
