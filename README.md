# DeepSeek V4.1 Flash on two Sparks with TensorFold

![DeepSeek V4.1 Flash with TensorFold on two Sparks](assets/hero.png)

A pinned deployment recipe for DeepSeek V4.1 Flash 1017 EXL3 2.9bpw across two DGX Spark / GB10 systems. It provides separate TensorFold and vLLM profiles using the same checkpoint and official 1017 Engram tables, plus a reproducible C1–C8 prefill and decode benchmark with thinking disabled and thinking effort 75.

The tested TensorFold path is its Python 0.6.0 engine with Jay Leaton's DeepSeek family patches and the small 1017 adaptations in this repository. The upstream Zig engine and 2.8bpw pack are separate deployment paths. Results here identify the engine, weights, thinking mode and measurement conditions explicitly.

## Measured results

The [quick TensorFold and vLLM results](results/README.md) cover C1 through C8, thinking disabled and reasoning effort 75, plus separate prefill-only runs. Both engines completed 144 requests successfully each. The matching vLLM matrix is included alongside TensorFold.

At C4, observed aggregate decode was **42.4 tok/s with thinking off** and **66.5 tok/s at effort 75**. Effective aggregate prefill was **431.7** and **422.1 tok/s**, respectively. These are short, single-pass endpoint measurements; read the method before comparing them with other workloads.

The matching vLLM C4 run measured **47.9 tok/s decode off** and **44.9 tok/s at effort 75**, with effective aggregate prefill of **780.9** and **708.8 tok/s**. These one-pass samples do not establish a general engine ranking.

See [the measurement method](docs/benchmark-method.md). Prefill-only tests use a one-token output cap; decode tests use 64 output tokens, with one pass at each point. Actual token usage, early stops, TTFT, observed decode and aggregate wall throughput are retained.

## Hardware and storage

- Two GB10 systems with 128 GB unified memory each, Linux arm64, NVIDIA Container Toolkit and Docker.
- Passwordless SSH from head to worker, with host keys already trusted.
- A direct fabric link with working RoCE. The measured configuration enabled both available fabric links.
- Approximately 196 GiB for EXL3 weights, 189 GiB for the official Engram source tables and 94.4 GiB for rank 0's packed Engrams on the head, plus container images and caches.
- Approximately 94.4 GiB of packed Engrams on the worker, plus images and compiled caches. The worker reads the checkpoint over a read-only NFS export on the fabric. TensorFold prepared-cache writes are disabled for this storage layout.

Keep Engram lookup local to each rank. The checkpoint and Engram source revisions are pinned in [config/manifest.json](config/manifest.json). The two large Engram tables are from **DeepSeek-V4.1-Flash-1017**, matched to this checkpoint.

## Prepare the sources and image

```bash
bash scripts/fetch-sources.sh
docker build -t deepseek41-tensorfold:1017 .
docker save deepseek41-tensorfold:1017 | ssh spark-b docker load
```

The Dockerfile uses NVIDIA PyTorch 26.07, preserves its torch and Triton versions, and installs the tested TensorFold source export. The source script applies upstream patches before the local 1017 patch. It leaves the original source checkouts available for review.

The local adaptations cover 1017's inline image-router biases, 6-bit dense MTP layers in the vLLM overlay, tool-only turn handling, the actual TensorFold shared-pool memory budget, revision-specific packed paths and repeatable metadata publication to the NFS share.

## Download and verify the assets

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install huggingface_hub==2.2.0 hf-xet==1.7.0
python scripts/download.py --root /srv/models/deepseek-v41-1017
```

The downloader verifies Hub LFS SHA-256 checksums, records a manifest and supplies the reference-tested chat template. The checkpoint does not supply a Hugging Face chat-template file; the TensorFold HTTP app needs one during initialization, even though its native encoder renders actual DeepSeek prompts.

## Publish weights and pack Engrams

Configure `build/mia-runtime/.env` using the [deployment example](docs/deployment.md) and the tuning values in [config/vllm-knobs.env.example](config/vllm-knobs.env.example). Set the new bundle paths and `EXPECTED_SHARDS=43`, then run:

```bash
bash build/mia-runtime/start.sh prepare
bash build/mia-runtime/start.sh pack
```

`prepare` publishes the source files and writes launch scripts without starting GPU servers. `pack` creates `engram-l1-r0of2.bin` and `engram-l14-r0of2.bin` on the head, and the rank 1 equivalents on the worker. Use unique directories for this model revision. Each file uses the same hash-head partitioning expected by both runtimes.

Check the packs with the upstream TensorFold checker; it compares headers, sizes and sampled records with the verified source:

```bash
python build/jayleaton/scripts/pack_engram.py \
  --src /srv/models/deepseek-v41-1017/engram-src-1017 \
  --config /srv/models/deepseek-v41-1017/checkpoint/config.json \
  --check /srv/deepseek-v41/engram-packed-1017 --rank 0
```

Run the rank 1 check on the worker with its source NFS volume mounted read-only. Both packs were checked with both runtimes' readers during deployment.

## Compile and start TensorFold

Copy [config/cluster.env.example](config/cluster.env.example) to `config/cluster.env` and set the real host, fabric and storage values. The worker model setting accepts a Docker NFS volume name. Algorithm settings are in [config/tensorfold-knobs.env.example](config/tensorfold-knobs.env.example).

[The deployment guide](docs/deployment.md) includes CPU-only extension precompilation and copying the compiled cache to the second identical GB10 system. Precompile before loading the model so compilation does not compete with the resident weights for memory.

```bash
python scripts/cluster.py check
python scripts/cluster.py args
python scripts/cluster.py start
python scripts/cluster.py health
```

The launcher uses the same container name on both nodes, starts rank 1 before rank 0 and affects only those containers. It refuses startup while another GPU workload is running. Stop the existing inference service explicitly before switching engines. Initial weight loading and graph warmup take several minutes.

To stop this profile:

```bash
python scripts/cluster.py stop
```

For vLLM, the patched Mia kit provides `bash build/mia-runtime/start.sh start`. Apply the pinned image digest and tuning file first. [Dashboard integration](docs/dashboard-integration.md) describes the two-selector layout used on port 11001.

## Benchmark

Run one engine at a time with other API clients paused:

```bash
python scripts/benchmark.py \
  --model deepseek-v4.1-flash-1017-tensorfold --label tensorfold \
  --thinking off --input-lines 27 --output-tokens 64 --repeats 1 --skip-warmup --out results/raw-local/tensorfold-off

python scripts/benchmark.py \
  --model deepseek-v4.1-flash-1017-tensorfold --label tensorfold \
  --thinking 75 --input-lines 27 --output-tokens 64 --repeats 1 --skip-warmup --out results/raw-local/tensorfold-thinking75
```

Use the vLLM model ID `deepseek-v4.1-flash-1017-exl3` with the same arguments after switching. For prefill-only tests, add `--output-tokens 1`. The complete matrix, token lengths, counting conventions and limitations are documented in [the method](docs/benchmark-method.md).

## Scope and attribution

This repository distributes the integration recipe, source patches and benchmark artifacts. Model weights and private machine configuration are downloaded or supplied separately. Upstream components retain their existing licenses, included in `LICENSES/`; no blanket license is added for original integration material.

## Credits

- [Jay Leaton's DeepSeek V4.1 TensorFold Spark recipe](https://github.com/jayleaton/deepseek-v41-tensorfold-spark): the DeepSeek TensorFold engine patches, packing format, tuning and deployment work.
- [TensorFold](https://github.com/ashhart/TensorFold): the Python inference runtime, CUDA kernels and TensorFold brand mark.
- [MiaAI's DeepSeek V4.1 EXL3 dual-Spark kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks): the vLLM runtime, file-backed Engram implementation, packing tools, chat template and sharing helpers.
- [LibertAIDAI's EXL3 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) and [DeepSeek's official 1017 model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): model weights, tokenizer, reference encoder and matching Engram tables.
- [npaka's TensorFold overview](https://note.com/npaka/n/n3d3e09549bdd): the supplied artwork reference. The TensorFold logo reference came from the upstream asset; TensorFold and DeepSeek marks belong to their respective owners. [Artwork prompt](assets/artwork-prompt.txt).
