# Deployment details

The examples use `/srv` paths and documentation-only fabric addresses. Create writable directories for your own account on each host and replace the host and interface values before use.

## vLLM and source sharing configuration

Start with `build/mia-runtime/.env.example`, then set the following in `build/mia-runtime/.env` and merge `config/vllm-knobs.env.example`:

```bash
HEAD_IP=192.0.2.1
WORKER_IP=192.0.2.2
WORKER_SSH=spark-b
WORKER_USER=YOUR_WORKER_USER
WORKER_HOME=/srv/deepseek-v41
HEAD_CX7_IF=YOUR_FABRIC_INTERFACE
WORKER_CX7_IF=YOUR_FABRIC_INTERFACE
HEAD_CX7_IB=YOUR_ROCE_DEVICE
WORKER_CX7_IB=YOUR_ROCE_DEVICE
HEAD_GID=3
WORKER_GID=3
NCCL_IB_GID_INDEX=3
HEAD_NCCL_IB_HCA=YOUR_ROCE_DEVICE,SECOND_ROCE_DEVICE
WORKER_NCCL_IB_HCA=YOUR_ROCE_DEVICE,SECOND_ROCE_DEVICE
MODEL_HOST=/srv/models/deepseek-v41-1017/checkpoint
ENGRAM_DIR=/srv/models/deepseek-v41-1017/engram-src-1017
ENGRAM_SRC=/srv/deepseek-v41/engram-slim-1017
HEAD_PACKED_DIR=/srv/deepseek-v41/engram-packed-1017
WORKER_PACKED_DIR=/srv/deepseek-v41/engram-packed-1017
EXPECTED_SHARDS=43
AUTO_DOWNLOAD=0
WEIGHT_SYNC=nfs
NFS_SHARE=1
NFS_SERVER_IP=192.0.2.1
NFS_CLIENTS=192.0.2.2
NFS_EXPORT_MODEL=dsv41-1017-exl3
NFS_EXPORT_ENGRAM=dsv41-1017-engram
NFS_VOLUME_MODEL=dsv41-1017-exl3-weights
NFS_VOLUME_ENGRAM=dsv41-1017-exl3-engram
HF_EXPORT_ROOT=/srv/deepseek-v41/nfs-export
IMAGE=ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce
SKIP_BUILD=1
SKIP_SHIP=1
SKIP_PULL=1
PORT=8888
TP=2
NNODES=2
MASTER_PORT=29521
CONTAINER_HEAD=deepseek-v41-flash-1017-exl3
CONTAINER_WORKER=deepseek-v41-flash-1017-exl3
SERVED_MODEL_NAME=deepseek-v4.1-flash-1017-exl3
```

Pull the pinned vLLM image on both hosts, or save and load it over SSH. Check `ibdev2netdev`, the active port state, and the RoCE v2 GID matching each fabric IP. The optional `HEAD_NCCL_IB_HCA` and `WORKER_NCCL_IB_HCA` lists enable both RoCE devices for NCCL while the physical GID check uses the primary device. With one link, omit those lists. GID 3 is the measured deployment's value; discover the appropriate value on your machines.

The patched `prepare` command performs source sharing without checking an idle GPU or starting model containers. It uses the existing NFS helper. The patch also republishes serving metadata after the checkpoint shards are already present, so a newly added chat template reaches the worker.

The Engram packer writes approximately 94.4 GiB per rank. Packing is CPU and disk work. For a machine already serving another model, put packing containers under explicit CPU and memory limits; the measured preparation used two CPUs and 512 MiB per packing container. Packing is optional only when the matching 1017 files have already been verified in the intended directories.

## Compile TensorFold extensions before model loading

On the head, after building the TensorFold image:

```bash
docker run --rm --network none --memory 8g --cpus 4 \
  -e CUDA_VISIBLE_DEVICES= -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=4 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -v deepseek41-1017-tf-cache:/cache \
  -v "$PWD/scripts/prebuild-offline.py:/offline.py:ro" \
  -v "$PWD/build/jayleaton/scripts/prebuild_ext.py:/prebuild.py:ro" \
  -v "$PWD/build/jayleaton/config/pfdense-table.json:/table.json:ro" \
  --entrypoint bash deepseek41-tensorfold:1017 \
  -c 'python /offline.py && cp /table.json /cache/pfdense-table.json'
```

The helper selects the known GB10 target `sm_121` for compilation without initializing a CUDA device. It does not replace runtime capability detection. All 18 upstream extension builders completed during deployment. The native runtime compiles any remaining Triton specializations and captures CUDA graphs during startup or first use.

For identical GB10 nodes using the same built image and package paths, copy the compiled cache:

```bash
docker run --rm --network none -v deepseek41-1017-tf-cache:/cache:ro \
  busybox:latest tar -C /cache -cf - . | \
  ssh spark-b 'docker run --rm -i --network none \
    -v deepseek41-1017-tf-cache:/cache busybox:latest tar -C /cache -xf -'
```

## Start and inspect

The public launcher reads `config/cluster.env`, then the algorithm settings file. `WORKER_MODEL` can name the Docker NFS volume created by the Mia helper. Rank-local Engram directories remain ordinary host paths. Cache and state locations are separate from checkpoint files.

`check` starts CPU-only containers to verify serving metadata and required files. `args` prints the two launch commands. `start` refuses an occupied API port or an existing compute workload. `stop` targets only the configured container name on both hosts.

Inspect startup on both nodes:

```bash
docker logs -f deepseek-v41-flash-1017-tensorfold
ssh spark-b docker logs -f deepseek-v41-flash-1017-tensorfold
```

The measured runtime announced eight slots, full exact prefill, a shared 524288-token pool and DSpark depth capped at three. Each request may use up to 262144 tokens, with admission subject to the shared pool and memory floor. TensorFold's session cache and prepared-cache writes are disabled in the published profile.

The TensorFold default is thinking on at effort 75. Requests override this with `chat_template_kwargs.enable_thinking=false`, or select effort 75 explicitly with `chat_template_kwargs.reasoning_effort=75`. Both modes are benchmarked.

The TensorFold profile sets `TF_DSV41_BIAS_VL=/model` on both ranks. The 1017 checkpoint includes these tensors, so use its own inline image-routing biases rather than an older release’s sidecar. The preflight checks all 43 tensors and the 40 main-layer attachments on CPU before GPU startup.

## Credits

These commands adapt the deployment and compilation paths from [Jay Leaton's TensorFold Spark repository](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) and [MiaAI's EXL3 dual-Spark kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks). Their source and license notices remain separate and pinned.
