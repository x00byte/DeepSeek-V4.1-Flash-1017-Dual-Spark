# Dashboard selector integration

The deployment uses a dashboard on port 11001 and a separate inference API on 8888. The dashboard exposes two profiles over one checkpoint bundle:

| Profile | Runtime | Served model ID |
| --- | --- | --- |
| `deepseek` | vLLM | `deepseek-v4.1-flash-1017-exl3` |
| `deepseek-tensorfold` | TensorFold Python | `deepseek-v4.1-flash-1017-tensorfold` |

Each profile has its own launcher, runtime image and container name. Both map to the same checkpoint and official 1017 Engram source. Worker readiness checks the shared NFS volumes and exact file sizes instead of requesting a second full checkpoint copy on worker NVMe.

The model manager validates source completeness, matching runtime images, serving metadata and both rank-local Engram packs before stopping the selected model. It includes both new container names in its managed set. A switch stops the managed inference model, starts the requested profile and waits for its expected model ID. The selected-profile state is updated by the manager, not by editing the dashboard's HTML.

A bundle remains unavailable while downloading, verifying or packing. Removal checks running bind mounts and Docker volume names so a shared model cannot be removed while a runtime uses it. Register both profiles against the same library entry rather than duplicating weights.

The production dashboard files and private host settings are outside this recipe repository. Adapt the profile mappings in your own manager and retain its existing lock and service lifecycle. The standalone recipe launcher intentionally manages only its own configured containers.

## Credits

The selector integrates with [anemll's dspark-vllm-gx10 dashboard](https://github.com/anemll/dspark-vllm-gx10), the [MiaAI vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) and [Jay Leaton's TensorFold Spark engine](https://github.com/jayleaton/deepseek-v41-tensorfold-spark).
