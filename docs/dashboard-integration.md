# Dashboard selector integration

The deployment uses a dashboard on port 11001 and a separate inference API on 8888. The dashboard exposes two profiles over one checkpoint bundle:

| Profile | Runtime | Served model ID |
| --- | --- | --- |
| `deepseek` | vLLM | `deepseek-v4.1-flash-1017-exl3` |
| `deepseek-tensorfold` | TensorFold Python | `deepseek-flash` |

Each profile has its own launcher, runtime image and container name. Both map to the same checkpoint and official 1017 Engram source. Worker readiness checks the shared NFS volumes and exact file sizes instead of requesting a second full checkpoint copy on worker NVMe.

The model manager validates source completeness, matching runtime images, serving metadata and both rank-local Engram packs before stopping the selected model. It includes both new container names in its managed set. A switch stops the managed inference model, starts the requested profile and waits for its expected model ID. The selected-profile state is updated by the manager, not by editing the dashboard's HTML.

A bundle remains unavailable while downloading, verifying or packing. Removal checks running bind mounts and Docker volume names so a shared model cannot be removed while a runtime uses it. Register both profiles against the same library entry rather than duplicating weights.

The production dashboard files and private host settings are outside this recipe repository. Adapt the profile mappings in your own manager and retain its existing lock and service lifecycle. The standalone recipe launcher intentionally manages only its own configured containers.

## Live TensorFold rates

The port-11001 dashboard patch recognizes both TensorFold metric formats:
`tensorfold_*` with a `model` label and `tensorfold:*` with a `model_name` label.
It detects `deepseek-flash` as TensorFold and reads `/health` for output tokens
from requests that are still running. The rolling decode chart therefore moves
before a request finishes. Matching health totals replace completed-request
metrics, avoiding duplicate request, token and timing counts.

Live DeepSeek prefill uses the host observer's
`deepseek_phase.prefill_tokens` counter. A chunk is counted when its execution
round starts, so the UI labels this **live scheduled prompt tokens per second**,
averaged over the same rolling window as decode. This is aggregate service
throughput, including scheduling and idle time; it is a different measurement
from the isolated decode-only tuning benchmark. Prefix-cache hits are separate.
Without live prefill telemetry, the dashboard retains its explicitly labelled
completed-request fallback or shows an unavailable rate.

Enable the optional [host observer](benchmark-method.md#exact-tuning-on-2026-10-18)
for live chunk progress. The production service uses this wrapper; it leaves
inference operations and rank communication unchanged. Ordinary TensorFold
serving works without it, with less detailed telemetry.

The patch applies to the installed dashboard version used for this recipe, which
has local integrations beyond its upstream checkout. Check compatibility before
applying it to another dashboard. [Before/after source hashes](../config/dashboard-patch.json)
identify the files; existing model-library, benchmark and accounting changes are
preserved.

From the dashboard repository root, with this recipe at `/path/to/recipe`:

```bash
git apply --check /path/to/recipe/patches/dashboard-tensorfold-metrics.patch
git apply /path/to/recipe/patches/dashboard-tensorfold-metrics.patch
python3 -m unittest discover -s dashboard -p 'test_tensorfold_metrics.py'
sudo systemctl restart dspark-live-dashboard.service
```

Restarting the dashboard does not restart inference. Refresh the browser to load
the updated rate labels. Regression checks cover native and legacy metrics,
live health preference, in-progress prefill and counter resets. A real streamed
request also verified both live rates on the retained production profile.

## Credits

The selector integrates with [anemll's dspark-vllm-gx10 dashboard](https://github.com/anemll/dspark-vllm-gx10), the [MiaAI vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) and [Jay Leaton's TensorFold Spark engine](https://github.com/jayleaton/deepseek-v41-tensorfold-spark).

The dashboard patch adapts the Anemll dashboard's metric parser and live-rate
display; its [MIT notice](../LICENSES/Anemll-MIT.txt) is retained. The host phase
observer accompanies the existing TensorFold/DeepSeek telemetry snapshot.
