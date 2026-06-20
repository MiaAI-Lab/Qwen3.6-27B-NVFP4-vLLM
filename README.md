# Qwen3.6-27B (NVFP4) — Self-Hosted Inference via vLLM

[![vLLM](https://img.shields.io/badge/vLLM-nightly-blue)](https://github.com/vllm-project/vllm)
[![Model](https://img.shields.io/badge/model-Qwen3.6--27B-informational)](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)
[![ARM64](https://img.shields.io/badge/arch-arm64-lightgrey)](#)

A production-ready vLLM deployment wrapper for **[Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)** — an FP4-quantized version of Qwen3.6.

This repo bundles a ready-to-run Docker container, a custom chat template, and start/stop scripts so you can spin up a fully OpenAI-compatible inference server in minutes.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Model** | Qwen3.6-27B-NVFP4 — NVFP4 quantised dense (~27B params) |
| **Inference Engine** | vLLM (nightly) with FlashInfer attention + Marlin MoE backend |
| **Speculative Decoding** | MTP (Multi-Token Prediction), 3 speculative tokens |
| **Context Window** | Up to **262 144 tokens** (256K) |
| **OpenAI-Compatible API** | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| **Vision Support** | Multi-modal image input (up to 4 images per request) |
| **Tool Use** | Qwen3-coder tool-call parser, auto tool choice enabled |
| **Thinking/Reasoning** | CoT / chain-of-thought with `<thinking>` block support (configurable) |
| **Reasoning Parser** | Qwen3-specific parser via `--reasoning-parser qwen3` |
| **Streaming** | Full SSE streaming support |
| **Prefix Caching** | Enabled via `--enable-prefix-caching` |
| **Chunked Prefill** | Enabled via `--enable-chunked-prefill` |
| **Async Scheduling** | Enabled via `--async-scheduling` |
| **Custom Chat Template** | Full Jinja template with vision, tool use, and thinking support |
| **ARM64 Ready** | Self-contained GCC, Python dev deps, and triton cache |

---

## 📋 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    Your Host Machine                  │
│                                                      │
│  start.sh / stop.sh                                  │
│  chat_template.jinja  ← Custom Jinja template        │
│  .bin/                ← Bundled GCC                  │
│  .deps/               ← Bundled .deb deps            │
│  .cache/              ← HuggingFace + Triton cache   │
│  .vllm.log            ← Container log                │
│  .vllm.pid            ← Container ID                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  Docker Container: vllm/vllm-openai:nightly  │    │
│  │                                              │    │
│  │  Qwen3.6-27B-NVFP4                            │    │
│  │  vLLM Server ← OpenAI API on :8888           │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

The container exposes an **OpenAI-compatible REST API** at `http://0.0.0.0:8888/v1`, so any client that speaks the OpenAI protocol (langchain, llama-cpp-python, oapi, custom HTTP) can connect directly.

---

## 🛠️ Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| **OS** | Ubuntu 22.04+ / Debian 12+ | ARM64 / aarch64 recommended |
| **GPU** | NVIDIA GPU with ≥ 40 GB VRAM | Tested on 48 GB+ (e.g. RTX 6000, A100) |
| **CUDA** | CUDA 12.x compatible | NVIDIA driver ≥ 535 |
| **Docker** | 24.0+ | With [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **curl** | Any | Used for readiness probes |
| **Disk** | ~50 GB free | Model weights + caches |

---

## 🚀 Quick Start

### 1. Clone & Navigate

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-vLLM
cd Qwen3.6-27B-NVFP4-vLLM
```

### 2. (Optional) Set HuggingFace Token

If the model repo (`unsloth/Qwen3.6-27B-NVFP4`) requires authenticated access:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Start the Server

```bash
./start.sh
```

This will:
1. Check for Docker and curl on PATH
2. Create cache directories (HuggingFace + Triton)
3. Remove any stale container with the same name
4. Pull the latest `vllm/vllm-openai:nightly` image
5. Launch the container with `--gpus all`
6. Stream logs to `.vllm.log`
7. Poll `/v1/models` until the server is ready
8. Print the OpenAI base URL on success

**Expected output:**
```
Starting vLLM container for unsloth/Qwen3.6-27B-NVFP4
Image: vllm/vllm-openai:nightly
Listening on 0.0.0.0:8888
Writing progress to .vllm.log
...
vLLM is ready and responding; shell is now free.
OpenAI base URL: http://0.0.0.0:8888/v1
```

### 4. Test It

```bash
# Quick health check
curl http://0.0.0.0:8888/v1/models | jq

# Chat completion
curl -s http://0.0.0.0:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-27B-NVFP4",
    "messages": [{"role": "user", "content": "Explain quantization in one sentence."}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 256,
    "stream": false
  }' | jq
```

### 5. Stop the Server

```bash
./stop.sh
```

---

## ⚙️ Configuration

All configurable options live in [`start.sh`](start.sh). Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3.6-27B-NVFP4` | HuggingFace model identifier |
| `IMAGE` | `vllm/vllm-openai:nightly` | vLLM Docker image tag |
| `CONTAINER_NAME` | `qwen3.6-27b-nvfp4-vllm` | Docker container name |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8888` | HTTP port |
| `HF_TOKEN` | env var | HuggingFace auth token |

### Model Inference Parameters

| Flag | Value | Description |
|---|---|---|
| `--tensor-parallel-size` | `1` | Single-GPU (adjust for multi-GPU) |
| `--trust-remote-code` | — | Required by Qwen models |
| `--attention-backend` | `flashinfer` | FlashInfer kernel backend |
| `--moe-backend` | `marlin` | Marlin MoE kernel |
| `--gpu-memory-utilization` | `0.4` | 40 % of GPU memory for model weights |
| `--max-model-len` | `262144` | 256K context window |
| `--max-num-seqs` | `4` | Max concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Max tokens per batch |
| `--enable-chunked-prefill` | — | Improves throughput |
| `--async-scheduling` | — | Async KV cache scheduling |
| `--enable-prefix-caching` | — | KV cache reuse across requests |
| `--limit-mm-per-prompt` | `{"image":4}` | Up to 4 images per prompt |
| `--allowed-media-domains` | `*` | Allowed media source domains |
| `--speculative-config` | MTP, 3 tokens, Triton MoE | Multi-token prediction with Triton MoE backend |
| `--load-format` | `fastsafetensors` | Fast weight loading |
| `--reasoning-parser` | `qwen3` | Qwen3 CoT parser |
| `--override-generation-config` | `{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}` | Server-wide sampling defaults |
| `--chat-template` | `chat_template.jinja` | Custom chat template |
| `--default-chat-template-kwargs` | `{"enable_thinking":true,"preserve_thinking":true}` | Thinking block behaviour |
| `--tool-call-parser` | `qwen3_coder` | Qwen3 tool-call format |
| `--enable-auto-tool-choice` | — | Auto tool selection |

---

## 🧩 Custom Chat Template

The file [`chat_template.jinja`](chat_template.jinja) is a comprehensive Jinja2 template (**v20**) designed for Qwen3.6 with:

- **Multi-modal content** — Renders images and videos as special tokens (`<|image|>`, `<|video|>`)
- **System / Developer / User / Assistant / Tool messages** — Full OpenAI-style role support
- **Thinking / Reasoning Blocks** — Wraps chain-of-thought in `<thinking>...</thinking>`; toggleable via `enable_thinking` and `preserve_thinking` template kwargs
- **Tool Calling** — Serialises function calls into the `<tool_call>\n<function=...>\n</tool_call>` format with JSON parameter rendering
- **Error Recovery** — Consecutive tool-call error detection with ⚠️ retry warnings to the model
- **Auto-disabling Thinking** — When tools are active, thinking is automatically disabled (`auto_disable_thinking_with_tools`)
- **Content Truncation** — `max_tool_arg_chars` / `max_tool_response_chars` for length-limited serialisation
- **Multi-Step Tool Chains** — Detects when tool calls require follow-up turns and preserves reasoning context

### Template Kwargs

| Kwargs | Type | Default | Description |
|---|---|---|---|
| `enable_thinking` | bool | `true` | Enable `<thinking>` blocks in generation |
| `preserve_thinking` | bool | `true` | Preserve thinking blocks from history |
| `auto_disable_thinking_with_tools` | bool | `false` | Disable thinking when tools are defined |
| `add_vision_id` | bool | `false` | Prepend "Picture N:" before vision tokens |
| `max_tool_arg_chars` | int | `0` | Truncate tool arguments (0 = no limit) |
| `max_tool_response_chars` | int | `0` | Truncate tool responses (0 = no limit) |

---

## 📁 Project Structure

```
Qwen3.6-27B-NVFP4-vLLM/
├── README.md             ← This file
├── start.sh              ← Launch script (vLLM container)
├── stop.sh               ← Stop & cleanup script
├── chat_template.jinja   ← Custom Jinja chat template (v20)
└── .gitignore            ← Git ignore rules

# Runtime artifacts (auto-created, gitignored):
#   .vllm.log     — Live container log
#   .vllm.pid     — Container ID file
#   .cache/       — HuggingFace downloads + Triton cache
#   .deps/        — Bundled Python dev .deb packages
#   .bin/         — Bundled GCC binary (ARM64)
```

---

## 🐳 Docker Details

| Property | Value |
|---|---|
| **Image** | `vllm/vllm-openai:nightly` |
| **Container Name** | `qwen3.6-27b-nvfp4-vllm` |
| **Network** | `host` mode (`--network host`) |
| **IPC** | `host` mode (`--ipc host`) |
| **GPUs** | All (`--gpus all`) |
| **Environment** | `VLLM_TARGET_DEVICE=cuda`, `HF_HOME`, `TRITON_CACHE_DIR` |
| **Volumes** | HF cache, Triton cache, chat template, working directory |

---

## 📊 Performance Notes

- **40 % GPU memory** is allocated for weights (`--gpu-memory-utilization 0.4`). The NVFP4 quantisation means the ~27B-parameter model fits in relatively modest VRAM (~24 GB).
- **Speculative decoding** (MTP with 3 tokens) can provide a 1.5–2× speedup on text-heavy prompts.
- **Max 4 concurrent sequences** is conservative; increase `--max-num-seqs` and `--max-num-batched-tokens` if your GPU has spare headroom.
- **Prefix caching** dramatically improves throughput for repeated prompts / system prompts.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `docker is not on PATH` | Install Docker or add it to your `PATH` |
| `vLLM container exited before becoming ready` | Check `.vllm.log` for errors; ensure GPU drivers are installed |
| `Error: cannot access '...'` | Set `HF_TOKEN` and re-run `start.sh` |
| OOM errors | Reduce `--gpu-memory-utilization` or `--max-num-seqs` |
| Model weights not downloading | Verify HF token and network access; check `.cache/huggingface/` |
| Container won't stop | `docker rm -f qwen3.6-27b-nvfp4-vllm` then re-run `stop.sh` |
| Template errors | Check `chat_template.jinja` syntax; refer to Jinja2 docs |

---

## 📝 License

- **Model weights:** Refer to the [HuggingFace repo](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) for licensing details
- **This codebase:** MIT License (or adjust as needed)

---

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3.6 on HuggingFace](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [Marlin MoE](https://github.com/IST-DASLab/marlin)
- [Jinja Templates](https://jinja.palletsprojects.com/)
