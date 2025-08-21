# Llama-2 Test

This repo is based on [Meta's Llama GitHub](https://github.com/meta-llama/llama), with added support for weight and KV cache offloading, and easier testing. Only HuggingFace format models are supported.

Weights and KV cache can be moved between GPU and CPU memory. You can set how many layers stay on CPU, and the rest will stay on GPU. Weights for the next layer can be prefetched from CPU to GPU. If the KV cache for a layer is offloaded to CPU, attention score is done directly on CPU. The offloading approach is similar to DeepSpeed. There are also detailed timers to record the time spent on different parts of the process.

## References:
- [Meta Llama GitHub](https://github.com/meta-llama/llama)
- [DeepSpeed GitHub](https://github.com/deepspeedai/DeepSpeed)