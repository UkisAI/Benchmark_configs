# Economics

What Swift's shorter reasoning is worth to an inference provider. One script, stdlib only:

```bash
python economics/kv_economics.py                  # H200, BF16 KV, FP32 GDN state
python economics/kv_economics.py --gpu b200 --kv-bytes 1 --json out.json
```

Every architecture number is in `ARCH` at the top of the script; every hardware, efficiency and price assumption is a flag.

## The architectures are hybrid, not attention-free

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---|---|
| Layers | 64 = 48 Gated DeltaNet + 16 **full** gated attention | 48 = 36 Gated DeltaNet + 12 Qwen Sparse Attention (QSA) |
| Attention layer | GQA, 24 Q / 4 KV heads × 256 | 24 Q / 2 KV heads × 256, reads top-2,048 tokens |
| KV cache per token (BF16) | **64 KiB**, linear in context | **24.8 KiB**, linear in context (QSA stores all K/V + 1 index key per 4 tokens) |
| vs. same model all full-attention | 4.0× smaller | 3.9× smaller |
| GDN state per sequence (FP32) | 147 MiB, fixed (= 2,349 tokens of KV) | 110 MiB, fixed (= 4,556 tokens of KV) |
| Decode attention compute per step | O(context) in 16 layers: still quadratic over a whole generation | O(2,048) + O(context/4) index scan: near-linear |
| Weights | 27B dense, BF16, 1 GPU | 125B MoE (6B active) + 55B embeddings/MTP, FP8, 2 GPUs |
| Swift 1.5 thinking-token cut | 58.5% | 55.8% mean (63.4% median), GPQA |

One layer in four still keeps a KV cache that grows with every token. That's what makes shorter reasoning worth more than its token count. A pure linear-attention model would have only the fixed state, and cutting tokens would free almost no memory.

## The math

A request has prompt length `P` and generates `T` tokens. `k` is the KV cache per token and `S` is the fixed GDN state. At decode step `t`, the request holds `S + k·(P + t)` bytes.

**Memory × time per request** sums that over the generation:

```
M(T) = Σ_{t=0}^{T} [S + k(P + t)]  =  S·T  +  k·P·T  +  k·T²/2
```

The `T²` term is the one shorter reasoning attacks. Cutting `T` by a fraction `r` cuts it by `1 − (1−r)²`. At `r = 0.585` that's an 83% cut, versus 58.5% for the linear terms. A request's memory-seconds therefore fall faster than its token count. For long reasoning (`T ≫ P`, `k·T ≫ S`), Swift requests cost 0.24–0.34× the memory-time of base requests (table below).

**Throughput per GPU.** Continuous batching fits `B ≈ cache / (S + k·(P + T/2))` sequences at once. Each output token then costs, on average, the KV of a context of `P + T/2`. Swift lowers that average, so:

1. **More sequences fit:** `B` rises up to 1.9×.
2. **Each step reads less KV per sequence:** for the 27B, attention bytes per step scale with `P + T/2`.

Both raise output tokens/s/GPU. Revenue per GPU-hour is `tokens/s/GPU × price`, so it rises with them.

## Results (H200, defaults)

Ratios are Swift ÷ base. Revenue uses illustrative prices of $0.30/M input and $2.40/M output; the ratios barely depend on the price level.

| Workload (prompt / base output) | Model | Concurrent seqs | Tok/s per user | Latency | Tok/s/GPU | Requests/GPU-h | $ per request (user pays) | **$/GPU-h (provider)** | Memory×time per request |
|---|---|---|---|---|---|---|---|---|---|
| Chat with reasoning (1k / 3.5k) | 27B | 1.21× | 0.96× | 0.52× | 1.00× | 2.0× | 0.52× | **1.04×** | 0.43× |
| | Flash-Next | 1.13× | 0.95× | 0.55× | 1.03× | 2.0× | 0.54× | **1.06×** | 0.49× |
| Math/science (0.5k / 12.5k) | 27B | 1.62× | 0.93× | 0.47× | 1.47× | 3.4× | 0.44× | **1.48×** | 0.29× |
| | Flash-Next | 1.42× | 0.87× | 0.53× | 1.23× | 2.6× | 0.47× | **1.23×** | 0.38× |
| Hard reasoning, xhigh (0.5k / 30.8k) | 27B | 1.94× | 0.94× | 0.46× | 1.80× | 4.2× | 0.43× | **1.81×** | 0.24× |
| | Flash-Next | 1.69× | 0.84× | 0.55× | 1.41× | 3.1× | 0.46× | **1.42×** | 0.32× |
| Agentic coding turn (30k / 7k) | 27B | 1.03× | 1.01× | 0.49× | 0.80× | 1.6× | 0.67× | **1.08×** | 0.47× |
| | Flash-Next | 1.05× | 0.98× | 0.53× | 0.88× | 1.7× | 0.69× | **1.16×** | 0.51× |

How to read it:

- **Users:** answers arrive in about half the time and cost 0.43–0.69× as much. Per-token stream speed drops 0–16%, because the provider packs more sequences into each step. Time to answer still roughly halves.
- **Providers:** each GPU serves 1.6–4.2× more requests. Revenue per GPU-hour rises 4–81%. The gain grows with reasoning length, because that's where the `T²` memory term dominates.
- **27B gains more than Flash-Next.** Full attention reads the whole KV every step, so shorter contexts also speed up each step. QSA already caps reads at 2,048 tokens, which leaves Flash-Next mostly the capacity gain.
- **Short and prompt-heavy requests gain least.** In agentic turns the 30k prompt dominates the cache, and the prefill (unchanged) is a bigger share of GPU time. Short chats are dominated by the fixed GDN state `S`. Either way, revenue per GPU-hour still rises only 4–16%.

Sensitivity (`--state-bytes 2`, `--kv-bytes 1`, `--gpu b200/h100`): revenue per GPU-hour on hard reasoning stays at 1.31–1.97×. An FP8 KV cache shrinks the gain most, since it makes `k` smaller relative to `S`. Flash-Next in FP8 doesn't fit on 2× H100.

## When "fewer tokens, more money" holds

Revenue per GPU-hour goes up. **Total revenue at fixed demand goes down.** If a provider serves the same number of requests as before and charges per token, it bills 0.43–0.69× as much per request. It needs 0.24–0.63× the GPUs to do it, but at typical margins that saving doesn't make up the lost revenue. The pitch holds when one of these is true:

1. **Capacity-bound:** the provider is GPU-limited and turning away or queueing traffic, which is the usual state for popular reasoning models. Freed GPUs serve new requests at higher revenue per GPU-hour. For the 27B on math/science, breakeven against the base model's profit needs about 2.2× the request volume, and one GPU can serve 3.4×.
2. **Priced per request, or per-token price raised:** at 0.44× the tokens, a provider can charge up to 2.27× per token and the user still pays the same per answer, with half the latency. Anywhere in between splits the saving.
3. **Fixed fleet:** a provider with a reserved, fixed GPU fleet can serve the same traffic on fewer GPUs, or more traffic on the same ones.

## Limits of the model

- Roofline estimate, not a measurement. Bandwidth efficiency 0.75, decode MFU 0.45, 2 ms of overhead per step. Validate with a vLLM/SGLang load test on the real models before quoting numbers externally.
- Steady state at full load with one request shape at a time. Real traffic mixes shapes and has a long tail, and long tails favour Swift more.
- Swift's cut is applied to thinking tokens only; answer tokens stay as they are. Cuts come from the model cards, measured at xhigh effort.
- No prefix caching, speculative/MTP decoding, or preemption costs.
- Architecture numbers come from the Qwen `config.json` files. GDN state dtype depends on the serving engine (`--state-bytes`).
