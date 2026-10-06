"""What Swift's shorter reasoning is worth to an inference provider: memory, throughput, revenue per GPU-hour.

Roofline model of continuous-batching decode on hybrid Qwen3.8 models (Gated DeltaNet + full/sparse attention).
Every architecture number is in ARCH; every hardware and price assumption is a flag. Stdlib only.

    python economics/kv_economics.py                      # defaults: H200, BF16 KV, illustrative prices
    python economics/kv_economics.py --gpu b200 --json out.json
"""
import argparse, json, math

GiB = 1024**3

# From the Qwen config.json files. GDN = Gated DeltaNet (fixed-size recurrent state, no KV growth).
ARCH = {
    "Qwen3.8-27B": dict(
        layers=64, attn_layers=16,                   # full_attention_interval=4: 3x GDN then 1x gated full attention
        q_heads=24, kv_heads=4, head_dim=256,        # attention layers (GQA)
        gdn_qk_heads=16, gdn_v_heads=48, gdn_dim=128, conv_kernel=4,
        params_total=27e9, params_active=27e9, params_read=27e9,  # dense: every weight is read every decode step
        sparse_topk=None,                            # full attention: each step reads the whole KV
        weight_bytes=2, gpus=1,                      # BF16 on one GPU
        swift_reduction=0.585,                       # Swift 1.5: 58.5% fewer thinking tokens (model card)
    ),
    "Qwen3.8-Flash-Next": dict(
        layers=48, attn_layers=12,                   # 36x GDN + 12x Qwen Sparse Attention (QSA)
        q_heads=24, kv_heads=2, head_dim=256,
        gdn_qk_heads=16, gdn_v_heads=48, gdn_dim=128, conv_kernel=4,
        params_total=125e9, params_active=6e9,       # MoE: 512 experts, 10 per token (+51B n-gram embed, 4B MTP)
        experts=512, experts_per_tok=10,
        sparse_topk=2048, qsa_ratio=4, qsa_index_dim=128,  # QSA stores all K/V, reads top-2048 tokens + 1 index key per 4 tokens
        weight_bytes=1, gpus=2,                      # FP8 on two GPUs (180B incl. embeddings does not fit one)
        extra_resident=55e9,                         # n-gram embedding + MTP: resident in memory, looked up not streamed
        swift_reduction=0.558,                       # Swift 1.5: 55.8% mean thinking-token cut on GPQA (63.4% median)
    ),
}

GPUS = {  # memory GiB, HBM TB/s, dense BF16 TFLOPS
    "h100": (80, 3.35, 989), "h200": (141, 4.8, 989), "b200": (180, 8.0, 2250), "mi355x": (288, 8.0, 2300),
}

# Request shapes: (prompt tokens, base thinking tokens, answer tokens). Swift cuts thinking only.
WORKLOADS = {
    "chat w/ reasoning":     (1_000, 3_000, 500),
    "math/science (GPQA-ish)": (500, 12_000, 500),
    "hard reasoning (xhigh)":  (500, 30_000, 800),
    "agentic coding turn":   (30_000, 6_000, 1_000),
}


def kv_bytes_per_token(m, kv_bytes):
    """Growing cache per token: K and V in every attention layer, plus QSA's compressed index key every qsa_ratio tokens."""
    kv = m["attn_layers"] * 2 * m["kv_heads"] * m["head_dim"] * kv_bytes
    if m["sparse_topk"]:
        kv += m["attn_layers"] * m["qsa_index_dim"] * 2 / m["qsa_ratio"]
    return kv


def gdn_state_bytes(m, state_bytes):
    """Fixed per-sequence cost, independent of length: delta-rule state (v_heads x d x d) + short-conv state, per GDN layer."""
    gdn_layers = m["layers"] - m["attn_layers"]
    state = m["gdn_v_heads"] * m["gdn_dim"] * m["gdn_dim"] * state_bytes
    conv_ch = 2 * m["gdn_qk_heads"] * m["gdn_dim"] + m["gdn_v_heads"] * m["gdn_dim"]
    conv = conv_ch * (m["conv_kernel"] - 1) * 2
    return gdn_layers * (state + conv)


def weights_read_per_step(m, batch):
    """Bytes of weights streamed per decode step. MoE: the expected number of distinct experts the batch touches."""
    if "experts" not in m:
        return m["params_read"] * m["weight_bytes"]
    e, k = m["experts"], m["experts_per_tok"]
    expert_params = (m["params_total"] - m["params_active"]) / (1 - k / e)   # solve shared + experts = total, shared + k/e*experts = active
    shared = m["params_total"] - expert_params
    touched = 1 - (1 - k / e) ** batch
    return (shared + expert_params * touched) * m["weight_bytes"]


def decode_step(m, hw, batch, ctx, a):
    """Seconds for one decode step of `batch` sequences at mean context `ctx`. Max of memory and compute time."""
    mem_gib, tbps, tflops = hw
    bw = tbps * 1e12 * m["gpus"] * a.bw_eff
    fl = tflops * 1e12 * m["gpus"] * a.mfu
    kvt = kv_bytes_per_token(m, a.kv_bytes)
    if m["sparse_topk"]:   # QSA reads K/V of top-k tokens + the compressed index for the whole context
        kv_read = m["attn_layers"] * 2 * m["kv_heads"] * m["head_dim"] * a.kv_bytes * min(ctx, m["sparse_topk"]) \
            + m["attn_layers"] * m["qsa_index_dim"] * 2 * ctx / m["qsa_ratio"]
        attn_ctx = min(ctx, m["sparse_topk"])
    else:
        kv_read, attn_ctx = kvt * ctx, ctx
    state_rw = 2 * gdn_state_bytes(m, a.state_bytes)   # GDN state is read and written back every step
    bytes_ = weights_read_per_step(m, batch) + batch * (kv_read + state_rw)
    flops = batch * (2 * m["params_active"] + 4 * m["attn_layers"] * m["q_heads"] * m["head_dim"] * attn_ctx)
    return max(bytes_ / bw, flops / fl) + a.step_overhead


def prefill_seconds(m, hw, prompt, a):
    """GPU-seconds to prefill one prompt (compute-bound): matmuls + causal attention, quadratic only in attention layers."""
    _, _, tflops = hw
    attn_ctx = prompt / 2 if not m["sparse_topk"] else min(prompt / 2, m["sparse_topk"])
    flops = prompt * (2 * m["params_active"] + 4 * m["attn_layers"] * m["q_heads"] * m["head_dim"] * attn_ctx)
    return flops / (tflops * 1e12 * m["gpus"] * a.mfu_prefill)


def serve(m, hw, prompt, out, a):
    """Steady-state serving of one request shape on one instance (m['gpus'] GPUs)."""
    mem_gib = hw[0]
    resident = m["params_total"] * m["weight_bytes"] + m.get("extra_resident", 0) * m["weight_bytes"]
    cache = mem_gib * GiB * m["gpus"] * a.mem_util - resident - a.activation_gib * GiB * m["gpus"]
    if cache <= 0:
        raise SystemExit(f"weights ({resident / GiB:.0f} GiB) don't fit on {m['gpus']}x {mem_gib} GiB; pick a bigger --gpu")
    kvt, state = kv_bytes_per_token(m, a.kv_bytes), gdn_state_bytes(m, a.state_bytes)
    # Continuous batching admits on current usage, so a sequence costs its *average* footprint over its life.
    footprint = state + kvt * (prompt + out / 2)
    batch = max(1, min(a.max_seqs, int(cache // footprint)))
    ctx = prompt + out / 2
    step = decode_step(m, hw, batch, ctx, a)
    decode_gpu_s = out * step / batch                     # this request's share of the batch's decode time
    gpu_s = (decode_gpu_s + prefill_seconds(m, hw, prompt, a)) * m["gpus"]
    rev = (prompt * a.price_in + out * a.price_out) / 1e6
    return dict(
        batch=batch, step_ms=step * 1e3, tok_s_user=1 / step, tok_s_gpu=out / gpu_s,
        latency_s=out * step, footprint_mib=footprint / 2**20, peak_mib=(state + kvt * (prompt + out)) / 2**20,
        mem_time_gib_s=(state * out + kvt * (prompt * out + out * out / 2)) * step / GiB,
        gpu_s=gpu_s, req_per_gpu_h=3600 / gpu_s, revenue=rev, rev_per_gpu_h=rev * 3600 / gpu_s,
        cost=gpu_s / 3600 * a.gpu_hour, margin=(rev - gpu_s / 3600 * a.gpu_hour) / rev,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpu", default="h200", choices=GPUS)
    p.add_argument("--kv-bytes", type=float, default=2, help="2 = BF16 KV cache, 1 = FP8")
    p.add_argument("--state-bytes", type=float, default=4, help="GDN recurrent state dtype: 4 = FP32, 2 = BF16 (check your engine's mamba/SSM cache dtype)")
    p.add_argument("--mem-util", type=float, default=0.90)
    p.add_argument("--activation-gib", type=float, default=6, help="per GPU, reserved for activations/CUDA graphs")
    p.add_argument("--max-seqs", type=int, default=1024, help="scheduler cap on concurrent sequences")
    p.add_argument("--bw-eff", type=float, default=0.75, help="achieved / peak HBM bandwidth")
    p.add_argument("--mfu", type=float, default=0.45, help="decode matmul efficiency")
    p.add_argument("--mfu-prefill", type=float, default=0.55)
    p.add_argument("--step-overhead", type=float, default=0.002, help="seconds of scheduler/launch overhead per step")
    p.add_argument("--price-in", type=float, default=0.30, help="$ per 1M input tokens (illustrative)")
    p.add_argument("--price-out", type=float, default=2.40, help="$ per 1M output tokens (illustrative)")
    p.add_argument("--gpu-hour", type=float, default=3.00, help="$ per GPU-hour, provider's cost (illustrative)")
    p.add_argument("--json", help="also write all numbers here")
    a = p.parse_args()
    hw = GPUS[a.gpu]

    report = {}
    for name, m in ARCH.items():
        kvt, state = kv_bytes_per_token(m, a.kv_bytes), gdn_state_bytes(m, a.state_bytes)
        full = m["layers"] * 2 * m["kv_heads"] * m["head_dim"] * a.kv_bytes
        print(f"\n=== {name} on {m['gpus']}x {a.gpu.upper()}  (Swift 1.5 cuts thinking by {m['swift_reduction']:.1%}) ===")
        print(f"KV cache per token   {kvt / 1024:6.1f} KiB   (an all-full-attention {m['layers']}-layer model: {full / 1024:.0f} KiB, {full / kvt:.1f}x)")
        print(f"GDN state per seq    {state / 2**20:6.1f} MiB   (fixed; = {state / kvt:,.0f} tokens of KV)")
        print(f"{'workload':26} {'':6} {'out':>7} {'batch':>6} {'tok/s/usr':>9} {'latency':>8} {'tok/s/GPU':>9} "
              f"{'req/GPU-h':>9} {'$/req':>8} {'$/GPU-h':>8} {'margin':>7}")
        report[name] = {}
        for wl, (prompt, think, ans) in WORKLOADS.items():
            base_out = think + ans
            swift_out = round(think * (1 - m["swift_reduction"])) + ans
            b, s = serve(m, hw, prompt, base_out, a), serve(m, hw, prompt, swift_out, a)
            for tag, r, out in (("base", b, base_out), ("swift", s, swift_out)):
                print(f"{wl if tag == 'base' else '':26} {tag:6} {out:7,} {r['batch']:6} {r['tok_s_user']:9.1f} {r['latency_s']:7.0f}s "
                      f"{r['tok_s_gpu']:9,.0f} {r['req_per_gpu_h']:9,.0f} {r['revenue']:8.4f} {r['rev_per_gpu_h']:8.2f} {r['margin']:6.0%}")
            d = lambda k: s[k] / b[k]
            print(f"{'':26} {'ratio':6} {swift_out / base_out:7.2f} {d('batch'):6.2f} {d('tok_s_user'):9.2f} {d('latency_s'):8.2f} "
                  f"{d('tok_s_gpu'):9.2f} {d('req_per_gpu_h'):9.2f} {d('revenue'):8.2f} {d('rev_per_gpu_h'):8.2f}"
                  f"   memory x time per request: {d('mem_time_gib_s'):.2f}")
            report[name][wl] = dict(prompt=prompt, base_out=base_out, swift_out=swift_out, base=b, swift=s)
    if a.json:
        json.dump(dict(args=vars(a), report=report), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
