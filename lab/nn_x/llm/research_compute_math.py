"""Compute math for LLM backbones on the Avito feed-ranking data (reports/llm_research.md, section 3).

Pure python, no torch/GPU. Re-run: python3 nn_x/llm/research_compute_math.py

Inputs (measured on this machine, 2026-11-12):
  * tokens per feed = 1 user token + H history tokens + M candidate tokens (data/flat/*/feeds.parquet):
      train mean 264.5 (p50 271, p90 415, max 557); val 270.7; test 276.5; holdout (M>=50) 331.3.
  * body (non-embedding, text-only) parameter counts from the local safetensors headers:
      Qwen3-0.6B 440.5M, Qwen3.5-2B 1373.3M, Qwen3.5-4B 3570.1M, Qwen3-4B-Instruct-2507 3633.5M;
      Qwen3.5-9B 6.92B from its config.json (weights are NOT in the local HF cache).
  * GPU dense BF16 peak 2250 TFLOP/s (SemiAnalysis / flopper spec sheets).
Assumptions (stated, not measured):
  * training FLOPs per token: LoRA + full activation checkpointing = 6N (2N fwd + 2N input-grad bwd + 2N recompute);
    LoRA without checkpointing = 4N; full fine-tune with checkpointing = 8N.
  * attention / gated-delta-rule / LoRA-matmul overhead +5%; padding efficiency 0.85 (bucketed, unpacked batches).
  * MFU: 15% for 0.6B (d=1024, launch-bound), 25% for 2B-9B. Pessimistic case = half of that.
"""

PEAK = 2250e12
OVH = 1.05
PAD = 0.85
T = dict(train=264.5, val=270.7, test=276.5, holdout=331.3)
N_FEEDS = dict(val=14423, test=28595, holdout=30000)
MODELS = [  # name, body params, layers, hidden, MFU
    ("Qwen3-0.6B", 440.5e6, 28, 1024, 0.15),
    ("Qwen3.5-2B", 1373.3e6, 24, 2048, 0.25),
    ("Qwen3.5-4B", 3570.1e6, 32, 2560, 0.25),
    ("Qwen3-4B-Instruct-2507", 3633.5e6, 36, 2560, 0.25),
    ("Qwen3.5-9B (not local)", 6.92e9, 32, 4096, 0.25),
]
RANKER_TRAIN_FLOP_PER_FEED = 1.8e6 * 3 * 78.3  # nnx_research_llm_arch.md: ~1.8 MFLOP fwd per candidate, x3 fwd+bwd


def fmt_t(sec):
    if sec < 120:
        return f"{sec:.0f} s"
    if sec < 7200:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


def main():
    print("Training cost (LoRA + grad checkpointing = 6NT), exclusive GPU")
    print(f"{'model':26s} {'TFLOP/feed':>10s} {'xRanker':>8s} {'eff TF/s':>8s} {'100K feeds':>11s} {'1M feeds':>9s} "
          f"{'3.15M (1 ep)':>12s} {'pessimistic 3.15M':>17s}")
    for name, n, L, d, mfu in MODELS:
        f = 6 * n * T["train"] * OVH
        eff = PEAK * mfu * PAD
        s100k = 1e5 * f / eff
        print(f"{name:26s} {f / 1e12:10.2f} {f / RANKER_TRAIN_FLOP_PER_FEED:8.0f} {eff / 1e12:8.0f} {fmt_t(s100k):>11s} "
              f"{fmt_t(10 * s100k):>9s} {fmt_t(31.51 * s100k):>12s} {fmt_t(2 * 31.51 * s100k):>17s}")
    print("\nInference (2NT), exclusive GPU")
    print(f"{'model':26s} {'val 14.4K':>10s} {'test 28.6K':>11s} {'holdout 30K':>12s}")
    for name, n, L, d, mfu in MODELS:
        eff = PEAK * mfu * PAD
        row = [fmt_t(N_FEEDS[s] * 2 * n * T[s] * OVH / eff) for s in ("val", "test", "holdout")]
        print(f"{name:26s} {row[0]:>10s} {row[1]:>11s} {row[2]:>12s}")
    print("\nMemory per feed at T=264.5, bf16, with checkpointing (layer inputs + one-layer recompute working set)")
    ff = {"Qwen3-0.6B": 3072, "Qwen3.5-2B": 6144, "Qwen3.5-4B": 9216, "Qwen3-4B-Instruct-2507": 9728,
          "Qwen3.5-9B (not local)": 12288}
    for name, n, L, d, mfu in MODELS:
        stored = L * T["train"] * d * 2
        work = T["train"] * (12 * d + 3 * ff[name]) * 2
        per = stored + work
        w = n * 2
        print(f"{name:26s} weights {w / 1e9:5.1f} GB  act/feed {per / 1e6:5.0f} MB  batch128 {128 * per / 1e9:5.1f} GB  "
              f"batch128 total ~{(w + 128 * per) / 1e9 * 1.3:5.0f} GB (x1.3 allocator/kernel slack)")
    # serialising the 128-d int8 embeddings as text (option b)
    tok_per_item = 128 * 2.2 + 5  # ~2.2 tokens per signed int8 value incl. separator, + field labels
    items = 1 + 185.2 + 78.3
    print(f"\nText serialisation of embeddings: ~{tok_per_item:.0f} tokens/item, ~{tok_per_item * items / 1e3:.0f}K "
          f"tokens/feed vs {T['train']:.0f} soft tokens ({tok_per_item:.0f}x); "
          f"Qwen3.5-2B fwd only on 3.15M feeds = {2 * 1373.3e6 * tok_per_item * items * 3.15e6 / 1e21:.1f} ZFLOP")


if __name__ == "__main__":
    main()
