#!/usr/bin/env python3
"""
hnswlib counterpart to `hnswz benchmark --dataset <dir>`.

Reads the same SIFT-family .fvecs / .ivecs files (`*base.fvecs`,
`*query.fvecs`, optional `*groundtruth.ivecs`), builds an hnswlib
index with matched parameters, times build + search, and emits a JSON
report with the same schema hnswz uses so the two are trivially
diff-able.

Distance: the vectors are L2-normalized before insertion on both sides,
so hnswlib's `space='cosine'` and hnswz's cosine-on-unit kernel agree
on ordering. This keeps SIFT's L2-computed groundtruth valid for
recall measurement (cosine and L2 induce the same NN ordering on unit
vectors).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    import hnswlib
except ImportError:
    sys.stderr.write(
        "hnswlib not installed. Run: pip install hnswlib numpy\n"
    )
    sys.exit(2)


def load_fvecs(path: Path) -> np.ndarray:
    """Standard .fvecs: for each vec, int32 dim then dim * float32."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    dim = int(raw[0])
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"{path}: size {raw.size} not multiple of {stride}")
    n = raw.size // stride
    matrix = raw.reshape(n, stride)
    if not (matrix[:, 0] == dim).all():
        raise ValueError(f"{path}: inconsistent per-record dim")
    return matrix[:, 1:].view(np.float32).copy()


def load_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    dim = int(raw[0])
    stride = dim + 1
    if raw.size % stride != 0:
        raise ValueError(f"{path}: size {raw.size} not multiple of {stride}")
    n = raw.size // stride
    matrix = raw.reshape(n, stride)
    if not (matrix[:, 0] == dim).all():
        raise ValueError(f"{path}: inconsistent per-record dim")
    return matrix[:, 1:].astype(np.uint32, copy=True)


def find_one(dir_path: Path, suffix: str) -> Path | None:
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.name.endswith(suffix):
            return p
    return None


def percentile(latencies_ns: np.ndarray, p: float) -> int:
    if latencies_ns.size == 0:
        return 0
    return int(np.percentile(latencies_ns, p * 100.0))


def phase_json(latencies_ns: np.ndarray, wall_ns: int) -> dict:
    return {
        "wall_ns": int(wall_ns),
        "ops": int(latencies_ns.size),
        "ops_per_sec": (float(latencies_ns.size) * 1e9 / wall_ns) if wall_ns > 0 else 0.0,
        "latency_ns": {
            "p50": percentile(latencies_ns, 0.50),
            "p90": percentile(latencies_ns, 0.90),
            "p95": percentile(latencies_ns, 0.95),
            "p99": percentile(latencies_ns, 0.99),
            "p100": int(latencies_ns.max()) if latencies_ns.size else 0,
            "mean": int(latencies_ns.mean()) if latencies_ns.size else 0,
        },
        "overflowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="hnswlib benchmark matching hnswz's JSON schema",
    )
    parser.add_argument("--dataset", required=True,
                        help="Directory with *base.fvecs / *query.fvecs / *groundtruth.ivecs")
    parser.add_argument("--num-vectors", type=int, default=0,
                        help="Cap number of base vectors (0 = all)")
    parser.add_argument("--num-queries", type=int, default=0,
                        help="Cap number of query vectors (0 = all)")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true",
                        help="Compute recall@k (requires groundtruth)")
    parser.add_argument("--json", action="store_true",
                        help="Emit the matched JSON blob (default: pretty)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Build/search threads for hnswlib. Default 1 so "
                             "per-op timings are comparable to hnswz in-process.")
    args = parser.parse_args()

    dir_path = Path(args.dataset)
    if not dir_path.is_dir():
        sys.stderr.write(f"not a directory: {dir_path}\n")
        return 2

    base_path = find_one(dir_path, "base.fvecs")
    query_path = find_one(dir_path, "query.fvecs")
    truth_path = find_one(dir_path, "groundtruth.ivecs")
    if base_path is None or query_path is None:
        sys.stderr.write(f"missing base.fvecs or query.fvecs in {dir_path}\n")
        return 2

    base = load_fvecs(base_path)
    queries = load_fvecs(query_path)
    truth = load_ivecs(truth_path) if (truth_path and args.validate) else None

    dim = base.shape[1]
    if queries.shape[1] != dim:
        sys.stderr.write("dim mismatch between base and query\n")
        return 2

    num_base = base.shape[0] if args.num_vectors <= 0 else min(args.num_vectors, base.shape[0])
    num_queries = queries.shape[0] if args.num_queries <= 0 else min(args.num_queries, queries.shape[0])

    base = base[:num_base]
    queries = queries[:num_queries]
    if truth is not None:
        truth = truth[:num_queries, :args.top_k]

    # L2-normalize so hnswlib's cosine and hnswz's cosineNormalized agree.
    base_norms = np.linalg.norm(base, axis=1, keepdims=True)
    base_norms[base_norms == 0] = 1.0
    base = (base / base_norms).astype(np.float32, copy=False)

    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    queries = (queries / q_norms).astype(np.float32, copy=False)

    # Per-item build — matches hnswz's histogram, but Python's per-call
    # overhead (slice + C-call trampoline ≈ a few µs) inflates hnswlib's
    # latency numbers. We do this in a disposable index, then throw it
    # away and rebuild with a single batched `add_items` on a fresh
    # index to get an honest wall-clock throughput number. Recall is
    # measured on the second index (post-batch) so ef/M/recall are the
    # same on both runs.
    scratch_index = hnswlib.Index(space="cosine", dim=dim)
    scratch_index.init_index(max_elements=num_base, ef_construction=args.ef_construction, M=args.M, random_seed=args.seed)
    scratch_index.set_num_threads(args.threads)

    ids = np.arange(num_base, dtype=np.uint64)

    build_latencies_ns = np.empty(num_base, dtype=np.int64)
    build_wall_start = time.perf_counter_ns()
    for i in range(num_base):
        t0 = time.perf_counter_ns()
        scratch_index.add_items(base[i:i+1], ids[i:i+1])
        build_latencies_ns[i] = time.perf_counter_ns() - t0
    build_wall_ns = time.perf_counter_ns() - build_wall_start
    del scratch_index

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=num_base, ef_construction=args.ef_construction, M=args.M, random_seed=args.seed)
    index.set_num_threads(args.threads)

    # Batched build — hnswlib's intended API shape, no per-call Python
    # trampoline. Measures the library's raw throughput.
    batched_start = time.perf_counter_ns()
    index.add_items(base, ids)
    batched_wall_ns = time.perf_counter_ns() - batched_start

    # Search phase.
    index.set_ef(args.ef_search)

    # Warmup (untimed).
    warmup_n = min(args.warmup, num_queries)
    for qi in range(warmup_n):
        index.knn_query(queries[qi:qi+1], k=args.top_k)

    search_latencies_ns = np.empty(num_queries, dtype=np.int64)
    recall_sum = 0.0
    search_wall_start = time.perf_counter_ns()
    for qi in range(num_queries):
        t0 = time.perf_counter_ns()
        got_ids, _ = index.knn_query(queries[qi:qi+1], k=args.top_k)
        search_latencies_ns[qi] = time.perf_counter_ns() - t0
        if truth is not None:
            hits = np.intersect1d(got_ids[0].astype(np.uint32), truth[qi], assume_unique=False).size
            recall_sum += hits / float(args.top_k)
    search_wall_ns = time.perf_counter_ns() - search_wall_start
    recall = (recall_sum / num_queries) if truth is not None else None

    report = {
        "schema_version": 2,
        "library": "hnswlib",
        "build_mode": "Release",
        "distance": "cosine",
        "params": {
            "seed": args.seed,
            "dim": dim,
            "M": args.M,
            "num_vectors": num_base,
            "num_queries": num_queries,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "top_k": args.top_k,
            "warmup": args.warmup,
            "upper_pool_slots": num_base,
        },
        "dataset": str(dir_path),
        "build": phase_json(build_latencies_ns, build_wall_ns),
        "build_batched": {
            "wall_ns": int(batched_wall_ns),
            "ops": int(num_base),
            "ops_per_sec": (float(num_base) * 1e9 / batched_wall_ns) if batched_wall_ns > 0 else 0.0,
            "_note": "single batched add_items() call — no Python per-item overhead",
        },
        "upper_used": 0,
        "search": phase_json(search_latencies_ns, search_wall_ns),
        "recall_at_k": round(float(recall), 6) if recall is not None else None,
        "hnswlib_threads": args.threads,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    }

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_pretty(report)
    return 0


def fmt_us(ns: int) -> str:
    return f"{ns / 1000:.1f}µs"


def print_pretty(r: dict) -> None:
    p = r["params"]
    print(f"hnswlib benchmark")
    print(f"  library    {r['library']}")
    print(f"  distance   {r['distance']}")
    print(f"  dataset    {r['dataset']}")
    print(f"  dim        {p['dim']}")
    print(f"  M          {p['M']}")
    print(f"  n          {p['num_vectors']}")
    print(f"  q          {p['num_queries']}")
    print(f"  ef_cons    {p['ef_construction']}")
    print(f"  ef_search  {p['ef_search']}")
    print(f"  top_k      {p['top_k']}")
    print(f"  threads    {r['hnswlib_threads']}")
    print()
    b = r["build"]
    bl = b["latency_ns"]
    print(f"Build phase (per-item; includes Python trampoline)")
    print(f"  wall       {b['wall_ns']/1e9:.3f} s")
    print(f"  inserts/s  {b['ops_per_sec']:.1f}")
    print(f"  latency    p50={fmt_us(bl['p50'])} p90={fmt_us(bl['p90'])} p95={fmt_us(bl['p95'])} "
          f"p99={fmt_us(bl['p99'])} p100={fmt_us(bl['p100'])} mean={fmt_us(bl['mean'])}")
    bb = r.get("build_batched")
    if bb:
        print(f"Build phase (batched; no Python per-call overhead)")
        print(f"  wall       {bb['wall_ns']/1e9:.3f} s")
        print(f"  inserts/s  {bb['ops_per_sec']:.1f}")
    s = r["search"]
    sl = s["latency_ns"]
    print()
    print(f"Search phase")
    print(f"  wall       {s['wall_ns']/1e9:.3f} s")
    print(f"  queries/s  {s['ops_per_sec']:.1f}")
    print(f"  latency    p50={fmt_us(sl['p50'])} p90={fmt_us(sl['p90'])} p95={fmt_us(sl['p95'])} "
          f"p99={fmt_us(sl['p99'])} p100={fmt_us(sl['p100'])} mean={fmt_us(sl['mean'])}")
    if r["recall_at_k"] is not None:
        print(f"  recall@{p['top_k']:<3} {r['recall_at_k']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
