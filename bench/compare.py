#!/usr/bin/env python3
"""
Side-by-side diff of two benchmark JSON reports (hnswz vs hnswlib).

Both sides emit schema_version 2 with the same shape; this script
renders a table, a delta column (positive = the second report is
faster / better), and a short parameter-alignment section so you can
see at a glance whether the two runs are actually comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fmt_us(ns: int) -> str:
    if ns == 0:
        return "0"
    return f"{ns / 1000:.1f}µs"


def fmt_qps(qps: float) -> str:
    if qps >= 1e6:
        return f"{qps/1e6:.2f}M/s"
    if qps >= 1e3:
        return f"{qps/1e3:.1f}k/s"
    return f"{qps:.1f}/s"


def compare_metric(first: float, second: float, higher_is_better: bool,
                   subject: str) -> str:
    """
    Frame both numbers from the first column's perspective:
    "<first> is Nx faster/slower than <second>".
    """
    if first <= 0 or second <= 0:
        return "—"
    if higher_is_better:
        ratio = first / second
    else:
        ratio = second / first
    if ratio >= 1.0:
        return f"{subject} {ratio:.2f}x faster"
    return f"{subject} {1/ratio:.2f}x slower"


def check_params(a: dict, b: dict) -> list[str]:
    keys = ["dim", "M", "num_vectors", "num_queries",
            "ef_construction", "ef_search", "top_k"]
    mismatches: list[str] = []
    for k in keys:
        if a["params"].get(k) != b["params"].get(k):
            mismatches.append(f"{k}: {a['params'].get(k)} vs {b['params'].get(k)}")
    if a.get("distance") != b.get("distance"):
        mismatches.append(f"distance: {a.get('distance')} vs {b.get('distance')}")
    return mismatches


def render(a: dict, b: dict) -> None:
    a_lib = a.get("library", Path(sys.argv[1]).stem)
    b_lib = b.get("library", Path(sys.argv[2]).stem)

    print(f"=== comparison: {a_lib}  vs  {b_lib} ===\n")

    mismatches = check_params(a, b)
    if mismatches:
        print("WARNING: parameter mismatch — comparison may be unfair:")
        for m in mismatches:
            print(f"  · {m}")
        print()

    p = a["params"]
    print(f"dataset: {a.get('dataset') or '(random)'}")
    print(f"dim={p['dim']}  n={p['num_vectors']}  q={p['num_queries']}  "
          f"M={p['M']}  ef_cons={p['ef_construction']}  "
          f"ef_search={p['ef_search']}  k={p['top_k']}")
    print()

    col_width = 28
    def row(label: str, va: str, vb: str, verdict: str = "") -> None:
        print(f"  {label:<20} {va:>14}   {vb:>14}   {verdict}")

    header_row = f"  {'metric':<20} {a_lib:>14}   {b_lib:>14}   verdict ({a_lib} vs {b_lib})"
    print(header_row)
    print("  " + "-" * max(len(header_row) - 2, col_width))

    # Build
    ba, bb = a["build"], b["build"]
    row("build wall (s)",
        f"{ba['wall_ns']/1e9:.3f}",
        f"{bb['wall_ns']/1e9:.3f}",
        compare_metric(ba["wall_ns"], bb["wall_ns"], higher_is_better=False,
                       subject=a_lib))
    row("build inserts/s",
        fmt_qps(ba["ops_per_sec"]),
        fmt_qps(bb["ops_per_sec"]),
        compare_metric(ba["ops_per_sec"], bb["ops_per_sec"],
                       higher_is_better=True, subject=a_lib))
    for pct in ("p50", "p95", "p99"):
        row(f"build lat {pct}",
            fmt_us(ba["latency_ns"][pct]),
            fmt_us(bb["latency_ns"][pct]),
            compare_metric(ba["latency_ns"][pct], bb["latency_ns"][pct],
                           higher_is_better=False, subject=a_lib))

    # hnswlib's per-item timing is polluted by Python per-call overhead,
    # so it also emits a `build_batched` wall-clock from a single
    # `add_items(base)` call. That's apples-to-apples against hnswz's
    # per-item wall (hnswz has no batched API; per-item loop IS its
    # native ingest shape).
    b_batched = b.get("build_batched")
    a_batched = a.get("build_batched")
    if b_batched or a_batched:
        a_throughput = a_batched["ops_per_sec"] if a_batched else ba["ops_per_sec"]
        b_throughput = b_batched["ops_per_sec"] if b_batched else bb["ops_per_sec"]
        row("build throughput (native API)",
            fmt_qps(a_throughput), fmt_qps(b_throughput),
            compare_metric(a_throughput, b_throughput,
                           higher_is_better=True, subject=a_lib))

    print()

    # Search
    sa, sb = a["search"], b["search"]
    row("search wall (s)",
        f"{sa['wall_ns']/1e9:.3f}",
        f"{sb['wall_ns']/1e9:.3f}",
        compare_metric(sa["wall_ns"], sb["wall_ns"], higher_is_better=False,
                       subject=a_lib))
    row("search QPS",
        fmt_qps(sa["ops_per_sec"]),
        fmt_qps(sb["ops_per_sec"]),
        compare_metric(sa["ops_per_sec"], sb["ops_per_sec"],
                       higher_is_better=True, subject=a_lib))
    for pct in ("p50", "p95", "p99"):
        row(f"search lat {pct}",
            fmt_us(sa["latency_ns"][pct]),
            fmt_us(sb["latency_ns"][pct]),
            compare_metric(sa["latency_ns"][pct], sb["latency_ns"][pct],
                           higher_is_better=False, subject=a_lib))

    # Recall
    ra = a.get("recall_at_k")
    rb = b.get("recall_at_k")
    if ra is not None and rb is not None:
        diff = ra - rb
        verdict = "identical" if abs(diff) < 1e-4 else (
            f"{a_lib} +{diff:.4f}" if diff > 0 else f"{a_lib} {diff:.4f}"
        )
        row(f"recall@{p['top_k']}", f"{ra:.4f}", f"{rb:.4f}", verdict)
    elif ra is not None or rb is not None:
        row(f"recall@{p['top_k']}",
            f"{ra:.4f}" if ra is not None else "—",
            f"{rb:.4f}" if rb is not None else "—",
            "")

    print()
    print(f"(verdict reads from {a_lib}'s perspective relative to {b_lib})")
    if b.get("build_batched") or a.get("build_batched"):
        print()
        print("Notes on build numbers:")
        print(" · 'build lat pXX' above is per-item: hnswlib's Python-loop per-call")
        print("   overhead (µs-scale) inflates its latency. Use 'build throughput")
        print("   (native API)' for the language-overhead-free comparison.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark JSONs")
    parser.add_argument("a", help="First JSON report (typically hnswz)")
    parser.add_argument("b", help="Second JSON report (typically hnswlib)")
    args = parser.parse_args()
    render(load(Path(args.a)), load(Path(args.b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
