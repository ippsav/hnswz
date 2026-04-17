# hnswz (WIP)

A vector database written in Zig. Embeds text locally via [Ollama](https://ollama.com), stores the vectors, and finds nearest neighbors fast using an HNSW index.

> **Status:** Work in progress

## Requirements

- Zig `>= 0.15.2`
- [Ollama](https://ollama.com) running locally with an embedding model pulled (only needed for `build` / `query`)

## Build

```sh
zig build                              # debug
zig build -Doptimize=ReleaseFast       # release
zig build test                         # run tests
```

## Configuration

All runtime knobs live in a JSON config. See [config.example.json](config.example.json).

```json
{
  "embedder": {
    "provider": "ollama",
    "base_url": "http://localhost:11434",
    "model": "qwen3-embedding",
    "dim": 4096,
    "normalize": false,
    "request_timeout_ms": 30000,
    "max_text_bytes": 131072
  },
  "index": {
    "ef_construction": 200,
    "ef_search": 100,
    "max_ef": 200,
    "seed": 42,
    "distance": "cosine"
  },
  "storage": {
    "data_dir": "./data",
    "max_vectors": 10000,
    "upper_pool_slots": 1000,
    "vectors_file": "vectors.hvsf",
    "graph_file": "graph.hgrf",
    "metadata_file": "metadata.hmtf",
    "wal_file": "wal.hwal",
    "lock_file": "hnswz.lock",
    "wal_enabled": true
  },
  "log_level": "info"
}
```

Pass the path with `--config <path>` or set the `HNSWZ_CONFIG` environment variable.

## Usage

Five subcommands: `build`, `query`, `benchmark`, `serve`, `client`.

### `build` — ingest a corpus

Embeds every `.txt` file in `<dir>` via Ollama, builds the HNSW graph, and writes vectors, graph, and filename metadata into `storage.data_dir`.

```sh
hnswz build --config config.json --source ./docs
```

### `query` — interactive REPL

Loads a prebuilt index and reads queries from stdin. Each query is embedded, searched, and the top-k nearest filenames + distances are printed. Exit with `Ctrl-D` or `:q`.

```sh
hnswz query --config config.json [--top-k 5]
```

### `benchmark` — synthetic perf regression harness

Builds an index on random vectors and reports build/search latency percentiles (p50/p90/p95/p99/p100) and throughput. Bypasses Ollama. Config is optional; if provided, `dim`/`ef_*`/`seed` are inherited from it.

```sh
zig build benchmark -- --num-vectors 50000 --validate
# or directly
hnswz benchmark [--config config.json] [flags]
```

Flags:

| flag | default | description |
|---|---|---|
| `--num-vectors <n>` | `10000` | dataset size |
| `--num-queries <n>` | `1000` | held-out queries |
| `--dim <n>` | config or `128` | vector dimension |
| `--ef-construction <n>` | config or `200` | |
| `--ef-search <n>` | config or `100` | |
| `--top-k <n>` | `10` | results per query |
| `--seed <u64>` | config or `42` | PRNG seed |
| `--warmup <n>` | `50` | untimed warmup queries |
| `--validate` | off | compute recall@k against brute force |
| `--json` | off | machine-readable output |
| `--dataset <dir>` | off | load base/query/groundtruth from SIFT-style `.fvecs`/`.ivecs` (sets `--dim` from file; uses shipped groundtruth for recall) |
| `--concurrent-clients <n>` | `1` | TCP search phase clients in parallel (driver threads) |
| `--server-workers <n>` | `0` (auto) | TCP server worker-pool size |

> Run release-mode for meaningful numbers: `zig build -Doptimize=ReleaseFast`.

#### vs `hnswlib`

A turnkey comparison harness lives under [bench/](bench/). It downloads
SIFT1M / siftsmall, runs hnswz and [hnswlib](https://github.com/nmslib/hnswlib)
on the same bytes with matched parameters (M=16, ef_construction=200,
ef_search=100, top-k=10), and diffs the JSON reports.

```sh
bench/run.sh siftsmall   # smoke (~20 s wall on Apple Silicon, <10 MB download)
bench/run.sh sift1m      # headline (~15 min wall, ~500 MB download)
```

**Single-threaded, Apple M-series, cosine distance, L2-normalized
vectors (cosine and L2 induce identical NN ordering on unit vectors
so SIFT's L2 groundtruth stays valid):**

SIFT1M — 1M × 128d, 10k queries:

| metric | hnswz | hnswlib | verdict |
|---|---:|---:|---|
| build wall | 358.6 s | 473.1 s | **hnswz 1.32× faster** |
| build throughput (native API) | 2.8k/s | 2.1k/s | **hnswz 1.30× faster** |
| search QPS | **4.6k/s** | 3.5k/s | **hnswz 1.33× faster** |
| search p50 | 218 µs | 291 µs | hnswz 1.33× faster |
| search p99 | 335 µs | 393 µs | hnswz 1.17× faster |
| recall@10 | 0.9804 | 0.9772 | effectively tied |

siftsmall — 10k × 128d, 100 queries:

| metric | hnswz | hnswlib | verdict |
|---|---:|---:|---|
| build throughput | 13.8k/s | 7.6k/s | hnswz 1.81× faster |
| search QPS | **26.0k/s** | 8.9k/s | hnswz 2.92× faster |
| recall@10 | 0.9920 | 0.9920 | identical |

Caveats:

- Single-threaded on both sides (`threads=1` on hnswlib, `in-process`
  on hnswz). Multi-threaded ingest is a different question — hnswlib
  parallelizes `add_items` natively, hnswz's writer path serializes.
- hnswlib's per-item latency percentiles include ~1–2 µs of Python
  trampoline per call; the "build throughput (native API)" row uses
  its batched `add_items(full)` call to neutralize that.
- Run on your own hardware before drawing conclusions; these numbers
  reflect one laptop, not a cloud fleet.

### `serve` — long-running TCP database

Loads (or creates, if `storage.data_dir` is empty) an index and serves
`INSERT / DELETE / REPLACE / GET / SEARCH / STATS / SNAPSHOT` operations
over a custom binary TCP protocol. Designed for performance first: the
dominant payload is the raw f32 vector blob (dim=4096 × 4 B = 16 KiB), and
any text framing (JSON, SQL, RESP text mode) would be a measurable tax on
both latency and memory.

```sh
hnswz serve --config config.json --listen 127.0.0.1:9000
# or
zig build serve -- --config config.json --listen 127.0.0.1:9000 --auto-snapshot-secs 60
```

Flags:

| flag | default | description |
|---|---|---|
| `--listen <host:port>` | `127.0.0.1:9000` | bind address |
| `--auto-snapshot-secs <n>` | `0` (off) | periodic snapshot cadence |
| `--max-connections <n>` | `64` | concurrent connection cap |
| `--max-frame-bytes <n>` | `64 MiB` | reject frames larger than this |
| `--idle-timeout-secs <n>` | `60` | close idle connections |
| `--workers <n>` | `0` (auto = cpu-2) | worker-pool size for HNSW compute |

**Wire format.** Every frame is a 9-byte header (`u32 body_len | u8
opcode_or_status | u32 req_id`) followed by an opcode-specific payload.
All multi-byte fields are little-endian, matching the on-disk HVSF/HGRF
formats. See [src/protocol.zig](src/protocol.zig) for the authoritative
spec and every opcode's exact byte layout.

**Concurrency.** Main thread runs a kqueue-driven event loop
([src/io/darwin.zig](src/io/darwin.zig)) that handles accept, the
per-connection read/write state machine, and dispatch. HNSW compute runs
on a pool of worker threads ([src/dispatcher.zig](src/dispatcher.zig)),
each with its own `Workspace` and scratch. A `std.Thread.RwLock` guards
the `Store` / `HnswIndex` / `MutableMetadata` triple — searches hold it
shared, inserts/deletes/replace/snapshot hold it exclusive. Workers post
results back over a pipe the loop reads; no polling.

**Text opcodes.** `INSERT_TEXT` / `SEARCH_TEXT` / `REPLACE_TEXT` do the
Ollama HTTP call outside the lock, so a slow embed no longer stalls
other clients. Still, pre-computed `_VEC` variants skip the HTTP
round-trip entirely and are preferred on the hot path.

**Durability.** Every mutation is recorded in a write-ahead log
([src/wal.zig](src/wal.zig)) and fsync'd before the server
acknowledges the client, so acknowledged writes survive a crash. On
startup the WAL is replayed on top of the last snapshot, restoring any
records written since the previous snapshot. When a snapshot completes
(`SNAPSHOT` opcode, `--auto-snapshot-secs` cadence, or clean shutdown),
the WAL is atomically truncated via a rename-over-a-temp-file so a
crash mid-truncate still leaves either the old or new WAL intact. Each
record is CRC32-protected; a torn write at the tail stops replay at the
last fully-valid record and the WAL is truncated back to that point.

**Concurrency safety across processes.** `serve`, `build`, and `query`
all acquire an exclusive `flock(2)` advisory lock on
`<data_dir>/hnswz.lock`. A second `serve` against the same `data_dir`
will refuse to start with a clear error, and a concurrent `build` or
`query` will be rejected the same way. The lock is released on process
exit (including `SIGKILL` / crash), so a stale lock file can never
strand a data directory.

### `client` — one-shot probe against a running `serve`

A companion to `serve` that sends exactly one operation, prints the
response, and exits. Useful for smoke tests, scripting, and ad-hoc
poking. Reuses [src/client.zig](src/client.zig) as its implementation, so
there's no separate client code path to keep in sync.

```sh
hnswz serve --config config.json --listen 127.0.0.1:9000 &

hnswz client --connect 127.0.0.1:9000 ping
hnswz client --connect 127.0.0.1:9000 stats
hnswz client --connect 127.0.0.1:9000 insert-text "machine learning"
hnswz client --connect 127.0.0.1:9000 search-text "ML" --top-k 5
hnswz client --connect 127.0.0.1:9000 get 0 --full-vec
hnswz client --connect 127.0.0.1:9000 delete 0
hnswz client --connect 127.0.0.1:9000 snapshot

# Raw vectors come from a file, stdin, or (for demos) a comma-list.
python -c 'import numpy; numpy.random.rand(128).astype("<f4").tofile("q.f32")'
hnswz client ... search-vec --dim 128 --from-file q.f32 --top-k 10
hnswz client ... insert-vec --dim 4 --literal "1.0,0,0,0"

# Machine-readable output for piping into jq / scripts.
hnswz client ... stats --json
hnswz client ... search-text "ML" --top-k 5 --json | jq '.results[0].id'
```

`--dim` is auto-discovered from `STATS` when omitted on `get`, but the
`*-vec` verbs need it up front to know how many bytes the vector
payload is. `--ef` defaults to `max(top_k, 10)`. Exit codes are 0 on
`status=OK`, 1 on server error (with the diagnostic printed), and 2 on
CLI usage errors.

### Benchmark — in-process vs TCP

The same `benchmark` subcommand drives either transport:

```sh
# baseline (direct HnswIndex calls)
zig build -Doptimize=ReleaseFast
zig build benchmark -- --num-vectors 50000 --json > in-process.json

# over the wire
zig build benchmark -- --transport tcp --num-vectors 50000 --json > tcp.json

# diff the search phase — the delta is the protocol overhead
diff -u <(jq .search in-process.json) <(jq .search tcp.json)
```

A dedicated protocol-floor micro-benchmark measures just the framing
round-trip without any HNSW cost:

```sh
zig build benchmark -- --bench-protocol --num-queries 10000
```
