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
    "graph_file": "graph.hgrf"
  },
  "log_level": "info"
}
```

Pass the path with `--config <path>` or set the `HNSWZ_CONFIG` environment variable.

## Usage

Three subcommands: `build`, `query`, `benchmark`.

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

> Run release-mode for meaningful numbers: `zig build -Doptimize=ReleaseFast`.
