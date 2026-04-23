const std = @import("std");

const log = std.log.scoped(.cli);

/// Default value for `--top-k` when the flag is omitted (query REPL).
pub const DEFAULT_TOP_K: usize = 5;

/// Default `--top-k` for the benchmark subcommand. HNSW recall is
/// conventionally reported at k=10, so we match that here.
pub const DEFAULT_BENCH_TOP_K: usize = 10;

/// Environment variable consulted when `--config` is not provided on the
/// command line.
pub const CONFIG_ENV_VAR = "HNSWZ_CONFIG";

/// Exit code used for any CLI / configuration usage error. Matches common
/// convention (2 = misuse of shell builtin / bad invocation).
pub const USAGE_EXIT_CODE: u8 = 2;

pub const Subcommand = enum { build, query, benchmark, serve, client };

/// How the benchmark drives the index: directly in-process, or through
/// a TCP server-client pair on localhost. The delta is the wire overhead.
pub const BenchTransport = enum { in_process, tcp };

/// One-shot verbs for `hnswz client`. Names map 1:1 to protocol opcodes
/// with `_` substituted for `-` in CLI form (e.g. `insert-text`).
pub const ClientVerb = enum {
    ping,
    stats,
    snapshot,
    delete,
    get,
    insert_text,
    insert_vec,
    replace_text,
    replace_vec,
    search_text,
    search_vec,
};

pub const BuildArgs = struct {
    source_dir: ?[]u8 = null,
    /// Embedding worker-pool size. Each worker owns its own Ollama HTTP
    /// client and embeds a distinct file in parallel, while main keeps
    /// the HNSW insert path single-threaded. 0 (or unset) = auto.
    n_workers: ?usize = null,

    pub fn deinit(self: *BuildArgs, allocator: std.mem.Allocator) void {
        if (self.source_dir) |p| allocator.free(p);
        self.* = undefined;
    }
};

pub const QueryArgs = struct {
    connect: ?[]u8 = null,
    top_k: ?usize = null,
    ef: ?u32 = null,

    pub fn deinit(self: *QueryArgs, allocator: std.mem.Allocator) void {
        if (self.connect) |p| allocator.free(p);
        self.* = undefined;
    }
};

pub const BenchArgs = struct {
    num_vectors: ?usize = null,
    num_queries: ?usize = null,
    dim: ?usize = null,
    ef_construction: ?usize = null,
    ef_search: ?usize = null,
    seed: ?u64 = null,
    warmup: ?usize = null,
    top_k: ?usize = null,
    validate: bool = false,
    json: bool = false,
    transport: ?BenchTransport = null,
    /// If true, run only the protocol-floor micro-benchmark (PING RTT +
    /// 1-vector SEARCH_VEC RTT). Skips the build+search workload entirely.
    /// Implies `--transport tcp`.
    bench_protocol: bool = false,
    /// Concurrent client threads for the TCP search phase. Default 1.
    concurrent_clients: ?usize = null,
    /// Server-side worker count for `--transport tcp`. Default 0 = auto.
    server_workers: ?usize = null,
    /// Directory holding SIFT-style `*base.fvecs` / `*query.fvecs` /
    /// optional `*groundtruth.ivecs`. When set, the benchmark replaces
    /// the seeded PRNG with vectors loaded from this directory and
    /// infers `dim` from the file.
    dataset: ?[]u8 = null,

    pub fn deinit(self: *BenchArgs, allocator: std.mem.Allocator) void {
        if (self.dataset) |p| allocator.free(p);
        self.* = undefined;
    }
};

pub const ServeArgs = struct {
    listen: ?[]u8 = null, // "host:port"
    auto_snapshot_secs: ?u32 = null,
    max_connections: ?u32 = null,
    max_frame_bytes: ?usize = null,
    idle_timeout_secs: ?u32 = null,
    /// 0 (or unset) = auto; otherwise the size of the worker pool.
    n_workers: ?usize = null,

    pub fn deinit(self: *ServeArgs, allocator: std.mem.Allocator) void {
        if (self.listen) |p| allocator.free(p);
        self.* = undefined;
    }
};

pub const ClientArgs = struct {
    /// `null` during parsing before the verb positional is seen; populated
    /// as soon as it is. `parseOrExit` enforces presence at dispatch time.
    verb: ?ClientVerb = null,
    connect: ?[]u8 = null,
    /// Verb-specific positional args (e.g. `delete <id>` has pos0="42",
    /// `replace-text <id> <text>` has pos0=id and pos1=text). No verb
    /// takes more than two positionals.
    pos0: ?[]u8 = null,
    pos1: ?[]u8 = null,
    dim: ?usize = null,
    top_k: ?usize = null,
    ef: ?u32 = null,
    from_file: ?[]u8 = null,
    from_stdin: bool = false,
    literal: ?[]u8 = null,
    full_vec: bool = false,
    json: bool = false,

    pub fn deinit(self: *ClientArgs, allocator: std.mem.Allocator) void {
        if (self.connect) |p| allocator.free(p);
        if (self.pos0) |p| allocator.free(p);
        if (self.pos1) |p| allocator.free(p);
        if (self.from_file) |p| allocator.free(p);
        if (self.literal) |p| allocator.free(p);
        self.* = undefined;
    }
};

/// Tagged-union of subcommand-specific arg bundles. Callers switch on
/// `args.command` and receive a typed pointer to the active variant —
/// no more "did this flag apply to benchmark or client?" ambiguity.
pub const Command = union(Subcommand) {
    build: BuildArgs,
    query: QueryArgs,
    benchmark: BenchArgs,
    serve: ServeArgs,
    client: ClientArgs,

    pub fn deinit(self: *Command, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*v| v.deinit(allocator),
        }
    }
};

/// Parsed command-line arguments. `config_path` is shared across all
/// subcommands (they all accept `--config` + `HNSWZ_CONFIG`); everything
/// else lives on the active `command` variant.
///
/// `config_path` is optional because `benchmark` runs fine without one
/// (it has its own defaults). `build` and `serve` enforce presence at
/// dispatch time in main.zig.
pub const Args = struct {
    config_path: ?[]u8 = null,
    command: Command,

    pub fn deinit(self: *Args, allocator: std.mem.Allocator) void {
        if (self.config_path) |p| allocator.free(p);
        self.command.deinit(allocator);
        self.* = undefined;
    }
};

/// Errors `parse` can surface. `HelpRequested` is modelled as an error so
/// `parse` stays a pure function; `parseOrExit` turns it into a normal
/// `exit(0)` with usage printed.
pub const ParseError = error{
    HelpRequested,
    MissingValue,
    UnknownArgument,
    UnknownSubcommand,
    MissingConfig,
    MissingSubcommand,
    TopKZero,
    InvalidTopK,
    InvalidBenchmarkNumber,
    InvalidBenchmarkTransport,
    InvalidBuildNumber,
    InvalidServeNumber,
    InvalidClientVerb,
    InvalidClientNumber,
    TooManyClientPositionals,
    MissingClientVerb,
} || std.mem.Allocator.Error || std.process.GetEnvVarOwnedError;

fn takeFlag(arg: []const u8, comptime long: []const u8) bool {
    return std.mem.eql(u8, arg, long);
}

/// Returns the raw (non-duped) value slice for `--long <v>` or `--long=v`.
fn takeRaw(
    it: anytype,
    arg: []const u8,
    comptime long: []const u8,
) ParseError!?[]const u8 {
    if (std.mem.eql(u8, arg, long)) {
        return it.next() orelse return error.MissingValue;
    }
    if (std.mem.startsWith(u8, arg, long ++ "=")) {
        return arg[long.len + 1 ..];
    }
    return null;
}

/// Like `takeRaw` but also matches a short alias (e.g. `-k` for `--top-k`).
fn takeRawEither(
    it: anytype,
    arg: []const u8,
    comptime long: []const u8,
    comptime short: []const u8,
) ParseError!?[]const u8 {
    if (std.mem.eql(u8, arg, short)) {
        return it.next() orelse return error.MissingValue;
    }
    return takeRaw(it, arg, long);
}

/// Dup `--long`'s value into `slot`, freeing any previous contents so
/// "last one wins" doesn't leak. Returns `true` if `arg` was consumed.
fn takeStrInto(
    allocator: std.mem.Allocator,
    slot: *?[]u8,
    it: anytype,
    arg: []const u8,
    comptime long: []const u8,
) ParseError!bool {
    const raw = (try takeRaw(it, arg, long)) orelse return false;
    const dup = try allocator.dupe(u8, raw);
    if (slot.*) |old| allocator.free(old);
    slot.* = dup;
    return true;
}

/// Parse `--long` / `--long=` as integer `T`. Returns null if unmatched;
/// raises `parse_err` on malformed input so callers get named errors like
/// `InvalidBenchmarkNumber` at the exact flag site.
fn takeInt(
    comptime T: type,
    it: anytype,
    arg: []const u8,
    comptime long: []const u8,
    parse_err: ParseError,
) ParseError!?T {
    const raw = (try takeRaw(it, arg, long)) orelse return null;
    return std.fmt.parseInt(T, raw, 10) catch return parse_err;
}

/// Like `takeInt` but with a short alias too.
fn takeIntEither(
    comptime T: type,
    it: anytype,
    arg: []const u8,
    comptime long: []const u8,
    comptime short: []const u8,
    parse_err: ParseError,
) ParseError!?T {
    const raw = (try takeRawEither(it, arg, long, short)) orelse return null;
    return std.fmt.parseInt(T, raw, 10) catch return parse_err;
}

fn parseSubcommand(arg: []const u8) ?Subcommand {
    if (std.mem.eql(u8, arg, "build")) return .build;
    if (std.mem.eql(u8, arg, "query")) return .query;
    if (std.mem.eql(u8, arg, "benchmark")) return .benchmark;
    if (std.mem.eql(u8, arg, "serve")) return .serve;
    if (std.mem.eql(u8, arg, "client")) return .client;
    return null;
}

fn parseClientVerb(arg: []const u8) ?ClientVerb {
    if (std.mem.eql(u8, arg, "ping")) return .ping;
    if (std.mem.eql(u8, arg, "stats")) return .stats;
    if (std.mem.eql(u8, arg, "snapshot")) return .snapshot;
    if (std.mem.eql(u8, arg, "delete")) return .delete;
    if (std.mem.eql(u8, arg, "get")) return .get;
    if (std.mem.eql(u8, arg, "insert-text")) return .insert_text;
    if (std.mem.eql(u8, arg, "insert-vec")) return .insert_vec;
    if (std.mem.eql(u8, arg, "replace-text")) return .replace_text;
    if (std.mem.eql(u8, arg, "replace-vec")) return .replace_vec;
    if (std.mem.eql(u8, arg, "search-text")) return .search_text;
    if (std.mem.eql(u8, arg, "search-vec")) return .search_vec;
    return null;
}

fn parseBenchTransport(s: []const u8) ParseError!BenchTransport {
    if (std.mem.eql(u8, s, "in-process")) return .in_process;
    if (std.mem.eql(u8, s, "tcp")) return .tcp;
    return error.InvalidBenchmarkTransport;
}

/// Fall back to `$HNSWZ_CONFIG` when `--config` wasn't supplied. Only the
/// subcommands that actually consume config (build / benchmark / serve)
/// call this; query and client ignore config and reject `--config` as an
/// unknown argument.
fn resolveConfigEnv(allocator: std.mem.Allocator, config_path: *?[]u8) ParseError!void {
    if (config_path.* != null) return;
    config_path.* = std.process.getEnvVarOwned(allocator, CONFIG_ENV_VAR) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => return err,
    };
}

fn parseBuildArgs(
    allocator: std.mem.Allocator,
    it: anytype,
    config_path: *?[]u8,
) ParseError!BuildArgs {
    var out: BuildArgs = .{};
    errdefer out.deinit(allocator);

    while (it.next()) |arg| {
        if (takeFlag(arg, "-h") or takeFlag(arg, "--help")) {
            return error.HelpRequested;
        } else if (try takeStrInto(allocator, config_path, it, arg, "--config")) {
            // consumed
        } else if (try takeStrInto(allocator, &out.source_dir, it, arg, "--source")) {
            // consumed
        } else if (try takeInt(usize, it, arg, "--workers", error.InvalidBuildNumber)) |v| {
            out.n_workers = v;
        } else if (try takeInt(usize, it, arg, "--n-workers", error.InvalidBuildNumber)) |v| {
            out.n_workers = v;
        } else {
            return error.UnknownArgument;
        }
    }
    try resolveConfigEnv(allocator, config_path);
    return out;
}

fn parseQueryArgs(
    allocator: std.mem.Allocator,
    it: anytype,
) ParseError!QueryArgs {
    var out: QueryArgs = .{};
    errdefer out.deinit(allocator);

    while (it.next()) |arg| {
        if (takeFlag(arg, "-h") or takeFlag(arg, "--help")) {
            return error.HelpRequested;
        } else if (try takeStrInto(allocator, &out.connect, it, arg, "--connect")) {
            // consumed
        } else if (try takeIntEither(usize, it, arg, "--top-k", "-k", error.InvalidTopK)) |v| {
            if (v == 0) return error.TopKZero;
            out.top_k = v;
        } else if (try takeInt(u32, it, arg, "--ef", error.InvalidClientNumber)) |v| {
            out.ef = v;
        } else {
            return error.UnknownArgument;
        }
    }
    return out;
}

fn parseBenchArgs(
    allocator: std.mem.Allocator,
    it: anytype,
    config_path: *?[]u8,
) ParseError!BenchArgs {
    var out: BenchArgs = .{};
    errdefer out.deinit(allocator);

    while (it.next()) |arg| {
        if (takeFlag(arg, "-h") or takeFlag(arg, "--help")) {
            return error.HelpRequested;
        } else if (try takeStrInto(allocator, config_path, it, arg, "--config")) {
            // consumed
        } else if (try takeInt(usize, it, arg, "--num-vectors", error.InvalidBenchmarkNumber)) |v| {
            out.num_vectors = v;
        } else if (try takeInt(usize, it, arg, "--num-queries", error.InvalidBenchmarkNumber)) |v| {
            out.num_queries = v;
        } else if (try takeInt(usize, it, arg, "--dim", error.InvalidBenchmarkNumber)) |v| {
            out.dim = v;
        } else if (try takeInt(usize, it, arg, "--ef-construction", error.InvalidBenchmarkNumber)) |v| {
            out.ef_construction = v;
        } else if (try takeInt(usize, it, arg, "--ef-search", error.InvalidBenchmarkNumber)) |v| {
            out.ef_search = v;
        } else if (try takeInt(u64, it, arg, "--seed", error.InvalidBenchmarkNumber)) |v| {
            out.seed = v;
        } else if (try takeInt(usize, it, arg, "--warmup", error.InvalidBenchmarkNumber)) |v| {
            out.warmup = v;
        } else if (try takeIntEither(usize, it, arg, "--top-k", "-k", error.InvalidTopK)) |v| {
            if (v == 0) return error.TopKZero;
            out.top_k = v;
        } else if (takeFlag(arg, "--validate")) {
            out.validate = true;
        } else if (takeFlag(arg, "--json")) {
            out.json = true;
        } else if (try takeRaw(it, arg, "--transport")) |raw| {
            out.transport = try parseBenchTransport(raw);
        } else if (takeFlag(arg, "--bench-protocol")) {
            out.bench_protocol = true;
        } else if (try takeInt(usize, it, arg, "--concurrent-clients", error.InvalidBenchmarkNumber)) |v| {
            out.concurrent_clients = v;
        } else if (try takeInt(usize, it, arg, "--server-workers", error.InvalidBenchmarkNumber)) |v| {
            out.server_workers = v;
        } else if (try takeStrInto(allocator, &out.dataset, it, arg, "--dataset")) {
            // consumed
        } else {
            return error.UnknownArgument;
        }
    }
    try resolveConfigEnv(allocator, config_path);
    return out;
}

fn parseServeArgs(
    allocator: std.mem.Allocator,
    it: anytype,
    config_path: *?[]u8,
) ParseError!ServeArgs {
    var out: ServeArgs = .{};
    errdefer out.deinit(allocator);

    while (it.next()) |arg| {
        if (takeFlag(arg, "-h") or takeFlag(arg, "--help")) {
            return error.HelpRequested;
        } else if (try takeStrInto(allocator, config_path, it, arg, "--config")) {
            // consumed
        } else if (try takeStrInto(allocator, &out.listen, it, arg, "--listen")) {
            // consumed
        } else if (try takeInt(u32, it, arg, "--auto-snapshot-secs", error.InvalidServeNumber)) |v| {
            out.auto_snapshot_secs = v;
        } else if (try takeInt(u32, it, arg, "--max-connections", error.InvalidServeNumber)) |v| {
            out.max_connections = v;
        } else if (try takeInt(usize, it, arg, "--max-frame-bytes", error.InvalidServeNumber)) |v| {
            out.max_frame_bytes = v;
        } else if (try takeInt(u32, it, arg, "--idle-timeout-secs", error.InvalidServeNumber)) |v| {
            out.idle_timeout_secs = v;
        } else if (try takeInt(usize, it, arg, "--n-workers", error.InvalidServeNumber)) |v| {
            out.n_workers = v;
        } else if (try takeInt(usize, it, arg, "--workers", error.InvalidServeNumber)) |v| {
            out.n_workers = v;
        } else {
            return error.UnknownArgument;
        }
    }
    try resolveConfigEnv(allocator, config_path);
    return out;
}

fn parseClientArgs(
    allocator: std.mem.Allocator,
    it: anytype,
) ParseError!ClientArgs {
    var out: ClientArgs = .{};
    errdefer out.deinit(allocator);

    while (it.next()) |arg| {
        if (takeFlag(arg, "-h") or takeFlag(arg, "--help")) {
            return error.HelpRequested;
        } else if (try takeStrInto(allocator, &out.connect, it, arg, "--connect")) {
            // consumed
        } else if (try takeIntEither(usize, it, arg, "--top-k", "-k", error.InvalidTopK)) |v| {
            if (v == 0) return error.TopKZero;
            out.top_k = v;
        } else if (try takeInt(u32, it, arg, "--ef", error.InvalidClientNumber)) |v| {
            out.ef = v;
        } else if (try takeInt(usize, it, arg, "--dim", error.InvalidClientNumber)) |v| {
            out.dim = v;
        } else if (try takeStrInto(allocator, &out.from_file, it, arg, "--from-file")) {
            // consumed
        } else if (takeFlag(arg, "--from-stdin")) {
            out.from_stdin = true;
        } else if (try takeStrInto(allocator, &out.literal, it, arg, "--literal")) {
            // consumed
        } else if (takeFlag(arg, "--full-vec")) {
            out.full_vec = true;
        } else if (takeFlag(arg, "--json")) {
            out.json = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional: verb first, then verb-specific args.
            if (out.verb == null) {
                out.verb = parseClientVerb(arg) orelse return error.InvalidClientVerb;
            } else if (out.pos0 == null) {
                out.pos0 = try allocator.dupe(u8, arg);
            } else if (out.pos1 == null) {
                out.pos1 = try allocator.dupe(u8, arg);
            } else {
                return error.TooManyClientPositionals;
            }
        } else {
            return error.UnknownArgument;
        }
    }
    return out;
}

/// Parse the current process's argv. Convenience wrapper over `parseFromIter`.
pub fn parse(allocator: std.mem.Allocator) ParseError!Args {
    var it = std.process.args();
    _ = it.next(); // drop argv[0]
    return parseFromIter(allocator, &it);
}

/// Pure argument parser. Takes anything that exposes
/// `fn next(*@This()) ?[]const u8`. Both `std.process.ArgIterator` and the
/// test `SliceIter` satisfy this. Program name must already be consumed.
pub fn parseFromIter(allocator: std.mem.Allocator, it: anytype) ParseError!Args {
    var config_path: ?[]u8 = null;
    errdefer if (config_path) |p| allocator.free(p);

    // Phase 1: the only universal pre-subcommand flag is --help. --config
    // is subcommand-local so `hnswz query --config x` and
    // `hnswz --config x query` both error (query doesn't accept config).
    var subcommand: ?Subcommand = null;
    while (it.next()) |arg| {
        if (takeFlag(arg, "-h") or takeFlag(arg, "--help")) {
            return error.HelpRequested;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            subcommand = parseSubcommand(arg) orelse return error.UnknownSubcommand;
            break;
        } else {
            return error.UnknownArgument;
        }
    }

    const sub = subcommand orelse return error.MissingSubcommand;

    // Phase 2: hand the rest of argv to the matching sub-parser. Parsers
    // that consume config (build/benchmark/serve) resolve HNSWZ_CONFIG
    // themselves; the others don't see config_path at all.
    const command: Command = switch (sub) {
        .build => .{ .build = try parseBuildArgs(allocator, it, &config_path) },
        .query => .{ .query = try parseQueryArgs(allocator, it) },
        .benchmark => .{ .benchmark = try parseBenchArgs(allocator, it, &config_path) },
        .serve => .{ .serve = try parseServeArgs(allocator, it, &config_path) },
        .client => .{ .client = try parseClientArgs(allocator, it) },
    };

    return .{ .config_path = config_path, .command = command };
}

/// Emit the usage text to stdout.
pub fn printUsage() !void {
    const usage =
        \\Usage:
        \\  hnswz build      --config <path> --source <dir> [--workers <n>]
        \\  hnswz query      [--connect host:port] [--top-k <n>] [--ef <n>]
        \\  hnswz benchmark  [--config <path>] [benchmark flags]
        \\  hnswz serve      --config <path> [serve flags]
        \\  hnswz client     [--connect host:port] <verb> [verb args/flags]
        \\
        \\Subcommands:
        \\  build      Ingest every .txt file in <dir>, embed via Ollama, build
        \\             the HNSW graph, and persist vectors/graph/metadata to
        \\             config.storage.data_dir.
        \\  query      Interactive REPL over a running `hnswz serve`. Bare
        \\             text lines run search-text; colon commands (:stats,
        \\             :get, :insert, :replace, :delete, :snapshot, :ping,
        \\             :help) invoke the other server verbs. Ctrl-D or :q exits.
        \\  benchmark  Run an end-to-end build+search workload on synthetic
        \\             vectors and print latency/throughput. Intended as the
        \\             performance regression signal across commits.
        \\  serve      Load (or create) an index and accept insert/delete/
        \\             replace/get/search operations over a custom binary TCP
        \\             protocol. See docs/ for the wire format.
        \\  client     One-shot client against a running `hnswz serve`. Sends
        \\             one operation, prints the response, exits. See below.
        \\
        \\Options:
        \\  --config <path>   Path to JSON config (or set HNSWZ_CONFIG env var).
        \\                    Required for build/serve; optional for benchmark
        \\                    (benchmark has its own defaults; if a config is
        \\                    provided, dim/ef_*/seed are inherited).
        \\  --source <dir>    (build only) Directory of .txt files to ingest.
        \\  --workers <n>     (build only) Embedding worker-pool size. Each
        \\                    worker runs an independent Ollama HTTP client
        \\                    in parallel; main thread keeps HNSW insertion
        \\                    strictly in filename-sorted order. Default 0
        \\                    (auto = min(cpu, 8)). Also accepts --n-workers.
        \\  --top-k <n>       (query/benchmark/client) Results per search.
        \\                    Default 5 for query/client, 10 for benchmark.
        \\
        \\Query options:
        \\  --connect <host:port>  Server to connect to. Default 127.0.0.1:9000.
        \\  --ef <n>               Search-time candidate pool. Defaults to
        \\                         max(top_k, 10).
        \\
        \\Benchmark options:
        \\  --num-vectors <n>      Dataset size. Default 10000.
        \\  --num-queries <n>      Held-out queries. Default 1000.
        \\  --dim <n>              Vector dimension. Default config.embedder.dim, else 128.
        \\  --ef-construction <n>  Default config.index.ef_construction, else 200.
        \\  --ef-search <n>        Default config.index.ef_search, else 100.
        \\  --seed <u64>           PRNG seed. Default config.index.seed, else 42.
        \\  --warmup <n>           Untimed warmup queries. Default 50.
        \\  --validate             Compute recall@k vs brute force (slower).
        \\  --json                 Emit machine-readable JSON to stdout.
        \\  --transport <t>        "in-process" (default) or "tcp". TCP mode
        \\                         spawns a server on a worker thread and
        \\                         drives it over the wire so the delta vs
        \\                         in-process is the protocol overhead.
        \\  --bench-protocol       Skip the normal build+search workload;
        \\                         measure PING RTT and 1-vector SEARCH_VEC
        \\                         RTT only. Implies --transport tcp.
        \\  --dataset <dir>        Load base/query/groundtruth from a
        \\                         SIFT-style fvecs/ivecs directory. Sets
        \\                         --dim from the file and uses the
        \\                         shipped groundtruth for recall
        \\                         (no brute-force pass).
        \\
        \\Serve options:
        \\  --listen <host:port>         Bind address. Default 127.0.0.1:9000.
        \\  --auto-snapshot-secs <n>     Periodic snapshot cadence. Default 0 (off).
        \\  --max-connections <n>        Concurrent connection cap. Default 64.
        \\  --max-frame-bytes <n>        Reject frames larger than this. Default 64 MiB.
        \\  --idle-timeout-secs <n>      Close idle connections. Default 60.
        \\
        \\Client verbs:
        \\  ping                                 Protocol floor round-trip.
        \\  stats                                Server-reported index state.
        \\  snapshot                             Ask the server to flush.
        \\  get <id>                             Fetch vector + name for <id>.
        \\  delete <id>                          Tombstone <id>.
        \\  insert-text <text>                   Embed and insert; prints id.
        \\  replace-text <id> <text>             Embed and overwrite <id>.
        \\  search-text <text>                   Embed and top-k search.
        \\  insert-vec                           Insert a raw vector.
        \\  replace-vec <id>                     Overwrite <id> with a raw vector.
        \\  search-vec                           Top-k search on a raw vector.
        \\
        \\Client options:
        \\  --connect <host:port>    Default 127.0.0.1:9000.
        \\  --dim <n>                Required for *-vec verbs; otherwise auto-
        \\                           discovered via STATS.
        \\  --top-k <n>              Results per search. Default 5.
        \\  --ef <n>                 Search-time candidate pool. Defaults to
        \\                           max(top_k, 10).
        \\  --from-file <path>       Read raw little-endian f32 bytes for the
        \\                           vector from this file. (*-vec verbs)
        \\  --from-stdin             Read raw f32 bytes from stdin.
        \\  --literal "a,b,c"        Comma-separated floats. For quick demos
        \\                           at small dim; use --from-file for real data.
        \\  --full-vec               (get) Print the full vector in pretty mode.
        \\                           In --json mode the vector is always included.
        \\  --json                   Machine-readable output (JSON) on stdout.
        \\
    ;
    var buf: [4096]u8 = undefined;
    var stdout_file = std.fs.File.stdout().writer(&buf);
    const w = &stdout_file.interface;
    try w.writeAll(usage);
    try w.flush();
}

/// Human-readable explanation for a `ParseError`. `HelpRequested` is
/// intentionally excluded — the caller handles it separately (exit 0).
fn describeParseError(err: ParseError) []const u8 {
    return switch (err) {
        error.MissingValue => "flag requires a value",
        error.UnknownArgument => "unknown argument",
        error.UnknownSubcommand => "unknown subcommand (expected 'build', 'query', 'benchmark', 'serve', or 'client')",
        error.MissingConfig => "no config specified (use --config <path> or set " ++ CONFIG_ENV_VAR ++ ")",
        error.MissingSubcommand => "missing subcommand (build | query | benchmark | serve | client)",
        error.TopKZero => "--top-k must be > 0",
        error.InvalidTopK => "--top-k must be a non-negative integer",
        error.InvalidBenchmarkNumber => "benchmark integer flag could not be parsed",
        error.InvalidBenchmarkTransport => "--transport must be 'in-process' or 'tcp'",
        error.InvalidBuildNumber => "build integer flag could not be parsed",
        error.InvalidServeNumber => "serve integer flag could not be parsed",
        error.InvalidClientVerb => "unknown client verb (expected one of: ping, stats, snapshot, get, delete, insert-text, insert-vec, replace-text, replace-vec, search-text, search-vec)",
        error.InvalidClientNumber => "client integer flag could not be parsed",
        error.TooManyClientPositionals => "too many positional arguments for this client verb",
        error.MissingClientVerb => "'hnswz client' requires a verb (e.g. 'ping', 'stats', 'insert-text ...')",
        error.HelpRequested => unreachable,
        else => "CLI parse failed",
    };
}

/// Parse argv, handling every terminal outcome (success, `--help`, error)
/// so the caller stays trivial.
///
///   * Success          → returns `Args`.
///   * `--help`/`-h`    → prints usage to stdout, exits 0.
///   * Any other error  → logs a diagnostic, prints usage, exits with
///                        `USAGE_EXIT_CODE`.
pub fn parseOrExit(allocator: std.mem.Allocator) Args {
    return parse(allocator) catch |err| {
        if (err == error.HelpRequested) {
            printUsage() catch {};
            std.process.exit(0);
        }

        switch (err) {
            error.MissingValue,
            error.UnknownArgument,
            error.UnknownSubcommand,
            error.MissingConfig,
            error.MissingSubcommand,
            error.TopKZero,
            error.InvalidTopK,
            error.InvalidBenchmarkNumber,
            error.InvalidBenchmarkTransport,
            error.InvalidBuildNumber,
            error.InvalidServeNumber,
            error.InvalidClientVerb,
            error.InvalidClientNumber,
            error.TooManyClientPositionals,
            error.MissingClientVerb,
            => log.err("{s}", .{describeParseError(err)}),
            error.HelpRequested => unreachable,
            else => log.err("CLI parse failed: {s}", .{@errorName(err)}),
        }
        printUsage() catch {};
        std.process.exit(USAGE_EXIT_CODE);
    };
}

const testing = std.testing;

/// Minimal iterator backing so tests don't have to shell out to a child
/// process to exercise `parseFromIter`.
const SliceIter = struct {
    slice: []const []const u8,
    i: usize = 0,

    pub fn next(self: *SliceIter) ?[]const u8 {
        if (self.i >= self.slice.len) return null;
        const v = self.slice[self.i];
        self.i += 1;
        return v;
    }
};

test "parse: build with space-separated flags" {
    var it: SliceIter = .{ .slice = &.{ "build", "--config", "cfg.json", "--source", "./docs" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.build, std.meta.activeTag(args.command));
    try testing.expectEqualStrings("cfg.json", args.config_path.?);
    try testing.expectEqualStrings("./docs", args.command.build.source_dir.?);
    try testing.expect(args.command.build.n_workers == null);
}

test "parse: build --workers" {
    var it: SliceIter = .{ .slice = &.{
        "build",     "--config", "cfg.json",
        "--source",  "./docs",   "--workers",
        "8",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 8), args.command.build.n_workers.?);
}

test "parse: build --n-workers alias" {
    var it: SliceIter = .{ .slice = &.{ "build", "--config=cfg.json", "--source=./docs", "--n-workers=4" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), args.command.build.n_workers.?);
}

test "parse: build --workers rejects non-numeric" {
    var it: SliceIter = .{ .slice = &.{ "build", "--workers", "abc" } };
    try testing.expectError(error.InvalidBuildNumber, parseFromIter(testing.allocator, &it));
}

test "parse: query with equals-form flags and custom top-k" {
    var it: SliceIter = .{ .slice = &.{ "query", "--top-k=42" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.query, std.meta.activeTag(args.command));
    try testing.expect(args.config_path == null);
    try testing.expectEqual(@as(usize, 42), args.command.query.top_k.?);
}

test "parse: -k short form" {
    var it: SliceIter = .{ .slice = &.{ "query", "-k", "7" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 7), args.command.query.top_k.?);
}

test "parse: later --config overrides earlier and does not leak" {
    var it: SliceIter = .{ .slice = &.{ "build", "--config", "first.json", "--config=second.json" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqualStrings("second.json", args.config_path.?);
}

test "parse: query rejects --config" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "cfg.json" } };
    try testing.expectError(error.UnknownArgument, parseFromIter(testing.allocator, &it));
}

test "parse: client rejects --config" {
    var it: SliceIter = .{ .slice = &.{ "client", "--config", "cfg.json", "ping" } };
    try testing.expectError(error.UnknownArgument, parseFromIter(testing.allocator, &it));
}

test "parse: --config before subcommand rejected" {
    // --config is subcommand-local, so even a legitimate target like
    // `build` can't sit behind a pre-subcommand --config.
    var it: SliceIter = .{ .slice = &.{ "--config", "cfg.json", "build" } };
    try testing.expectError(error.UnknownArgument, parseFromIter(testing.allocator, &it));
}

test "parse: --help returns HelpRequested" {
    var it: SliceIter = .{ .slice = &.{"--help"} };
    try testing.expectError(error.HelpRequested, parseFromIter(testing.allocator, &it));
}

test "parse: -h returns HelpRequested even with other args" {
    var it: SliceIter = .{ .slice = &.{ "build", "--config", "c.json", "-h" } };
    try testing.expectError(error.HelpRequested, parseFromIter(testing.allocator, &it));
}

test "parse: missing subcommand" {
    var it: SliceIter = .{ .slice = &.{} };
    try testing.expectError(error.MissingSubcommand, parseFromIter(testing.allocator, &it));
}

test "parse: unknown subcommand" {
    var it: SliceIter = .{ .slice = &.{"frobnicate"} };
    try testing.expectError(error.UnknownSubcommand, parseFromIter(testing.allocator, &it));
}

test "parse: --top-k 0 rejected" {
    var it: SliceIter = .{ .slice = &.{ "query", "--top-k", "0" } };
    try testing.expectError(error.TopKZero, parseFromIter(testing.allocator, &it));
}

test "parse: non-numeric --top-k rejected" {
    var it: SliceIter = .{ .slice = &.{ "query", "--top-k", "abc" } };
    try testing.expectError(error.InvalidTopK, parseFromIter(testing.allocator, &it));
}

test "parse: missing value after --config" {
    var it: SliceIter = .{ .slice = &.{ "build", "--config" } };
    try testing.expectError(error.MissingValue, parseFromIter(testing.allocator, &it));
}

test "parse: unknown flag rejected" {
    var it: SliceIter = .{ .slice = &.{ "build", "--config", "c.json", "--source", "d", "--bogus" } };
    try testing.expectError(error.UnknownArgument, parseFromIter(testing.allocator, &it));
}

test "parse: extra positional after subcommand rejected" {
    var it: SliceIter = .{ .slice = &.{ "build", "leftover", "--config", "c.json" } };
    try testing.expectError(error.UnknownArgument, parseFromIter(testing.allocator, &it));
}

test "parse: benchmark with no config succeeds" {
    var it: SliceIter = .{ .slice = &.{"benchmark"} };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.benchmark, std.meta.activeTag(args.command));
    try testing.expect(args.config_path == null);
    try testing.expect(!args.command.benchmark.validate);
    try testing.expect(!args.command.benchmark.json);
}

test "parse: benchmark with all knobs set" {
    var it: SliceIter = .{ .slice = &.{
        "benchmark",
        "--num-vectors",
        "5000",
        "--num-queries=200",
        "--dim",
        "64",
        "--ef-construction=150",
        "--ef-search",
        "80",
        "--top-k",
        "10",
        "--seed=7",
        "--warmup",
        "25",
        "--validate",
        "--json",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    const b = args.command.benchmark;
    try testing.expectEqual(Subcommand.benchmark, std.meta.activeTag(args.command));
    try testing.expectEqual(@as(usize, 5000), b.num_vectors.?);
    try testing.expectEqual(@as(usize, 200), b.num_queries.?);
    try testing.expectEqual(@as(usize, 64), b.dim.?);
    try testing.expectEqual(@as(usize, 150), b.ef_construction.?);
    try testing.expectEqual(@as(usize, 80), b.ef_search.?);
    try testing.expectEqual(@as(usize, 10), b.top_k.?);
    try testing.expectEqual(@as(u64, 7), b.seed.?);
    try testing.expectEqual(@as(usize, 25), b.warmup.?);
    try testing.expect(b.validate);
    try testing.expect(b.json);
}

test "parse: benchmark with malformed number rejected" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--num-vectors", "abc" } };
    try testing.expectError(error.InvalidBenchmarkNumber, parseFromIter(testing.allocator, &it));
}

test "parse: benchmark --seed=nonnumeric rejected" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--seed=xyz" } };
    try testing.expectError(error.InvalidBenchmarkNumber, parseFromIter(testing.allocator, &it));
}

test "parse: serve with all knobs" {
    var it: SliceIter = .{ .slice = &.{
        "serve",
        "--config",
        "c.json",
        "--listen",
        "0.0.0.0:9999",
        "--auto-snapshot-secs=30",
        "--max-connections",
        "128",
        "--max-frame-bytes=1048576",
        "--idle-timeout-secs",
        "120",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    const s = args.command.serve;
    try testing.expectEqual(Subcommand.serve, std.meta.activeTag(args.command));
    try testing.expectEqualStrings("c.json", args.config_path.?);
    try testing.expectEqualStrings("0.0.0.0:9999", s.listen.?);
    try testing.expectEqual(@as(u32, 30), s.auto_snapshot_secs.?);
    try testing.expectEqual(@as(u32, 128), s.max_connections.?);
    try testing.expectEqual(@as(usize, 1_048_576), s.max_frame_bytes.?);
    try testing.expectEqual(@as(u32, 120), s.idle_timeout_secs.?);
}

test "parse: serve equals-form --listen" {
    var it: SliceIter = .{ .slice = &.{ "serve", "--config=c.json", "--listen=127.0.0.1:5000" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqualStrings("127.0.0.1:5000", args.command.serve.listen.?);
}

test "parse: serve rejects non-numeric snapshot-secs" {
    var it: SliceIter = .{ .slice = &.{ "serve", "--auto-snapshot-secs", "foo" } };
    try testing.expectError(error.InvalidServeNumber, parseFromIter(testing.allocator, &it));
}

test "parse: benchmark --transport tcp" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--transport", "tcp" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqual(BenchTransport.tcp, args.command.benchmark.transport.?);
    try testing.expect(!args.command.benchmark.bench_protocol);
}

test "parse: benchmark --transport=in-process" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--transport=in-process" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqual(BenchTransport.in_process, args.command.benchmark.transport.?);
}

test "parse: benchmark --transport rejects unknown" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--transport", "udp" } };
    try testing.expectError(error.InvalidBenchmarkTransport, parseFromIter(testing.allocator, &it));
}

test "parse: benchmark --bench-protocol" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--bench-protocol" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expect(args.command.benchmark.bench_protocol);
}

test "parse: client ping with explicit connect" {
    var it: SliceIter = .{ .slice = &.{ "client", "--connect", "127.0.0.1:9000", "ping" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    const c = args.command.client;
    try testing.expectEqual(Subcommand.client, std.meta.activeTag(args.command));
    try testing.expectEqual(ClientVerb.ping, c.verb.?);
    try testing.expectEqualStrings("127.0.0.1:9000", c.connect.?);
    try testing.expect(c.pos0 == null);
}

test "parse: client delete with id positional" {
    var it: SliceIter = .{ .slice = &.{ "client", "delete", "42" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(ClientVerb.delete, args.command.client.verb.?);
    try testing.expectEqualStrings("42", args.command.client.pos0.?);
    try testing.expect(args.command.client.pos1 == null);
}

test "parse: client replace-text with id + text positionals" {
    var it: SliceIter = .{ .slice = &.{ "client", "replace-text", "7", "new body" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(ClientVerb.replace_text, args.command.client.verb.?);
    try testing.expectEqualStrings("7", args.command.client.pos0.?);
    try testing.expectEqualStrings("new body", args.command.client.pos1.?);
}

test "parse: client insert-vec with --dim and --literal" {
    var it: SliceIter = .{ .slice = &.{
        "client",    "insert-vec",
        "--dim",     "4",
        "--literal", "1.0,2.0,3.0,4.0",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    const c = args.command.client;
    try testing.expectEqual(ClientVerb.insert_vec, c.verb.?);
    try testing.expectEqual(@as(usize, 4), c.dim.?);
    try testing.expectEqualStrings("1.0,2.0,3.0,4.0", c.literal.?);
}

test "parse: client search-text with top-k and ef" {
    var it: SliceIter = .{ .slice = &.{
        "client",     "search-text", "foo bar",
        "--top-k=10", "--ef=40",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    const c = args.command.client;
    try testing.expectEqual(ClientVerb.search_text, c.verb.?);
    try testing.expectEqualStrings("foo bar", c.pos0.?);
    try testing.expectEqual(@as(usize, 10), c.top_k.?);
    try testing.expectEqual(@as(u32, 40), c.ef.?);
}

test "parse: client --json flag" {
    var it: SliceIter = .{ .slice = &.{ "client", "stats", "--json" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expect(args.command.client.json);
}

test "parse: client unknown verb rejected" {
    var it: SliceIter = .{ .slice = &.{ "client", "frobnicate" } };
    try testing.expectError(error.InvalidClientVerb, parseFromIter(testing.allocator, &it));
}

test "parse: client too many positionals rejected" {
    var it: SliceIter = .{ .slice = &.{ "client", "get", "1", "2", "3" } };
    try testing.expectError(error.TooManyClientPositionals, parseFromIter(testing.allocator, &it));
}

test "parse: client --from-file and --from-stdin" {
    var it: SliceIter = .{ .slice = &.{
        "client",      "insert-vec",
        "--dim",       "128",
        "--from-file", "vec.f32",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqualStrings("vec.f32", args.command.client.from_file.?);
    try testing.expect(!args.command.client.from_stdin);
}
