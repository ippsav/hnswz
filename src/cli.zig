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

/// Parsed command-line arguments. Owns the heap allocations referenced by
/// `config_path` and `source_dir`; call `deinit` once the value is no
/// longer needed.
///
/// `config_path` is optional because the `benchmark` subcommand runs fine
/// without one (it has its own defaults). `build` and `query` enforce
/// presence at dispatch time.
///
/// `top_k` is shared between `query` and `benchmark`; when null each
/// subcommand applies its own default (DEFAULT_TOP_K vs DEFAULT_BENCH_TOP_K).
pub const Args = struct {
    subcommand: Subcommand,
    config_path: ?[]u8 = null,
    source_dir: ?[]u8 = null, // build only
    top_k: ?usize = null, // query / benchmark

    // benchmark-only knobs. All optional; main.zig resolves missing values
    // against the config (if any) and then against benchmark defaults.
    bench_num_vectors: ?usize = null,
    bench_num_queries: ?usize = null,
    bench_dim: ?usize = null,
    bench_ef_construction: ?usize = null,
    bench_ef_search: ?usize = null,
    bench_seed: ?u64 = null,
    bench_warmup: ?usize = null,
    bench_validate: bool = false,
    bench_json: bool = false,
    bench_transport: ?BenchTransport = null,
    /// Concurrent client threads for the TCP search phase. Default 1.
    bench_concurrent_clients: ?usize = null,
    /// Server-side worker count for `--transport tcp`. Default 0 = auto.
    bench_server_workers: ?usize = null,
    /// If true, run only the protocol-floor micro-benchmark (PING RTT +
    /// 1-vector SEARCH_VEC RTT). Skips the build+search workload entirely.
    /// Implies `--transport tcp`.
    bench_protocol: bool = false,
    /// Directory holding SIFT-style `*base.fvecs` / `*query.fvecs` /
    /// optional `*groundtruth.ivecs`. When set, the benchmark replaces
    /// the seeded PRNG with vectors loaded from this directory and
    /// infers `dim` from the file.
    bench_dataset: ?[]u8 = null,

    // serve-only knobs. All optional; main.zig falls back to ServeOptions defaults.
    serve_listen: ?[]u8 = null, // "host:port"
    serve_auto_snapshot_secs: ?u32 = null,
    serve_max_connections: ?u32 = null,
    serve_max_frame_bytes: ?usize = null,
    serve_idle_timeout_secs: ?u32 = null,
    /// 0 (or unset) = auto; otherwise the size of the worker pool.
    serve_n_workers: ?usize = null,

    // client-only knobs. `client_pos0` / `client_pos1` carry verb-specific
    // positional args (e.g. `delete <id>` has pos0="42"). Interpretation
    // happens in runClient, not here.
    client_verb: ?ClientVerb = null,
    client_connect: ?[]u8 = null,
    client_pos0: ?[]u8 = null,
    client_pos1: ?[]u8 = null,
    client_dim: ?usize = null,
    client_ef: ?u32 = null,
    client_from_file: ?[]u8 = null,
    client_from_stdin: bool = false,
    client_literal: ?[]u8 = null,
    client_full_vec: bool = false,
    client_json: bool = false,

    pub fn deinit(self: *Args, allocator: std.mem.Allocator) void {
        if (self.config_path) |p| allocator.free(p);
        if (self.source_dir) |s| allocator.free(s);
        if (self.serve_listen) |p| allocator.free(p);
        if (self.client_connect) |p| allocator.free(p);
        if (self.client_pos0) |p| allocator.free(p);
        if (self.client_pos1) |p| allocator.free(p);
        if (self.client_from_file) |p| allocator.free(p);
        if (self.client_literal) |p| allocator.free(p);
        if (self.bench_dataset) |p| allocator.free(p);
        self.* = undefined;
    }
};

/// Errors `parse` can surface. Ordered roughly as the user would hit them.
/// `HelpRequested` is modelled as an error so `parse` stays a pure function;
/// `parseOrExit` turns it into a normal `exit(0)` with usage printed.
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
    InvalidServeNumber,
    InvalidClientVerb,
    InvalidClientNumber,
    TooManyClientPositionals,
    MissingClientVerb,
} || std.mem.Allocator.Error || std.process.GetEnvVarOwnedError;

/// Parse the current process's argv. Convenience wrapper over `parseFromIter`
/// that consumes the program name and delegates to the pure parser.
pub fn parse(allocator: std.mem.Allocator) ParseError!Args {
    var it = std.process.args();
    _ = it.next(); // drop argv[0]
    return parseFromIter(allocator, &it);
}

/// Pure argument parser. Takes anything that exposes `fn next(*@This()) ?[]const u8`
/// (or a null-terminated variant — both `std.process.ArgIterator` and the
/// test `SliceIter` below satisfy this). The caller is expected to have
/// already consumed the program name.
pub fn parseFromIter(allocator: std.mem.Allocator, it: anytype) ParseError!Args {
    var subcommand: ?Subcommand = null;
    var config_path: ?[]u8 = null;
    var source_dir: ?[]u8 = null;
    var top_k: ?usize = null;

    var bench_num_vectors: ?usize = null;
    var bench_num_queries: ?usize = null;
    var bench_dim: ?usize = null;
    var bench_ef_construction: ?usize = null;
    var bench_ef_search: ?usize = null;
    var bench_seed: ?u64 = null;
    var bench_warmup: ?usize = null;
    var bench_validate = false;
    var bench_json = false;
    var bench_transport: ?BenchTransport = null;
    var bench_protocol = false;
    var bench_concurrent_clients: ?usize = null;
    var bench_server_workers: ?usize = null;
    var bench_dataset: ?[]u8 = null;

    var serve_listen: ?[]u8 = null;
    var serve_auto_snapshot_secs: ?u32 = null;
    var serve_max_connections: ?u32 = null;
    var serve_max_frame_bytes: ?usize = null;
    var serve_idle_timeout_secs: ?u32 = null;
    var serve_n_workers: ?usize = null;

    var client_verb: ?ClientVerb = null;
    var client_connect: ?[]u8 = null;
    var client_pos0: ?[]u8 = null;
    var client_pos1: ?[]u8 = null;
    var client_dim: ?usize = null;
    var client_ef: ?u32 = null;
    var client_from_file: ?[]u8 = null;
    var client_from_stdin = false;
    var client_literal: ?[]u8 = null;
    var client_full_vec = false;
    var client_json = false;

    errdefer {
        if (config_path) |p| allocator.free(p);
        if (source_dir) |p| allocator.free(p);
        if (serve_listen) |p| allocator.free(p);
        if (client_connect) |p| allocator.free(p);
        if (client_pos0) |p| allocator.free(p);
        if (client_pos1) |p| allocator.free(p);
        if (client_from_file) |p| allocator.free(p);
        if (client_literal) |p| allocator.free(p);
        if (bench_dataset) |p| allocator.free(p);
    }

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else if (std.mem.eql(u8, arg, "--config")) {
            const p = it.next() orelse return error.MissingValue;
            if (config_path) |old| allocator.free(old);
            config_path = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--config=")) {
            if (config_path) |old| allocator.free(old);
            config_path = try allocator.dupe(u8, arg["--config=".len..]);
        } else if (std.mem.eql(u8, arg, "--source")) {
            const p = it.next() orelse return error.MissingValue;
            if (source_dir) |old| allocator.free(old);
            source_dir = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--source=")) {
            if (source_dir) |old| allocator.free(old);
            source_dir = try allocator.dupe(u8, arg["--source=".len..]);
        } else if (std.mem.eql(u8, arg, "-k") or std.mem.eql(u8, arg, "--top-k")) {
            const p = it.next() orelse return error.MissingValue;
            top_k = std.fmt.parseInt(usize, p, 10) catch return error.InvalidTopK;
        } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
            top_k = std.fmt.parseInt(usize, arg["--top-k=".len..], 10) catch return error.InvalidTopK;
        } else if (std.mem.eql(u8, arg, "--num-vectors")) {
            const p = it.next() orelse return error.MissingValue;
            bench_num_vectors = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--num-vectors=")) {
            bench_num_vectors = parseUsize(arg["--num-vectors=".len..]) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--num-queries")) {
            const p = it.next() orelse return error.MissingValue;
            bench_num_queries = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--num-queries=")) {
            bench_num_queries = parseUsize(arg["--num-queries=".len..]) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--dim")) {
            const p = it.next() orelse return error.MissingValue;
            if (subcommand == .client) {
                client_dim = parseUsize(p) orelse return error.InvalidClientNumber;
            } else {
                bench_dim = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
            }
        } else if (std.mem.startsWith(u8, arg, "--dim=")) {
            const v = arg["--dim=".len..];
            if (subcommand == .client) {
                client_dim = parseUsize(v) orelse return error.InvalidClientNumber;
            } else {
                bench_dim = parseUsize(v) orelse return error.InvalidBenchmarkNumber;
            }
        } else if (std.mem.eql(u8, arg, "--ef-construction")) {
            const p = it.next() orelse return error.MissingValue;
            bench_ef_construction = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--ef-construction=")) {
            bench_ef_construction = parseUsize(arg["--ef-construction=".len..]) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--ef-search")) {
            const p = it.next() orelse return error.MissingValue;
            bench_ef_search = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--ef-search=")) {
            bench_ef_search = parseUsize(arg["--ef-search=".len..]) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const p = it.next() orelse return error.MissingValue;
            bench_seed = std.fmt.parseInt(u64, p, 10) catch return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            bench_seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            const p = it.next() orelse return error.MissingValue;
            bench_warmup = std.fmt.parseInt(usize, p, 10) catch return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            bench_warmup = std.fmt.parseInt(usize, arg["--warmup=".len..], 10) catch return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--validate")) {
            bench_validate = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            // Both benchmark and client honor --json; only one is live per
            // invocation so setting both fields is harmless.
            bench_json = true;
            client_json = true;
        } else if (std.mem.eql(u8, arg, "--transport")) {
            const p = it.next() orelse return error.MissingValue;
            bench_transport = try parseBenchTransport(p);
        } else if (std.mem.startsWith(u8, arg, "--transport=")) {
            bench_transport = try parseBenchTransport(arg["--transport=".len..]);
        } else if (std.mem.eql(u8, arg, "--bench-protocol")) {
            bench_protocol = true;
        } else if (std.mem.eql(u8, arg, "--concurrent-clients")) {
            const p = it.next() orelse return error.MissingValue;
            bench_concurrent_clients = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--concurrent-clients=")) {
            bench_concurrent_clients = parseUsize(arg["--concurrent-clients=".len..]) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--server-workers")) {
            const p = it.next() orelse return error.MissingValue;
            bench_server_workers = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--server-workers=")) {
            bench_server_workers = parseUsize(arg["--server-workers=".len..]) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.eql(u8, arg, "--dataset")) {
            const p = it.next() orelse return error.MissingValue;
            if (bench_dataset) |old| allocator.free(old);
            bench_dataset = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--dataset=")) {
            if (bench_dataset) |old| allocator.free(old);
            bench_dataset = try allocator.dupe(u8, arg["--dataset=".len..]);
        } else if (std.mem.eql(u8, arg, "--listen")) {
            const p = it.next() orelse return error.MissingValue;
            if (serve_listen) |old| allocator.free(old);
            serve_listen = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--listen=")) {
            if (serve_listen) |old| allocator.free(old);
            serve_listen = try allocator.dupe(u8, arg["--listen=".len..]);
        } else if (std.mem.eql(u8, arg, "--auto-snapshot-secs")) {
            const p = it.next() orelse return error.MissingValue;
            serve_auto_snapshot_secs = std.fmt.parseInt(u32, p, 10) catch return error.InvalidServeNumber;
        } else if (std.mem.startsWith(u8, arg, "--auto-snapshot-secs=")) {
            serve_auto_snapshot_secs = std.fmt.parseInt(u32, arg["--auto-snapshot-secs=".len..], 10) catch return error.InvalidServeNumber;
        } else if (std.mem.eql(u8, arg, "--max-connections")) {
            const p = it.next() orelse return error.MissingValue;
            serve_max_connections = std.fmt.parseInt(u32, p, 10) catch return error.InvalidServeNumber;
        } else if (std.mem.startsWith(u8, arg, "--max-connections=")) {
            serve_max_connections = std.fmt.parseInt(u32, arg["--max-connections=".len..], 10) catch return error.InvalidServeNumber;
        } else if (std.mem.eql(u8, arg, "--max-frame-bytes")) {
            const p = it.next() orelse return error.MissingValue;
            serve_max_frame_bytes = parseUsize(p) orelse return error.InvalidServeNumber;
        } else if (std.mem.startsWith(u8, arg, "--max-frame-bytes=")) {
            serve_max_frame_bytes = parseUsize(arg["--max-frame-bytes=".len..]) orelse return error.InvalidServeNumber;
        } else if (std.mem.eql(u8, arg, "--idle-timeout-secs")) {
            const p = it.next() orelse return error.MissingValue;
            serve_idle_timeout_secs = std.fmt.parseInt(u32, p, 10) catch return error.InvalidServeNumber;
        } else if (std.mem.startsWith(u8, arg, "--idle-timeout-secs=")) {
            serve_idle_timeout_secs = std.fmt.parseInt(u32, arg["--idle-timeout-secs=".len..], 10) catch return error.InvalidServeNumber;
        } else if (std.mem.eql(u8, arg, "--n-workers") or std.mem.eql(u8, arg, "--workers")) {
            const p = it.next() orelse return error.MissingValue;
            serve_n_workers = parseUsize(p) orelse return error.InvalidServeNumber;
        } else if (std.mem.startsWith(u8, arg, "--n-workers=")) {
            serve_n_workers = parseUsize(arg["--n-workers=".len..]) orelse return error.InvalidServeNumber;
        } else if (std.mem.startsWith(u8, arg, "--workers=")) {
            serve_n_workers = parseUsize(arg["--workers=".len..]) orelse return error.InvalidServeNumber;
        } else if (std.mem.eql(u8, arg, "--connect")) {
            const p = it.next() orelse return error.MissingValue;
            if (client_connect) |old| allocator.free(old);
            client_connect = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--connect=")) {
            if (client_connect) |old| allocator.free(old);
            client_connect = try allocator.dupe(u8, arg["--connect=".len..]);
        } else if (std.mem.eql(u8, arg, "--ef")) {
            const p = it.next() orelse return error.MissingValue;
            client_ef = std.fmt.parseInt(u32, p, 10) catch return error.InvalidClientNumber;
        } else if (std.mem.startsWith(u8, arg, "--ef=")) {
            client_ef = std.fmt.parseInt(u32, arg["--ef=".len..], 10) catch return error.InvalidClientNumber;
        } else if (std.mem.eql(u8, arg, "--from-file")) {
            const p = it.next() orelse return error.MissingValue;
            if (client_from_file) |old| allocator.free(old);
            client_from_file = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--from-file=")) {
            if (client_from_file) |old| allocator.free(old);
            client_from_file = try allocator.dupe(u8, arg["--from-file=".len..]);
        } else if (std.mem.eql(u8, arg, "--from-stdin")) {
            client_from_stdin = true;
        } else if (std.mem.eql(u8, arg, "--literal")) {
            const p = it.next() orelse return error.MissingValue;
            if (client_literal) |old| allocator.free(old);
            client_literal = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--literal=")) {
            if (client_literal) |old| allocator.free(old);
            client_literal = try allocator.dupe(u8, arg["--literal=".len..]);
        } else if (std.mem.eql(u8, arg, "--full-vec")) {
            client_full_vec = true;
        } else if (subcommand == null and !std.mem.startsWith(u8, arg, "-")) {
            if (std.mem.eql(u8, arg, "build")) {
                subcommand = .build;
            } else if (std.mem.eql(u8, arg, "query")) {
                subcommand = .query;
            } else if (std.mem.eql(u8, arg, "benchmark")) {
                subcommand = .benchmark;
            } else if (std.mem.eql(u8, arg, "serve")) {
                subcommand = .serve;
            } else if (std.mem.eql(u8, arg, "client")) {
                subcommand = .client;
            } else {
                return error.UnknownSubcommand;
            }
        } else if (subcommand == .client and client_verb == null and !std.mem.startsWith(u8, arg, "-")) {
            client_verb = parseClientVerb(arg) orelse return error.InvalidClientVerb;
        } else if (subcommand == .client and !std.mem.startsWith(u8, arg, "-")) {
            // Positional arg for the active verb. No verb takes more than 2.
            if (client_pos0 == null) {
                client_pos0 = try allocator.dupe(u8, arg);
            } else if (client_pos1 == null) {
                client_pos1 = try allocator.dupe(u8, arg);
            } else {
                return error.TooManyClientPositionals;
            }
        } else {
            return error.UnknownArgument;
        }
    }

    // Fall back to the env var if --config was not given.
    if (config_path == null) {
        config_path = std.process.getEnvVarOwned(allocator, CONFIG_ENV_VAR) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => return err,
        };
    }
    if (subcommand == null) return error.MissingSubcommand;
    // config_path requirement is enforced at dispatch time (benchmark doesn't need one).
    if (top_k) |k| if (k == 0) return error.TopKZero;

    return .{
        .subcommand = subcommand.?,
        .config_path = config_path,
        .source_dir = source_dir,
        .top_k = top_k,
        .bench_num_vectors = bench_num_vectors,
        .bench_num_queries = bench_num_queries,
        .bench_dim = bench_dim,
        .bench_ef_construction = bench_ef_construction,
        .bench_ef_search = bench_ef_search,
        .bench_seed = bench_seed,
        .bench_warmup = bench_warmup,
        .bench_validate = bench_validate,
        .bench_json = bench_json,
        .bench_transport = bench_transport,
        .bench_protocol = bench_protocol,
        .bench_concurrent_clients = bench_concurrent_clients,
        .bench_server_workers = bench_server_workers,
        .bench_dataset = bench_dataset,
        .serve_listen = serve_listen,
        .serve_auto_snapshot_secs = serve_auto_snapshot_secs,
        .serve_max_connections = serve_max_connections,
        .serve_max_frame_bytes = serve_max_frame_bytes,
        .serve_idle_timeout_secs = serve_idle_timeout_secs,
        .serve_n_workers = serve_n_workers,
        .client_verb = client_verb,
        .client_connect = client_connect,
        .client_pos0 = client_pos0,
        .client_pos1 = client_pos1,
        .client_dim = client_dim,
        .client_ef = client_ef,
        .client_from_file = client_from_file,
        .client_from_stdin = client_from_stdin,
        .client_literal = client_literal,
        .client_full_vec = client_full_vec,
        .client_json = client_json,
    };
}

fn parseUsize(s: []const u8) ?usize {
    return std.fmt.parseInt(usize, s, 10) catch null;
}

fn parseBenchTransport(s: []const u8) ParseError!BenchTransport {
    if (std.mem.eql(u8, s, "in-process")) return .in_process;
    if (std.mem.eql(u8, s, "tcp")) return .tcp;
    return error.InvalidBenchmarkTransport;
}

fn parseClientVerb(s: []const u8) ?ClientVerb {
    if (std.mem.eql(u8, s, "ping")) return .ping;
    if (std.mem.eql(u8, s, "stats")) return .stats;
    if (std.mem.eql(u8, s, "snapshot")) return .snapshot;
    if (std.mem.eql(u8, s, "delete")) return .delete;
    if (std.mem.eql(u8, s, "get")) return .get;
    if (std.mem.eql(u8, s, "insert-text")) return .insert_text;
    if (std.mem.eql(u8, s, "insert-vec")) return .insert_vec;
    if (std.mem.eql(u8, s, "replace-text")) return .replace_text;
    if (std.mem.eql(u8, s, "replace-vec")) return .replace_vec;
    if (std.mem.eql(u8, s, "search-text")) return .search_text;
    if (std.mem.eql(u8, s, "search-vec")) return .search_vec;
    return null;
}

/// Emit the usage text to stdout.
pub fn printUsage() !void {
    const usage =
        \\Usage:
        \\  hnswz build      --config <path> --source <dir>
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

/// Human-readable explanation for a `ParseError`. `--help` is intentionally
/// excluded — the caller handles it separately (exit 0, not exit 2).
fn describeParseError(err: ParseError) []const u8 {
    return switch (err) {
        error.MissingValue => "flag requires a value",
        error.UnknownArgument => "unknown argument",
        error.UnknownSubcommand => "unknown subcommand (expected 'build', 'query', 'benchmark', or 'serve')",
        error.MissingConfig => "no config specified (use --config <path> or set " ++ CONFIG_ENV_VAR ++ ")",
        error.MissingSubcommand => "missing subcommand (build | query | benchmark | serve)",
        error.TopKZero => "--top-k must be > 0",
        error.InvalidTopK => "--top-k must be a non-negative integer",
        error.InvalidBenchmarkNumber => "benchmark integer flag could not be parsed",
        error.InvalidBenchmarkTransport => "--transport must be 'in-process' or 'tcp'",
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

        // Include the Zig error name when the mapping is generic so the user
        // can still tell OutOfMemory from InvalidUtf8 etc.
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

    try testing.expectEqual(Subcommand.build, args.subcommand);
    try testing.expectEqualStrings("cfg.json", args.config_path.?);
    try testing.expectEqualStrings("./docs", args.source_dir.?);
    try testing.expect(args.top_k == null);
}

test "parse: query with equals-form flags and custom top-k" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config=cfg.json", "--top-k=42" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.query, args.subcommand);
    try testing.expectEqualStrings("cfg.json", args.config_path.?);
    try testing.expect(args.source_dir == null);
    try testing.expectEqual(@as(usize, 42), args.top_k.?);
}

test "parse: -k short form" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "c.json", "-k", "7" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 7), args.top_k.?);
}

test "parse: later --config overrides earlier and does not leak" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "first.json", "--config=second.json" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqualStrings("second.json", args.config_path.?);
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
    var it: SliceIter = .{ .slice = &.{ "--config", "cfg.json" } };
    try testing.expectError(error.MissingSubcommand, parseFromIter(testing.allocator, &it));
}

test "parse: unknown subcommand" {
    var it: SliceIter = .{ .slice = &.{ "frobnicate", "--config", "cfg.json" } };
    try testing.expectError(error.UnknownSubcommand, parseFromIter(testing.allocator, &it));
}

test "parse: --top-k 0 rejected" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "cfg.json", "--top-k", "0" } };
    try testing.expectError(error.TopKZero, parseFromIter(testing.allocator, &it));
}

test "parse: non-numeric --top-k rejected" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "cfg.json", "--top-k", "abc" } };
    try testing.expectError(error.InvalidTopK, parseFromIter(testing.allocator, &it));
}

test "parse: missing value after --config" {
    var it: SliceIter = .{ .slice = &.{"--config"} };
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

    try testing.expectEqual(Subcommand.benchmark, args.subcommand);
    try testing.expect(args.config_path == null);
    try testing.expect(!args.bench_validate);
    try testing.expect(!args.bench_json);
}

test "parse: benchmark with all knobs set" {
    var it: SliceIter = .{ .slice = &.{
        "benchmark",
        "--num-vectors", "5000",
        "--num-queries=200",
        "--dim",         "64",
        "--ef-construction=150",
        "--ef-search",   "80",
        "--top-k",       "10",
        "--seed=7",
        "--warmup",      "25",
        "--validate",
        "--json",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.benchmark, args.subcommand);
    try testing.expectEqual(@as(usize, 5000), args.bench_num_vectors.?);
    try testing.expectEqual(@as(usize, 200), args.bench_num_queries.?);
    try testing.expectEqual(@as(usize, 64), args.bench_dim.?);
    try testing.expectEqual(@as(usize, 150), args.bench_ef_construction.?);
    try testing.expectEqual(@as(usize, 80), args.bench_ef_search.?);
    try testing.expectEqual(@as(usize, 10), args.top_k.?);
    try testing.expectEqual(@as(u64, 7), args.bench_seed.?);
    try testing.expectEqual(@as(usize, 25), args.bench_warmup.?);
    try testing.expect(args.bench_validate);
    try testing.expect(args.bench_json);
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
        "--config",                "c.json",
        "--listen",                "0.0.0.0:9999",
        "--auto-snapshot-secs=30",
        "--max-connections",       "128",
        "--max-frame-bytes=1048576",
        "--idle-timeout-secs",     "120",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.serve, args.subcommand);
    try testing.expectEqualStrings("c.json", args.config_path.?);
    try testing.expectEqualStrings("0.0.0.0:9999", args.serve_listen.?);
    try testing.expectEqual(@as(u32, 30), args.serve_auto_snapshot_secs.?);
    try testing.expectEqual(@as(u32, 128), args.serve_max_connections.?);
    try testing.expectEqual(@as(usize, 1_048_576), args.serve_max_frame_bytes.?);
    try testing.expectEqual(@as(u32, 120), args.serve_idle_timeout_secs.?);
}

test "parse: serve equals-form --listen" {
    var it: SliceIter = .{ .slice = &.{ "serve", "--config=c.json", "--listen=127.0.0.1:5000" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqualStrings("127.0.0.1:5000", args.serve_listen.?);
}

test "parse: serve rejects non-numeric snapshot-secs" {
    var it: SliceIter = .{ .slice = &.{ "serve", "--auto-snapshot-secs", "foo" } };
    try testing.expectError(error.InvalidServeNumber, parseFromIter(testing.allocator, &it));
}

test "parse: benchmark --transport tcp" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--transport", "tcp" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqual(BenchTransport.tcp, args.bench_transport.?);
    try testing.expect(!args.bench_protocol);
}

test "parse: benchmark --transport=in-process" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--transport=in-process" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expectEqual(BenchTransport.in_process, args.bench_transport.?);
}

test "parse: benchmark --transport rejects unknown" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--transport", "udp" } };
    try testing.expectError(error.InvalidBenchmarkTransport, parseFromIter(testing.allocator, &it));
}

test "parse: benchmark --bench-protocol" {
    var it: SliceIter = .{ .slice = &.{ "benchmark", "--bench-protocol" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expect(args.bench_protocol);
}

test "parse: client ping with explicit connect" {
    var it: SliceIter = .{ .slice = &.{ "client", "--connect", "127.0.0.1:9000", "ping" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.client, args.subcommand);
    try testing.expectEqual(ClientVerb.ping, args.client_verb.?);
    try testing.expectEqualStrings("127.0.0.1:9000", args.client_connect.?);
    try testing.expect(args.client_pos0 == null);
}

test "parse: client delete with id positional" {
    var it: SliceIter = .{ .slice = &.{ "client", "delete", "42" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(ClientVerb.delete, args.client_verb.?);
    try testing.expectEqualStrings("42", args.client_pos0.?);
    try testing.expect(args.client_pos1 == null);
}

test "parse: client replace-text with id + text positionals" {
    var it: SliceIter = .{ .slice = &.{ "client", "replace-text", "7", "new body" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(ClientVerb.replace_text, args.client_verb.?);
    try testing.expectEqualStrings("7", args.client_pos0.?);
    try testing.expectEqualStrings("new body", args.client_pos1.?);
}

test "parse: client insert-vec with --dim and --literal" {
    var it: SliceIter = .{ .slice = &.{
        "client",             "insert-vec",
        "--dim",              "4",
        "--literal",          "1.0,2.0,3.0,4.0",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(ClientVerb.insert_vec, args.client_verb.?);
    try testing.expectEqual(@as(usize, 4), args.client_dim.?);
    try testing.expectEqualStrings("1.0,2.0,3.0,4.0", args.client_literal.?);
}

test "parse: client search-text with top-k and ef" {
    var it: SliceIter = .{ .slice = &.{
        "client",     "search-text", "foo bar",
        "--top-k=10", "--ef=40",
    } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(ClientVerb.search_text, args.client_verb.?);
    try testing.expectEqualStrings("foo bar", args.client_pos0.?);
    try testing.expectEqual(@as(usize, 10), args.top_k.?);
    try testing.expectEqual(@as(u32, 40), args.client_ef.?);
}

test "parse: client --json flag" {
    var it: SliceIter = .{ .slice = &.{ "client", "stats", "--json" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);
    try testing.expect(args.client_json);
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
    try testing.expectEqualStrings("vec.f32", args.client_from_file.?);
    try testing.expect(!args.client_from_stdin);
}
