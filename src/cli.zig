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

pub const Subcommand = enum { build, query, benchmark };

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

    pub fn deinit(self: *Args, allocator: std.mem.Allocator) void {
        if (self.config_path) |p| allocator.free(p);
        if (self.source_dir) |s| allocator.free(s);
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

    errdefer {
        if (config_path) |p| allocator.free(p);
        if (source_dir) |p| allocator.free(p);
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
            bench_dim = parseUsize(p) orelse return error.InvalidBenchmarkNumber;
        } else if (std.mem.startsWith(u8, arg, "--dim=")) {
            bench_dim = parseUsize(arg["--dim=".len..]) orelse return error.InvalidBenchmarkNumber;
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
            bench_json = true;
        } else if (subcommand == null and !std.mem.startsWith(u8, arg, "-")) {
            if (std.mem.eql(u8, arg, "build")) {
                subcommand = .build;
            } else if (std.mem.eql(u8, arg, "query")) {
                subcommand = .query;
            } else if (std.mem.eql(u8, arg, "benchmark")) {
                subcommand = .benchmark;
            } else {
                return error.UnknownSubcommand;
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
    };
}

fn parseUsize(s: []const u8) ?usize {
    return std.fmt.parseInt(usize, s, 10) catch null;
}

/// Emit the usage text to stdout.
pub fn printUsage() !void {
    const usage =
        \\Usage:
        \\  hnswz build      --config <path> --source <dir>
        \\  hnswz query      --config <path> [--top-k <n>]
        \\  hnswz benchmark  [--config <path>] [benchmark flags]
        \\
        \\Subcommands:
        \\  build      Ingest every .txt file in <dir>, embed via Ollama, build
        \\             the HNSW graph, and persist vectors/graph/metadata to
        \\             config.storage.data_dir.
        \\  query      Load a prebuilt index and enter a REPL reading queries
        \\             from stdin (one per line). Ctrl-D or :q exits.
        \\  benchmark  Run an end-to-end build+search workload on synthetic
        \\             vectors and print latency/throughput. Intended as the
        \\             performance regression signal across commits.
        \\
        \\Options:
        \\  --config <path>   Path to JSON config (or set HNSWZ_CONFIG env var).
        \\                    Required for build/query; optional for benchmark
        \\                    (benchmark has its own defaults; if a config is
        \\                    provided, dim/ef_*/seed are inherited from it).
        \\  --source <dir>    (build only) Directory of .txt files to ingest.
        \\  --top-k <n>       (query/benchmark) Results per query. Default 5
        \\                    for query, 10 for benchmark.
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
        error.UnknownSubcommand => "unknown subcommand (expected 'build', 'query', or 'benchmark')",
        error.MissingConfig => "no config specified (use --config <path> or set " ++ CONFIG_ENV_VAR ++ ")",
        error.MissingSubcommand => "missing subcommand (build | query | benchmark)",
        error.TopKZero => "--top-k must be > 0",
        error.InvalidTopK => "--top-k must be a non-negative integer",
        error.InvalidBenchmarkNumber => "benchmark integer flag could not be parsed",
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
            => log.err("{s}", .{describeParseError(err)}),
            error.HelpRequested => unreachable,
            else => log.err("CLI parse failed: {s}", .{@errorName(err)}),
        }
        printUsage() catch {};
        std.process.exit(USAGE_EXIT_CODE);
    };
}

// ── tests ──────────────────────────────────────────────────────────────

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
