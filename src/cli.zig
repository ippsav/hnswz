const std = @import("std");

const log = std.log.scoped(.cli);

/// Default value for `--top-k` when the flag is omitted.
pub const DEFAULT_TOP_K: usize = 5;

/// Environment variable consulted when `--config` is not provided on the
/// command line.
pub const CONFIG_ENV_VAR = "HNSWZ_CONFIG";

/// Exit code used for any CLI / configuration usage error. Matches common
/// convention (2 = misuse of shell builtin / bad invocation).
pub const USAGE_EXIT_CODE: u8 = 2;

pub const Subcommand = enum { build, query };

/// Parsed command-line arguments. Owns the heap allocations referenced by
/// `config_path` and `source_dir`; call `deinit` once the value is no
/// longer needed.
pub const Args = struct {
    subcommand: Subcommand,
    config_path: []u8,
    source_dir: ?[]u8 = null, // build only
    top_k: usize = DEFAULT_TOP_K, // query only

    pub fn deinit(self: *Args, allocator: std.mem.Allocator) void {
        allocator.free(self.config_path);
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
    var top_k: usize = DEFAULT_TOP_K;

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
        } else if (subcommand == null and !std.mem.startsWith(u8, arg, "-")) {
            if (std.mem.eql(u8, arg, "build")) {
                subcommand = .build;
            } else if (std.mem.eql(u8, arg, "query")) {
                subcommand = .query;
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
    if (config_path == null) return error.MissingConfig;
    if (subcommand == null) return error.MissingSubcommand;
    if (top_k == 0) return error.TopKZero;

    return .{
        .subcommand = subcommand.?,
        .config_path = config_path.?,
        .source_dir = source_dir,
        .top_k = top_k,
    };
}

/// Emit the usage text to stdout.
pub fn printUsage() !void {
    const usage =
        \\Usage:
        \\  hnswz build  --config <path> --source <dir>
        \\  hnswz query  --config <path> [--top-k <n>]
        \\
        \\Subcommands:
        \\  build  Ingest every .txt file in <dir>, embed via Ollama, build the
        \\         HNSW graph, and persist vectors/graph/metadata to
        \\         config.storage.data_dir.
        \\  query  Load a prebuilt index and enter a REPL reading queries from
        \\         stdin (one per line). Ctrl-D or :q exits.
        \\
        \\Options:
        \\  --config <path>  Path to JSON config (or set HNSWZ_CONFIG env var).
        \\  --source <dir>   (build only) Directory of .txt files to ingest.
        \\  --top-k <n>      (query only) Number of results per query. Default 5.
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
        error.UnknownSubcommand => "unknown subcommand (expected 'build' or 'query')",
        error.MissingConfig => "no config specified (use --config <path> or set " ++ CONFIG_ENV_VAR ++ ")",
        error.MissingSubcommand => "missing subcommand (build | query)",
        error.TopKZero => "--top-k must be > 0",
        error.InvalidTopK => "--top-k must be a non-negative integer",
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
    try testing.expectEqualStrings("cfg.json", args.config_path);
    try testing.expectEqualStrings("./docs", args.source_dir.?);
    try testing.expectEqual(@as(usize, DEFAULT_TOP_K), args.top_k);
}

test "parse: query with equals-form flags and custom top-k" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config=cfg.json", "--top-k=42" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(Subcommand.query, args.subcommand);
    try testing.expectEqualStrings("cfg.json", args.config_path);
    try testing.expect(args.source_dir == null);
    try testing.expectEqual(@as(usize, 42), args.top_k);
}

test "parse: -k short form" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "c.json", "-k", "7" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 7), args.top_k);
}

test "parse: later --config overrides earlier and does not leak" {
    var it: SliceIter = .{ .slice = &.{ "query", "--config", "first.json", "--config=second.json" } };
    var args = try parseFromIter(testing.allocator, &it);
    defer args.deinit(testing.allocator);

    try testing.expectEqualStrings("second.json", args.config_path);
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
