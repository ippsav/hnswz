//! Runtime configuration: parsed from a JSON file at startup.
//!
//! Every runtime-tunable knob lives here. After load+validate, nothing
//! mutates these values. Preallocation sizes in `main` are computed from
//! this struct and only this struct.
const std = @import("std");

pub const Embedder = struct {
    provider: []const u8 = "ollama",
    base_url: []const u8 = "http://localhost:11434",
    model: []const u8 = "qwen3-embedding",
    dim: usize,
    normalize: bool = false,
    request_timeout_ms: u32 = 30_000,
    max_text_bytes: usize = 65_536,
};

pub const Distance = enum {
    cosine,
    dot,
};

pub const Index = struct {
    ef_construction: usize = 200,
    ef_search: usize = 100,
    max_ef: usize = 200,
    seed: u64 = 42,
    distance: Distance = .cosine,
};

pub const Storage = struct {
    data_dir: []const u8,
    max_vectors: usize,
    upper_pool_slots: usize = 0, // 0 means "derive at load time"
    vectors_file: []const u8 = "vectors.hvsf",
    graph_file: []const u8 = "graph.hgrf",
    metadata_file: []const u8 = "metadata.hmtf",
};

pub const LogLevel = enum {
    debug,
    info,
    warn,
    @"error",
};

pub const Config = struct {
    embedder: Embedder,
    index: Index = .{},
    storage: Storage,
    log_level: LogLevel = .info,
};

pub const ValidationError = error{
    EmbedderDimZero,
    EmbedderMaxTextBytesZero,
    IndexEfZero,
    IndexMaxEfTooSmall,
    StorageMaxVectorsZero,
    StorageDataDirEmpty,
    UnsupportedProvider,
};

pub const Error = error{
    FileOpenFailed,
    FileReadFailed,
    ParseFailed,
    OutOfMemory,
} || ValidationError;

/// Return a human-readable explanation for a validation error. Intended for
/// CLI/startup diagnostics; not for structured logs.
pub fn describeValidationError(err: ValidationError, c: Config, buf: []u8) []const u8 {
    return switch (err) {
        error.EmbedderDimZero => std.fmt.bufPrint(buf, "embedder.dim must be > 0", .{}) catch buf[0..0],
        error.EmbedderMaxTextBytesZero => std.fmt.bufPrint(buf, "embedder.max_text_bytes must be > 0", .{}) catch buf[0..0],
        error.IndexEfZero => std.fmt.bufPrint(buf, "index.ef_construction and ef_search must be > 0", .{}) catch buf[0..0],
        error.IndexMaxEfTooSmall => std.fmt.bufPrint(
            buf,
            "index.max_ef ({d}) must be >= max(ef_construction={d}, ef_search={d})",
            .{ c.index.max_ef, c.index.ef_construction, c.index.ef_search },
        ) catch buf[0..0],
        error.StorageMaxVectorsZero => std.fmt.bufPrint(buf, "storage.max_vectors must be > 0", .{}) catch buf[0..0],
        error.StorageDataDirEmpty => std.fmt.bufPrint(buf, "storage.data_dir must be set", .{}) catch buf[0..0],
        error.UnsupportedProvider => std.fmt.bufPrint(
            buf,
            "only embedder.provider=\"ollama\" is supported (got \"{s}\")",
            .{c.embedder.provider},
        ) catch buf[0..0],
    };
}

/// Owns the parsed config and its backing JSON memory.
/// Call `deinit` to free everything.
pub const Loaded = struct {
    arena: std.heap.ArenaAllocator,
    config: Config,

    pub fn deinit(self: *Loaded) void {
        self.arena.deinit();
    }
};

/// Load and validate a config from disk. All string fields in the returned
/// `Loaded.config` are owned by the internal arena.
pub fn loadFromPath(gpa: std.mem.Allocator, path: []const u8) Error!Loaded {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const aa = arena.allocator();

    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileOpenFailed;
    defer file.close();

    const bytes = file.readToEndAlloc(aa, 16 * 1024 * 1024) catch return error.FileReadFailed;

    // Keep ownership of parsed strings in the arena by using parseFromSliceLeaky.
    const config = std.json.parseFromSliceLeaky(Config, aa, bytes, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    }) catch return error.ParseFailed;

    try validate(config);

    return .{ .arena = arena, .config = config };
}

fn validate(c: Config) ValidationError!void {
    if (c.embedder.dim == 0) return error.EmbedderDimZero;
    if (c.embedder.max_text_bytes == 0) return error.EmbedderMaxTextBytesZero;
    if (c.index.ef_construction == 0 or c.index.ef_search == 0) return error.IndexEfZero;
    if (c.index.max_ef < @max(c.index.ef_construction, c.index.ef_search)) return error.IndexMaxEfTooSmall;
    if (c.storage.max_vectors == 0) return error.StorageMaxVectorsZero;
    if (c.storage.data_dir.len == 0) return error.StorageDataDirEmpty;
    if (!std.mem.eql(u8, c.embedder.provider, "ollama")) return error.UnsupportedProvider;
}



const testing = std.testing;

test "loadFromPath parses a minimal valid config" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const body =
        \\{
        \\  "embedder": { "dim": 4096 },
        \\  "storage":  { "data_dir": "./data", "max_vectors": 1000 }
        \\}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "c.json", .data = body });

    // Build absolute path for loadFromPath (which uses cwd).
    const real = try tmp.dir.realpathAlloc(testing.allocator, "c.json");
    defer testing.allocator.free(real);

    var loaded = try loadFromPath(testing.allocator, real);
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 4096), loaded.config.embedder.dim);
    try testing.expectEqualStrings("./data", loaded.config.storage.data_dir);
    try testing.expectEqual(@as(usize, 1000), loaded.config.storage.max_vectors);
    // Defaults.
    try testing.expectEqualStrings("qwen3-embedding", loaded.config.embedder.model);
    try testing.expectEqual(Distance.cosine, loaded.config.index.distance);
}

test "loadFromPath rejects max_ef < ef_search" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const body =
        \\{
        \\  "embedder": { "dim": 4096 },
        \\  "index":    { "ef_search": 500, "max_ef": 100 },
        \\  "storage":  { "data_dir": "./data", "max_vectors": 10 }
        \\}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "c.json", .data = body });
    const real = try tmp.dir.realpathAlloc(testing.allocator, "c.json");
    defer testing.allocator.free(real);

    try testing.expectError(error.IndexMaxEfTooSmall, loadFromPath(testing.allocator, real));
}

test "loadFromPath rejects dim == 0" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const body =
        \\{
        \\  "embedder": { "dim": 0 },
        \\  "storage":  { "data_dir": "./data", "max_vectors": 10 }
        \\}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "c.json", .data = body });
    const real = try tmp.dir.realpathAlloc(testing.allocator, "c.json");
    defer testing.allocator.free(real);

    try testing.expectError(error.EmbedderDimZero, loadFromPath(testing.allocator, real));
}

test "loadFromPath parses custom distance enum" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const body =
        \\{
        \\  "embedder": { "dim": 128 },
        \\  "index":    { "distance": "dot" },
        \\  "storage":  { "data_dir": "./d", "max_vectors": 1 }
        \\}
    ;
    try tmp.dir.writeFile(.{ .sub_path = "c.json", .data = body });
    const real = try tmp.dir.realpathAlloc(testing.allocator, "c.json");
    defer testing.allocator.free(real);

    var loaded = try loadFromPath(testing.allocator, real);
    defer loaded.deinit();

    try testing.expectEqual(Distance.dot, loaded.config.index.distance);
}
