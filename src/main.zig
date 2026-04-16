const std = @import("std");
const hnswz = @import("hnswz");
const cli = @import("cli.zig");

// Comptime tuning knob. Changing M requires a recompile. Everything else
// (dim, ef_*, max_vectors, model, URL, …) comes from config.json.
const M: usize = 16;

const log = std.log.scoped(.main);

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = cli.parseOrExit(allocator);
    defer args.deinit(allocator);

    switch (args.subcommand) {
        .build, .query => {
            const path = args.config_path orelse {
                log.err("{s} requires --config <path> (or set {s})", .{
                    @tagName(args.subcommand), cli.CONFIG_ENV_VAR,
                });
                std.process.exit(cli.USAGE_EXIT_CODE);
            };

            var loaded = hnswz.config.loadFromPath(allocator, path) catch |err| {
                switch (err) {
                    error.FileOpenFailed => log.err("cannot open config file: {s}", .{path}),
                    error.FileReadFailed => log.err("cannot read config file: {s}", .{path}),
                    error.ParseFailed => log.err("config JSON parse failed: {s}", .{path}),
                    else => log.err("config load/validation failed: {s}", .{@errorName(err)}),
                }
                std.process.exit(cli.USAGE_EXIT_CODE);
            };
            defer loaded.deinit();
            const cfg = &loaded.config;

            log.info("config loaded: model={s} dim={d} max_vectors={d}", .{
                cfg.embedder.model, cfg.embedder.dim, cfg.storage.max_vectors,
            });

            switch (args.subcommand) {
                .build => {
                    const src = args.source_dir orelse {
                        log.err("build requires --source <dir>", .{});
                        std.process.exit(cli.USAGE_EXIT_CODE);
                    };
                    try runBuild(allocator, cfg, src);
                },
                .query => {
                    const top_k = args.top_k orelse cli.DEFAULT_TOP_K;
                    try runQuery(allocator, cfg, top_k);
                },
                .benchmark => unreachable,
            }
        },
        .benchmark => {
            // Config is optional for benchmark. Load only if --config or the
            // HNSWZ_CONFIG env var supplied a path; otherwise fall back to
            // benchmark defaults.
            var cfg_loaded: ?hnswz.config.Loaded = null;
            defer if (cfg_loaded) |*l| l.deinit();

            if (args.config_path) |path| {
                cfg_loaded = hnswz.config.loadFromPath(allocator, path) catch |err| {
                    switch (err) {
                        error.FileOpenFailed => log.err("cannot open config file: {s}", .{path}),
                        error.FileReadFailed => log.err("cannot read config file: {s}", .{path}),
                        error.ParseFailed => log.err("config JSON parse failed: {s}", .{path}),
                        else => log.err("config load/validation failed: {s}", .{@errorName(err)}),
                    }
                    std.process.exit(cli.USAGE_EXIT_CODE);
                };
                log.info("config loaded (benchmark defaults derive from it)", .{});
            }

            const cfg_opt: ?*const hnswz.config.Config =
                if (cfg_loaded) |*l| &l.config else null;
            try runBenchmark(allocator, cfg_opt, &args);
        },
    }
}

// ── build ─────────────────────────────────────────────────────────────

fn strLessThan(_: void, a: []u8, b: []u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

fn runBuild(allocator: std.mem.Allocator, cfg: *const hnswz.config.Config, source: []const u8) !void {
    var src_dir = std.fs.cwd().openDir(source, .{ .iterate = true }) catch |err| {
        log.err("cannot open --source '{s}': {s}", .{ source, @errorName(err) });
        std.process.exit(1);
    };
    defer src_dir.close();

    std.fs.cwd().makePath(cfg.storage.data_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => {
            log.err("cannot create data_dir '{s}': {s}", .{ cfg.storage.data_dir, @errorName(err) });
            std.process.exit(1);
        },
    };
    var data_dir = try std.fs.cwd().openDir(cfg.storage.data_dir, .{});
    defer data_dir.close();

    // Collect & sort filenames.
    var filenames: std.ArrayList([]u8) = .{};
    defer {
        for (filenames.items) |n| allocator.free(n);
        filenames.deinit(allocator);
    }
    var it = src_dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".txt")) continue;
        try filenames.append(allocator, try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]u8, filenames.items, {}, strLessThan);

    const n = filenames.items.len;
    if (n == 0) {
        log.err("no .txt files found in {s}", .{source});
        std.process.exit(1);
    }
    if (n > cfg.storage.max_vectors) {
        log.err("{d} files exceeds config.storage.max_vectors={d}", .{ n, cfg.storage.max_vectors });
        std.process.exit(1);
    }
    log.info("ingesting {d} files from {s}", .{ n, source });

    // Embedder.
    var client = try hnswz.OllamaClient.init(
        allocator,
        cfg.embedder.model,
        cfg.embedder.base_url,
        .{ .max_text_bytes = cfg.embedder.max_text_bytes, .dim = cfg.embedder.dim },
    );
    defer client.deinit();
    const embedder = client.embedder();

    // Store sized for max_vectors.
    var store = try hnswz.Store.init(allocator, cfg.embedder.dim, cfg.storage.max_vectors);
    defer store.deinit(allocator);

    const Index = hnswz.HnswIndex(M);
    var index = try Index.init(allocator, &store, .{
        .max_vectors = cfg.storage.max_vectors,
        .max_upper_slots = cfg.storage.upper_pool_slots,
        .ef_construction = cfg.index.ef_construction,
        .seed = cfg.index.seed,
    });
    defer index.deinit();

    var ws = try Index.Workspace.init(allocator, cfg.storage.max_vectors, cfg.index.max_ef);
    defer ws.deinit(allocator);

    const vec_buf = try allocator.alloc(f32, cfg.embedder.dim);
    defer allocator.free(vec_buf);

    for (filenames.items, 0..) |name, i| {
        const file = try src_dir.openFile(name, .{});
        defer file.close();
        const content = try file.readToEndAlloc(allocator, cfg.embedder.max_text_bytes);
        defer allocator.free(content);

        try embedder.embed(content, vec_buf);
        if (cfg.embedder.normalize) hnswz.bruteforce.normalize(vec_buf);

        const id = try store.add(vec_buf);
        try index.insert(&ws, id);

        std.debug.print("\rEmbedding+insert [{d}/{d}]", .{ i + 1, n });
    }
    std.debug.print("\n", .{});

    try store.save(data_dir, cfg.storage.vectors_file);
    try index.save(data_dir, cfg.storage.graph_file, null);
    try hnswz.metadata.save(data_dir, cfg.storage.metadata_file, filenames.items);

    log.info("index built: {d} vectors saved to {s}", .{ n, cfg.storage.data_dir });
}

// ── query ─────────────────────────────────────────────────────────────

fn runQuery(allocator: std.mem.Allocator, cfg: *const hnswz.config.Config, top_k: usize) !void {
    if (top_k > cfg.index.ef_search) {
        log.err("--top-k ({d}) must be <= config.index.ef_search ({d})", .{ top_k, cfg.index.ef_search });
        std.process.exit(cli.USAGE_EXIT_CODE);
    }

    var data_dir = std.fs.cwd().openDir(cfg.storage.data_dir, .{}) catch |err| {
        log.err("cannot open storage.data_dir '{s}': {s}", .{ cfg.storage.data_dir, @errorName(err) });
        std.process.exit(1);
    };
    defer data_dir.close();

    var store = hnswz.Store.load(allocator, data_dir, cfg.storage.vectors_file, cfg.storage.max_vectors) catch |err| {
        log.err("cannot load vectors file '{s}/{s}': {s}", .{
            cfg.storage.data_dir, cfg.storage.vectors_file, @errorName(err),
        });
        std.process.exit(1);
    };
    defer store.deinit(allocator);

    if (store.dim != cfg.embedder.dim) {
        log.err("vectors file dim ({d}) != config embedder.dim ({d})", .{ store.dim, cfg.embedder.dim });
        std.process.exit(1);
    }

    const Index = hnswz.HnswIndex(M);

    const file_sz = Index.scanSize(data_dir, cfg.storage.graph_file) catch |err| {
        log.err("cannot scan graph file '{s}/{s}': {s}", .{
            cfg.storage.data_dir, cfg.storage.graph_file, @errorName(err),
        });
        std.process.exit(1);
    };

    const upper_slots = @max(cfg.storage.upper_pool_slots, file_sz.upper_slots);

    var index = Index.load(allocator, &store, data_dir, cfg.storage.graph_file, .{
        .max_vectors = cfg.storage.max_vectors,
        .max_upper_slots = upper_slots,
        .ef_construction = cfg.index.ef_construction,
        .seed = cfg.index.seed,
    }) catch |err| {
        log.err("graph load failed: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer index.deinit();

    var md = hnswz.metadata.load(allocator, data_dir, cfg.storage.metadata_file) catch |err| {
        log.err("cannot load metadata file '{s}/{s}': {s}", .{
            cfg.storage.data_dir, cfg.storage.metadata_file, @errorName(err),
        });
        std.process.exit(1);
    };
    defer md.deinit(allocator);

    if (md.count != store.live_count) {
        log.err("metadata count ({d}) != vectors count ({d})", .{ md.count, store.live_count });
        std.process.exit(1);
    }

    log.info("index loaded: {d} vectors, dim={d}, upper_slots={d}/{d}", .{
        index.node_count, store.dim, index.upper_used, index.max_upper_slots,
    });

    var client = try hnswz.OllamaClient.init(
        allocator,
        cfg.embedder.model,
        cfg.embedder.base_url,
        .{ .max_text_bytes = cfg.embedder.max_text_bytes, .dim = cfg.embedder.dim },
    );
    defer client.deinit();
    const embedder = client.embedder();

    var ws = try Index.Workspace.init(allocator, cfg.storage.max_vectors, cfg.index.max_ef);
    defer ws.deinit(allocator);

    const qvec = try allocator.alloc(f32, cfg.embedder.dim);
    defer allocator.free(qvec);

    const results_buf = try allocator.alloc(hnswz.heap.Entry, top_k);
    defer allocator.free(results_buf);

    // Line buffer is sized to cfg.embedder.max_text_bytes so any query that
    // the embedder could accept also fits in one line. Allocated once at
    // startup — no per-query allocation.
    const line_buf = try allocator.alloc(u8, cfg.embedder.max_text_bytes);
    defer allocator.free(line_buf);

    var stdin_file_reader = std.fs.File.stdin().readerStreaming(line_buf);
    const in = &stdin_file_reader.interface;

    var out_buf: [8192]u8 = undefined;
    var stdout_file_writer = std.fs.File.stdout().writer(&out_buf);
    const out = &stdout_file_writer.interface;

    try out.writeAll("hnswz query REPL — type a query and press Enter. Ctrl-D or :q to exit.\n");
    try out.flush();

    while (true) {
        try out.writeAll("> ");
        try out.flush();

        const raw = in.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                try out.print("query too long (max {d} bytes); try a shorter query\n", .{line_buf.len});
                // Drain the rest of the oversized line so the next prompt is clean.
                _ = in.discardDelimiterInclusive('\n') catch {};
                try out.flush();
                continue;
            },
            error.ReadFailed => {
                log.err("stdin read failed", .{});
                return;
            },
        };
        const line_opt = raw orelse {
            try out.writeAll("\n");
            try out.flush();
            break; // EOF
        };
        const line = std.mem.trim(u8, line_opt, " \t\r\n");
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, ":q") or std.mem.eql(u8, line, ":quit") or std.mem.eql(u8, line, ":exit")) {
            break;
        }

        embedder.embed(line, qvec) catch |err| {
            try out.print("embed failed: {s}\n", .{@errorName(err)});
            try out.flush();
            continue;
        };
        if (cfg.embedder.normalize) hnswz.bruteforce.normalize(qvec);

        const results = index.search(&ws, qvec, top_k, cfg.index.ef_search, results_buf) catch |err| {
            try out.print("search failed: {s}\n", .{@errorName(err)});
            try out.flush();
            continue;
        };

        for (results) |r| {
            try out.print("{s}\t{d:.6}\n", .{ md.get(r.id), r.dist });
        }
        try out.flush();
    }
}

// ── benchmark ─────────────────────────────────────────────────────────

/// Hardcoded benchmark defaults. These apply when neither CLI flag nor
/// config sets the value. Chosen so `zig build benchmark` with zero args
/// finishes in a few seconds on a laptop while still producing a
/// reasonable recall signal.
const BENCH_DEFAULTS = struct {
    const num_vectors: usize = 10_000;
    const num_queries: usize = 1_000;
    const dim: usize = 128;
    const ef_construction: usize = 200;
    const ef_search: usize = 100;
    const seed: u64 = 42;
    const warmup: usize = 50;
};

fn runBenchmark(
    allocator: std.mem.Allocator,
    cfg_opt: ?*const hnswz.config.Config,
    args: *const cli.Args,
) !void {
    // Resolve each knob: CLI > config > hardcoded default.
    const dim: usize = blk: {
        if (args.bench_dim) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.embedder.dim;
        break :blk BENCH_DEFAULTS.dim;
    };
    const ef_construction: usize = blk: {
        if (args.bench_ef_construction) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.index.ef_construction;
        break :blk BENCH_DEFAULTS.ef_construction;
    };
    const ef_search: usize = blk: {
        if (args.bench_ef_search) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.index.ef_search;
        break :blk BENCH_DEFAULTS.ef_search;
    };
    const seed: u64 = blk: {
        if (args.bench_seed) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.index.seed;
        break :blk BENCH_DEFAULTS.seed;
    };
    const top_k: usize = args.top_k orelse cli.DEFAULT_BENCH_TOP_K;
    const num_vectors: usize = args.bench_num_vectors orelse BENCH_DEFAULTS.num_vectors;
    const num_queries: usize = args.bench_num_queries orelse BENCH_DEFAULTS.num_queries;
    const warmup: usize = args.bench_warmup orelse BENCH_DEFAULTS.warmup;

    // Semantic validation — caught here so CLI users get a clean message
    // instead of a panicking assert inside the benchmark loop.
    if (num_vectors == 0) {
        log.err("--num-vectors must be > 0", .{});
        std.process.exit(cli.USAGE_EXIT_CODE);
    }
    if (num_queries == 0) {
        log.err("--num-queries must be > 0", .{});
        std.process.exit(cli.USAGE_EXIT_CODE);
    }
    if (dim == 0) {
        log.err("--dim must be > 0", .{});
        std.process.exit(cli.USAGE_EXIT_CODE);
    }
    if (ef_construction == 0 or ef_search == 0) {
        log.err("--ef-construction and --ef-search must be > 0", .{});
        std.process.exit(cli.USAGE_EXIT_CODE);
    }
    if (top_k > ef_search) {
        log.err("--top-k ({d}) must be <= --ef-search ({d})", .{ top_k, ef_search });
        std.process.exit(cli.USAGE_EXIT_CODE);
    }
    if (num_vectors < top_k) {
        log.err("--num-vectors ({d}) must be >= --top-k ({d})", .{ num_vectors, top_k });
        std.process.exit(cli.USAGE_EXIT_CODE);
    }

    const max_ef = @max(ef_construction, ef_search);

    const opts: hnswz.benchmark.Options = .{
        .num_vectors = num_vectors,
        .num_queries = num_queries,
        .dim = dim,
        .ef_construction = ef_construction,
        .ef_search = ef_search,
        .max_ef = max_ef,
        .top_k = top_k,
        .seed = seed,
        .warmup = warmup,
        .validate = args.bench_validate,
        .json = args.bench_json,
        // Generous default that matches the per-node slot budget used in
        // the hnsw tests. Expected mean is num_vectors / (M-1).
        .upper_pool_slots = num_vectors,
    };

    var out_buf: [8192]u8 = undefined;
    var stdout_file_writer = std.fs.File.stdout().writer(&out_buf);
    const out = &stdout_file_writer.interface;

    try hnswz.benchmark.run(allocator, opts, out);
}

// ── tests ──────────────────────────────────────────────────────────────

test {
    // Ensure cli.zig's tests get compiled when running the exe test binary.
    _ = @import("cli.zig");
}
