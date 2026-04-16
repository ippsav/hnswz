const std = @import("std");
const hnswz = @import("hnswz");

// Comptime tuning knob. Changing M requires a recompile. Everything else
// (dim, ef_*, max_vectors, model, URL, …) comes from config.json.
const M: usize = 16;
const DEFAULT_TOP_K: usize = 5;

const log = std.log.scoped(.main);

const Subcommand = enum { build, query };

const CliArgs = struct {
    subcommand: Subcommand,
    config_path: []u8,
    source_dir: ?[]u8 = null, // build only
    top_k: usize = DEFAULT_TOP_K, // query only
};

fn parseCli(allocator: std.mem.Allocator) !CliArgs {
    var args = std.process.args();
    _ = args.next();

    var subcommand: ?Subcommand = null;
    var config_path: ?[]u8 = null;
    var source_dir: ?[]u8 = null;
    var top_k: usize = DEFAULT_TOP_K;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--config")) {
            const p = args.next() orelse return error.MissingValue;
            config_path = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--config=")) {
            config_path = try allocator.dupe(u8, arg["--config=".len..]);
        } else if (std.mem.eql(u8, arg, "--source")) {
            const p = args.next() orelse return error.MissingValue;
            source_dir = try allocator.dupe(u8, p);
        } else if (std.mem.startsWith(u8, arg, "--source=")) {
            source_dir = try allocator.dupe(u8, arg["--source=".len..]);
        } else if (std.mem.eql(u8, arg, "-k") or std.mem.eql(u8, arg, "--top-k")) {
            const p = args.next() orelse return error.MissingValue;
            top_k = try std.fmt.parseInt(usize, p, 10);
        } else if (std.mem.startsWith(u8, arg, "--top-k=")) {
            top_k = try std.fmt.parseInt(usize, arg["--top-k=".len..], 10);
        } else if (subcommand == null and !std.mem.startsWith(u8, arg, "-")) {
            if (std.mem.eql(u8, arg, "build")) {
                subcommand = .build;
            } else if (std.mem.eql(u8, arg, "query")) {
                subcommand = .query;
            } else {
                return error.UnknownSubcommand;
            }
        }
    }

    // Fall back to HNSWZ_CONFIG env var if --config not provided.
    if (config_path == null) {
        config_path = std.process.getEnvVarOwned(allocator, "HNSWZ_CONFIG") catch |err| switch (err) {
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

fn printUsage() !void {
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

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cli = parseCli(allocator) catch |err| {
        switch (err) {
            error.MissingConfig => log.err("no config specified (use --config <path> or HNSWZ_CONFIG)", .{}),
            error.MissingSubcommand => log.err("missing subcommand (build | query)", .{}),
            error.MissingValue => log.err("flag requires a value", .{}),
            error.UnknownSubcommand => log.err("unknown subcommand", .{}),
            error.TopKZero => log.err("--top-k must be > 0", .{}),
            else => log.err("CLI parse failed: {s}", .{@errorName(err)}),
        }
        try printUsage();
        std.process.exit(2);
    };
    defer allocator.free(cli.config_path);
    defer if (cli.source_dir) |s| allocator.free(s);

    var loaded = hnswz.config.loadFromPath(allocator, cli.config_path) catch |err| {
        switch (err) {
            error.FileOpenFailed => log.err("cannot open config file: {s}", .{cli.config_path}),
            error.FileReadFailed => log.err("cannot read config file: {s}", .{cli.config_path}),
            error.ParseFailed => log.err("config JSON parse failed: {s}", .{cli.config_path}),
            else => log.err("config load/validation failed: {s}", .{@errorName(err)}),
        }
        std.process.exit(2);
    };
    defer loaded.deinit();
    const cfg = &loaded.config;

    log.info("config loaded: model={s} dim={d} max_vectors={d}", .{
        cfg.embedder.model, cfg.embedder.dim, cfg.storage.max_vectors,
    });

    switch (cli.subcommand) {
        .build => {
            const src = cli.source_dir orelse {
                log.err("build requires --source <dir>", .{});
                std.process.exit(2);
            };
            try runBuild(allocator, cfg, src);
        },
        .query => try runQuery(allocator, cfg, cli.top_k),
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
    try index.save(data_dir, cfg.storage.graph_file);
    try hnswz.metadata.save(data_dir, cfg.storage.metadata_file, filenames.items);

    log.info("index built: {d} vectors saved to {s}", .{ n, cfg.storage.data_dir });
}

// ── query ─────────────────────────────────────────────────────────────

fn runQuery(allocator: std.mem.Allocator, cfg: *const hnswz.config.Config, top_k: usize) !void {
    if (top_k > cfg.index.ef_search) {
        log.err("--top-k ({d}) must be <= config.index.ef_search ({d})", .{ top_k, cfg.index.ef_search });
        std.process.exit(2);
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

    if (md.count != store.count) {
        log.err("metadata count ({d}) != vectors count ({d})", .{ md.count, store.count });
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
