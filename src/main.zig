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

    switch (args.command) {
        .client => |*c| try runClient(allocator, c),
        .query => |*q| try runQuery(allocator, q),
        .build, .serve => {
            const path = args.config_path orelse {
                log.err("{s} requires --config <path> (or set {s})", .{
                    @tagName(std.meta.activeTag(args.command)), cli.CONFIG_ENV_VAR,
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

            switch (args.command) {
                .build => |*b| {
                    const src = b.source_dir orelse {
                        log.err("build requires --source <dir>", .{});
                        std.process.exit(cli.USAGE_EXIT_CODE);
                    };
                    try runBuild(allocator, cfg, src);
                },
                .serve => |*s| try runServe(allocator, cfg, s),
                .benchmark, .client, .query => unreachable,
            }
        },
        .benchmark => |*be| {
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
            try runBenchmark(allocator, cfg_opt, be);
        },
    }
}

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

    // Refuse to run concurrently with another hnswz process touching the
    // same data_dir. flock(2)-based advisory lock; released on process
    // exit (incl. kill -9) so a stale file never strands the dir.
    var lock = hnswz.lockfile.LockFile.acquire(data_dir, cfg.storage.lock_file) catch |err| switch (err) {
        error.AlreadyLocked => {
            log.err("data_dir '{s}' is already locked by another hnswz process", .{cfg.storage.data_dir});
            std.process.exit(1);
        },
        else => {
            log.err("cannot acquire lock on '{s}/{s}': {s}", .{ cfg.storage.data_dir, cfg.storage.lock_file, @errorName(err) });
            std.process.exit(1);
        },
    };
    defer lock.release();

    // `build` produces a fresh snapshot from scratch. If a stale WAL
    // exists from a previous `serve` session, drop it — its records
    // reference the old (possibly incompatible) vectors.
    data_dir.deleteFile(cfg.storage.wal_file) catch |err| switch (err) {
        error.FileNotFound => {},
        else => {
            log.warn("could not delete stale WAL '{s}/{s}': {s}", .{ cfg.storage.data_dir, cfg.storage.wal_file, @errorName(err) });
        },
    };

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

    const slots = try allocator.alloc(?[]const u8, filenames.items.len);
    defer allocator.free(slots);
    for (filenames.items, 0..) |name, i| slots[i] = name;
    try hnswz.metadata.save(data_dir, cfg.storage.metadata_file, slots, filenames.items.len);

    log.info("index built: {d} vectors saved to {s}", .{ n, cfg.storage.data_dir });
}

/// REPL over a running `hnswz serve`. Bare lines run SEARCH_TEXT; colon
/// commands invoke the rest of the server verbs.
fn runQuery(allocator: std.mem.Allocator, args: *const cli.QueryArgs) !void {
    const connect = args.connect orelse "127.0.0.1:9000";
    const parts = splitHostPort(connect) orelse {
        log.err("--connect must be host:port (got '{s}')", .{connect});
        std.process.exit(cli.USAGE_EXIT_CODE);
    };
    const addr = std.net.Address.parseIp(parts.host, parts.port) catch |err| {
        log.err("invalid connect address '{s}': {s}", .{ connect, @errorName(err) });
        std.process.exit(cli.USAGE_EXIT_CODE);
    };

    var client = hnswz.client.Client.connect(allocator, addr, .{
        .recv_buf_size = 4 * 1024 * 1024,
        .send_buf_size = 4 * 1024 * 1024,
    }) catch |err| {
        log.err("cannot connect to {s}: {s}", .{ connect, @errorName(err) });
        std.process.exit(1);
    };
    defer client.deinit();

    const top_k: u16 = @intCast(args.top_k orelse cli.DEFAULT_TOP_K);
    const ef: u16 = if (args.ef) |e| @intCast(e) else @max(top_k, 10);

    // One STATS up front: it verifies the server speaks our protocol and
    // hands back `dim`, which :get needs to size its vec buffer (the
    // response payload has no length prefix for the vec).
    const initial_stats = client.stats() catch |err| {
        log.err("initial stats failed: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    const dim = initial_stats.dim;

    // Protocol caps text_len at u16 = 65535 anyway; pick a line buffer
    // that fits any text the server could accept. The server may enforce
    // a stricter `max_text_bytes` from its config and reject with
    // text_too_long; we surface that verbatim.
    const LINE_BUF_BYTES: usize = 65 * 1024;
    const line_buf = try allocator.alloc(u8, LINE_BUF_BYTES);
    defer allocator.free(line_buf);

    var stdin_file_reader = std.fs.File.stdin().readerStreaming(line_buf);
    const in = &stdin_file_reader.interface;

    var out_buf: [8192]u8 = undefined;
    var stdout_file_writer = std.fs.File.stdout().writer(&out_buf);
    const out = &stdout_file_writer.interface;

    try out.print(
        "hnswz query — connected to {s} (dim={d}, live={d}). :help for commands, :q to exit.\n",
        .{ connect, dim, initial_stats.live_count },
    );
    try out.flush();

    while (true) {
        try out.writeAll("> ");
        try out.flush();

        const raw = in.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                try out.print("line too long (max {d} bytes)\n", .{line_buf.len});
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

        const action = dispatchQueryReplLine(allocator, &client, line, dim, top_k, ef, out) catch |err| {
            if (err == error.ServerError) {
                try replPrintServerError(&client, out);
                try out.flush();
                continue;
            }
            log.err("connection error: {s}", .{@errorName(err)});
            return err;
        };
        try out.flush();
        if (action == .exit) break;
    }
}

const ReplAction = enum { continue_, exit };

fn dispatchQueryReplLine(
    allocator: std.mem.Allocator,
    client: *hnswz.client.Client,
    line: []const u8,
    dim: usize,
    top_k: u16,
    ef: u16,
    out: *std.io.Writer,
) !ReplAction {
    if (!std.mem.startsWith(u8, line, ":")) {
        const results = try client.searchText(line, top_k, ef);
        defer client.freeSearchResults(results);
        for (results) |r| {
            try out.print("{d}\t{d:.6}\t{s}\n", .{ r.id, r.dist, r.name });
        }
        return .continue_;
    }

    // Colon command. Split verb from the rest on the first whitespace.
    const sep = std.mem.indexOfAny(u8, line, " \t") orelse line.len;
    const cmd = line[0..sep];
    const rest = std.mem.trim(u8, line[sep..], " \t");

    if (std.mem.eql(u8, cmd, ":q") or std.mem.eql(u8, cmd, ":quit") or std.mem.eql(u8, cmd, ":exit")) {
        return .exit;
    }
    if (std.mem.eql(u8, cmd, ":help") or std.mem.eql(u8, cmd, ":h")) {
        try writeQueryReplHelp(out);
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":ping")) {
        try client.ping();
        try out.writeAll("OK\n");
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":stats")) {
        const s = try client.stats();
        try writeStats(out, s, false);
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":snapshot")) {
        const elapsed = try client.snapshot();
        try out.print("snapshot: {d} ns\n", .{elapsed});
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":get")) {
        const id = (try parseReplIdArg(rest, out, ":get")) orelse return .continue_;
        const got = try client.get(id, dim);
        defer allocator.free(got.name);
        defer allocator.free(got.vec);
        try writeGet(out, got.name, got.vec, false, false);
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":delete")) {
        const id = (try parseReplIdArg(rest, out, ":delete")) orelse return .continue_;
        try client.delete(id);
        try out.writeAll("OK\n");
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":insert")) {
        if (rest.len == 0) {
            try out.writeAll("error: :insert requires <text>\n");
            return .continue_;
        }
        const id = try client.insertText(rest);
        try out.print("id={d}\n", .{id});
        return .continue_;
    }
    if (std.mem.eql(u8, cmd, ":replace")) {
        const arg_sep = std.mem.indexOfAny(u8, rest, " \t") orelse {
            try out.writeAll("error: :replace requires <id> <text>\n");
            return .continue_;
        };
        const id_str = rest[0..arg_sep];
        const text = std.mem.trim(u8, rest[arg_sep..], " \t");
        const id = std.fmt.parseInt(u32, id_str, 10) catch {
            try out.writeAll("error: :replace <id> must be a non-negative integer\n");
            return .continue_;
        };
        if (text.len == 0) {
            try out.writeAll("error: :replace requires <text> after <id>\n");
            return .continue_;
        }
        try client.replaceText(id, text);
        try out.writeAll("OK\n");
        return .continue_;
    }
    try out.print("unknown command '{s}' (try :help)\n", .{cmd});
    return .continue_;
}

fn parseReplIdArg(rest: []const u8, out: *std.io.Writer, cmd: []const u8) !?u32 {
    if (rest.len == 0) {
        try out.print("error: {s} requires an <id>\n", .{cmd});
        return null;
    }
    return std.fmt.parseInt(u32, rest, 10) catch {
        try out.print("error: {s} <id> must be a non-negative integer\n", .{cmd});
        return null;
    };
}

fn replPrintServerError(client: *hnswz.client.Client, out: *std.io.Writer) !void {
    const status = @intFromEnum(client.last_status);
    const msg = if (client.last_message.len > 0)
        client.last_message
    else
        hnswz.protocol.statusMessage(client.last_status);
    try out.print("error (status={d}): {s}\n", .{ status, msg });
}

fn writeQueryReplHelp(out: *std.io.Writer) !void {
    try out.writeAll(
        \\Commands:
        \\  <text>                 search-text for <text> (top-k ranked results)
        \\  :stats                 server index state
        \\  :ping                  round-trip check
        \\  :get <id>              fetch name + vec summary for <id>
        \\  :insert <text>         insert a document; prints new id
        \\  :replace <id> <text>   replace <id> with new <text>
        \\  :delete <id>           tombstone <id>
        \\  :snapshot              flush index to disk
        \\  :help | :h             this help
        \\  :q | :quit | :exit     exit
        \\
    );
}

/// Split "host:port" on the LAST colon so IPv6 literals (when supported
/// in the future) don't get mis-parsed. Returns null on malformed input.
fn splitHostPort(s: []const u8) ?struct { host: []const u8, port: u16 } {
    const colon = std.mem.lastIndexOfScalar(u8, s, ':') orelse return null;
    if (colon == 0 or colon == s.len - 1) return null;
    const host = s[0..colon];
    const port = std.fmt.parseInt(u16, s[colon + 1 ..], 10) catch return null;
    return .{ .host = host, .port = port };
}

fn runServe(
    allocator: std.mem.Allocator,
    cfg: *const hnswz.config.Config,
    args: *const cli.ServeArgs,
) !void {
    const Index = hnswz.HnswIndex(M);
    const Server = hnswz.server.Server(M);

    // Resolve listen host:port
    const listen = args.listen orelse "127.0.0.1:9000";
    const parts = splitHostPort(listen) orelse {
        log.err("--listen must be host:port (got '{s}')", .{listen});
        std.process.exit(cli.USAGE_EXIT_CODE);
    };

    // Ensure data_dir exists so snapshot calls don't fail.
    std.fs.cwd().makePath(cfg.storage.data_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => {
            log.err("cannot create data_dir '{s}': {s}", .{ cfg.storage.data_dir, @errorName(err) });
            std.process.exit(1);
        },
    };
    var data_dir = try std.fs.cwd().openDir(cfg.storage.data_dir, .{});
    defer data_dir.close();

    // Hold the lock for the server's lifetime so a second `serve` or
    // concurrent `build` / `query` can't corrupt state under us.
    var lock = hnswz.lockfile.LockFile.acquire(data_dir, cfg.storage.lock_file) catch |err| switch (err) {
        error.AlreadyLocked => {
            log.err("data_dir '{s}' is already locked by another hnswz process", .{cfg.storage.data_dir});
            std.process.exit(1);
        },
        else => {
            log.err("cannot acquire lock on '{s}/{s}': {s}", .{ cfg.storage.data_dir, cfg.storage.lock_file, @errorName(err) });
            std.process.exit(1);
        },
    };
    defer lock.release();

    // Try to load existing store+graph+metadata. If the files are absent
    // we start with a fresh empty index sized from config.
    const had_snapshot = blk: {
        data_dir.access(cfg.storage.vectors_file, .{}) catch break :blk false;
        data_dir.access(cfg.storage.graph_file, .{}) catch break :blk false;
        break :blk true;
    };

    var store: hnswz.Store = if (had_snapshot)
        try hnswz.Store.load(allocator, data_dir, cfg.storage.vectors_file, cfg.storage.max_vectors)
    else
        try hnswz.Store.init(allocator, cfg.embedder.dim, cfg.storage.max_vectors);
    defer store.deinit(allocator);

    if (store.dim != cfg.embedder.dim) {
        log.err("vectors file dim ({d}) != config embedder.dim ({d})", .{ store.dim, cfg.embedder.dim });
        std.process.exit(1);
    }

    const upper_slots: usize = if (had_snapshot) blk: {
        const sz = try Index.scanSize(data_dir, cfg.storage.graph_file);
        break :blk @max(cfg.storage.upper_pool_slots, sz.upper_slots);
    } else @max(cfg.storage.upper_pool_slots, cfg.storage.max_vectors);

    var index: Index = if (had_snapshot)
        try Index.load(allocator, &store, data_dir, cfg.storage.graph_file, .{
            .max_vectors = cfg.storage.max_vectors,
            .max_upper_slots = upper_slots,
            .ef_construction = cfg.index.ef_construction,
            .seed = cfg.index.seed,
        })
    else
        try Index.init(allocator, &store, .{
            .max_vectors = cfg.storage.max_vectors,
            .max_upper_slots = upper_slots,
            .ef_construction = cfg.index.ef_construction,
            .seed = cfg.index.seed,
        });
    defer index.deinit();

    // Populate MutableMetadata from the static sidecar if present.
    var mm = hnswz.metadata_mut.MutableMetadata.init();
    defer mm.deinit(allocator);
    try mm.ensureCapacity(allocator, cfg.storage.max_vectors);
    if (had_snapshot) {
        data_dir.access(cfg.storage.metadata_file, .{}) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        if (data_dir.access(cfg.storage.metadata_file, .{})) |_| {
            var static_md = try hnswz.metadata.load(allocator, data_dir, cfg.storage.metadata_file);
            defer static_md.deinit(allocator);
            if (static_md.count != store.count or static_md.live_count != store.live_count) {
                log.err(
                    "metadata slots ({d}/{d}) != vectors slots ({d}/{d})",
                    .{ static_md.live_count, static_md.count, store.live_count, store.count },
                );
                std.process.exit(1);
            }
            for (0..static_md.count) |i| {
                const id: u32 = @intCast(i);
                if (static_md.isTombstone(id)) continue;
                try mm.setAt(allocator, id, static_md.get(id));
            }
        } else |_| {}
    }

    // Workspaces are owned by the dispatcher's worker pool now — no
    // server-side Workspace allocation here.

    // Open the WAL and replay any records that post-date the snapshot
    // we just loaded. Mutating handlers in the dispatcher will append +
    // fsync to this file for each ack'd write.
    var wal_handle: ?hnswz.wal.Wal = null;
    defer if (wal_handle) |*w| w.close();
    if (cfg.storage.wal_enabled) {
        const res = hnswz.wal.Wal.open(
            allocator,
            data_dir,
            cfg.storage.wal_file,
            @intCast(cfg.embedder.dim),
            &store,
            &index,
            &mm,
        ) catch |err| switch (err) {
            error.DimMismatch => {
                log.err("WAL dim mismatch — config.embedder.dim changed since last run?", .{});
                std.process.exit(1);
            },
            error.CorruptHeader => {
                log.err("WAL header is corrupt; delete '{s}/{s}' if you intend to discard uncommitted writes", .{ cfg.storage.data_dir, cfg.storage.wal_file });
                std.process.exit(1);
            },
            else => {
                log.err("cannot open WAL '{s}/{s}': {s}", .{ cfg.storage.data_dir, cfg.storage.wal_file, @errorName(err) });
                std.process.exit(1);
            },
        };
        if (res.replayed > 0) log.info("replayed {d} WAL record(s){s}", .{
            res.replayed,
            if (res.truncated) " (truncated at corruption)" else "",
        });
        wal_handle = res.wal;
    }
    const wal_ptr: ?*hnswz.wal.Wal = if (wal_handle != null) &wal_handle.? else null;

    // Embedder: Ollama if configured. Text opcodes fail cleanly otherwise.
    var maybe_client: ?hnswz.OllamaClient = null;
    defer if (maybe_client) |*c| c.deinit();
    var embedder: ?hnswz.Embedder = null;
    if (std.mem.eql(u8, cfg.embedder.provider, "ollama")) {
        maybe_client = try hnswz.OllamaClient.init(
            allocator,
            cfg.embedder.model,
            cfg.embedder.base_url,
            .{ .max_text_bytes = cfg.embedder.max_text_bytes, .dim = cfg.embedder.dim },
        );
        embedder = maybe_client.?.embedder();
    }

    const opts: hnswz.server.ServeOptions = .{
        .listen_addr = parts.host,
        .listen_port = parts.port,
        .max_connections = if (args.max_connections) |v| @intCast(v) else 64,
        .max_frame_bytes = if (args.max_frame_bytes) |v| @intCast(v) else hnswz.protocol.MAX_FRAME_BYTES_DEFAULT,
        .idle_timeout_secs = args.idle_timeout_secs orelse 60,
        .auto_snapshot_secs = args.auto_snapshot_secs orelse 0,
        .reuse_address = true,
        .n_workers = args.n_workers orelse 0,
    };

    var server_inst = try Server.init(allocator, &store, &index, &mm, wal_ptr, embedder, cfg, opts);
    defer server_inst.deinit();

    var shutdown_flag: std.atomic.Value(bool) = .init(false);
    hnswz.server.installSignalHandlers(&shutdown_flag, server_inst.shutdown_w);

    log.info("serve: listening on {s}:{d} (loaded={d} vectors)", .{
        parts.host, server_inst.bound_port, store.live_count,
    });

    try server_inst.run(&shutdown_flag);
    log.info("serve: shut down cleanly", .{});
}

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
    args: *const cli.BenchArgs,
) !void {
    // Resolve each knob: CLI > config > hardcoded default.
    const dim: usize = blk: {
        if (args.dim) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.embedder.dim;
        break :blk BENCH_DEFAULTS.dim;
    };
    const ef_construction: usize = blk: {
        if (args.ef_construction) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.index.ef_construction;
        break :blk BENCH_DEFAULTS.ef_construction;
    };
    const ef_search: usize = blk: {
        if (args.ef_search) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.index.ef_search;
        break :blk BENCH_DEFAULTS.ef_search;
    };
    const seed: u64 = blk: {
        if (args.seed) |v| break :blk v;
        if (cfg_opt) |c| break :blk c.index.seed;
        break :blk BENCH_DEFAULTS.seed;
    };
    const top_k: usize = args.top_k orelse cli.DEFAULT_BENCH_TOP_K;
    const num_vectors: usize = args.num_vectors orelse BENCH_DEFAULTS.num_vectors;
    const num_queries: usize = args.num_queries orelse BENCH_DEFAULTS.num_queries;
    const warmup: usize = args.warmup orelse BENCH_DEFAULTS.warmup;

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

    const transport: hnswz.benchmark.Transport = switch (args.transport orelse .in_process) {
        .in_process => .in_process,
        .tcp => .tcp,
    };

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
        .validate = args.validate,
        .json = args.json,
        // Generous default that matches the per-node slot budget used in
        // the hnsw tests. Expected mean is num_vectors / (M-1).
        .upper_pool_slots = num_vectors,
        .transport = transport,
        .bench_protocol = args.bench_protocol,
        .concurrent_clients = args.concurrent_clients orelse 1,
        .server_n_workers = args.server_workers orelse 0,
        .tcp_max_connections = @intCast(@min(
            @as(usize, std.math.maxInt(u16)),
            // Always have room for at least the concurrent clients + a
            // small buffer.
            (args.concurrent_clients orelse 1) + 8,
        )),
        .dataset_dir = args.dataset,
    };

    var out_buf: [8192]u8 = undefined;
    var stdout_file_writer = std.fs.File.stdout().writer(&out_buf);
    const out = &stdout_file_writer.interface;

    try hnswz.benchmark.run(allocator, opts, out);
}

fn runClient(allocator: std.mem.Allocator, args: *const cli.ClientArgs) !void {
    const verb = args.verb orelse {
        log.err("'hnswz client' requires a verb (e.g. 'ping', 'stats', 'insert-text ...')", .{});
        std.process.exit(cli.USAGE_EXIT_CODE);
    };

    const connect = args.connect orelse "127.0.0.1:9000";
    const parts = splitHostPort(connect) orelse {
        log.err("--connect must be host:port (got '{s}')", .{connect});
        std.process.exit(cli.USAGE_EXIT_CODE);
    };

    const addr = std.net.Address.parseIp(parts.host, parts.port) catch |err| {
        log.err("invalid connect address '{s}': {s}", .{ connect, @errorName(err) });
        std.process.exit(cli.USAGE_EXIT_CODE);
    };

    var client = hnswz.client.Client.connect(allocator, addr, .{
        // Generous: STATS needs tiny, GET at dim=4096 is ~16 KiB, SEARCH
        // with top_k=100 and name<=256 is ~27 KiB.
        .recv_buf_size = 4 * 1024 * 1024,
        .send_buf_size = 4 * 1024 * 1024,
    }) catch |err| {
        log.err("cannot connect to {s}: {s}", .{ connect, @errorName(err) });
        std.process.exit(1);
    };
    defer client.deinit();

    var out_buf: [8192]u8 = undefined;
    var stdout_file_writer = std.fs.File.stdout().writer(&out_buf);
    const out = &stdout_file_writer.interface;

    runClientVerb(allocator, &client, args, verb, out) catch |err| switch (err) {
        error.ClientVerbFailed => {
            out.flush() catch {};
            std.process.exit(1);
        },
        else => return err,
    };
    try out.flush();
}

/// Split out from `runClient` so tests can drive it directly with an
/// already-connected client and a fixed-buffer writer — no real
/// subprocess, no stdout capture gymnastics.
///
/// On "expected" failures (server returned non-OK, user supplied bad
/// arguments, malformed vector input), the diagnostic is written to
/// `out` and `error.ClientVerbFailed` is returned. The outer `runClient`
/// maps that to an exit code of 1. Tests assert on `out`'s contents.
fn runClientVerb(
    allocator: std.mem.Allocator,
    client: *hnswz.client.Client,
    args: *const cli.ClientArgs,
    verb: cli.ClientVerb,
    out: *std.io.Writer,
) !void {
    const json_mode = args.json;

    switch (verb) {
        .ping => {
            client.ping() catch |err| return reportVerbError(err, client, out, json_mode);
            if (json_mode) try out.writeAll("{\"ok\":true}\n") else try out.writeAll("OK\n");
        },
        .stats => {
            const s = client.stats() catch |err| return reportVerbError(err, client, out, json_mode);
            try writeStats(out, s, json_mode);
        },
        .snapshot => {
            const elapsed = client.snapshot() catch |err| return reportVerbError(err, client, out, json_mode);
            if (json_mode) {
                try out.print("{{\"elapsed_ns\":{d}}}\n", .{elapsed});
            } else {
                try out.print("snapshot: {d} ns\n", .{elapsed});
            }
        },
        .delete => {
            const id = try requireIdPos(args.pos0, out, json_mode);
            client.delete(id) catch |err| return reportVerbError(err, client, out, json_mode);
            if (json_mode) try out.writeAll("{\"ok\":true}\n") else try out.writeAll("OK\n");
        },
        .get => {
            const id = try requireIdPos(args.pos0, out, json_mode);
            const dim = args.dim orelse (client.stats() catch |err|
                return reportVerbError(err, client, out, json_mode)).dim;
            const got = client.get(id, dim) catch |err| return reportVerbError(err, client, out, json_mode);
            defer allocator.free(got.name);
            defer allocator.free(got.vec);
            try writeGet(out, got.name, got.vec, args.full_vec, json_mode);
        },
        .insert_text => {
            const text = try requireTextPos(args.pos0, out, json_mode);
            const id = client.insertText(text) catch |err| return reportVerbError(err, client, out, json_mode);
            try writeIdOnly(out, id, json_mode);
        },
        .replace_text => {
            const id = try requireIdPos(args.pos0, out, json_mode);
            const text = try requireTextPos(args.pos1, out, json_mode);
            client.replaceText(id, text) catch |err| return reportVerbError(err, client, out, json_mode);
            if (json_mode) try out.writeAll("{\"ok\":true}\n") else try out.writeAll("OK\n");
        },
        .search_text => {
            const text = try requireTextPos(args.pos0, out, json_mode);
            const top_k: u16 = @intCast(args.top_k orelse cli.DEFAULT_TOP_K);
            const ef: u16 = if (args.ef) |e| @intCast(e) else @max(top_k, 10);
            const results = client.searchText(text, top_k, ef) catch |err|
                return reportVerbError(err, client, out, json_mode);
            defer client.freeSearchResults(results);
            try writeSearchResults(out, results, json_mode);
        },
        .insert_vec => {
            const dim = try resolveDim(args, client, out, json_mode);
            const vec = try readVec(allocator, args, dim, out, json_mode);
            defer allocator.free(vec);
            const id = client.insertVec(vec) catch |err| return reportVerbError(err, client, out, json_mode);
            try writeIdOnly(out, id, json_mode);
        },
        .replace_vec => {
            const id = try requireIdPos(args.pos0, out, json_mode);
            const dim = try resolveDim(args, client, out, json_mode);
            const vec = try readVec(allocator, args, dim, out, json_mode);
            defer allocator.free(vec);
            client.replaceVec(id, vec) catch |err| return reportVerbError(err, client, out, json_mode);
            if (json_mode) try out.writeAll("{\"ok\":true}\n") else try out.writeAll("OK\n");
        },
        .search_vec => {
            const dim = try resolveDim(args, client, out, json_mode);
            const vec = try readVec(allocator, args, dim, out, json_mode);
            defer allocator.free(vec);
            const top_k: u16 = @intCast(args.top_k orelse cli.DEFAULT_TOP_K);
            const ef: u16 = if (args.ef) |e| @intCast(e) else @max(top_k, 10);
            const results = client.searchVec(vec, top_k, ef) catch |err|
                return reportVerbError(err, client, out, json_mode);
            defer client.freeSearchResults(results);
            try writeSearchResults(out, results, json_mode);
        },
    }
}

fn requireIdPos(maybe: ?[]u8, out: *std.io.Writer, json_mode: bool) !u32 {
    const s = maybe orelse return writeUsageError(out, json_mode, "verb requires an <id> positional argument");
    return std.fmt.parseInt(u32, s, 10) catch {
        return writeUsageError(out, json_mode, "<id> must be a non-negative integer");
    };
}

fn requireTextPos(maybe: ?[]u8, out: *std.io.Writer, json_mode: bool) ![]const u8 {
    return maybe orelse return writeUsageError(out, json_mode, "verb requires a <text> positional argument");
}

fn resolveDim(args: *const cli.ClientArgs, client: *hnswz.client.Client, out: *std.io.Writer, json_mode: bool) !usize {
    if (args.dim) |d| return d;
    const s = client.stats() catch |err| return reportVerbError(err, client, out, json_mode);
    return s.dim;
}

/// Materialize a `[]f32` of length `dim` from whichever source the user
/// picked. Caller must free. Accepts exactly one of --from-file /
/// --from-stdin / --literal.
fn readVec(
    allocator: std.mem.Allocator,
    args: *const cli.ClientArgs,
    dim: usize,
    out: *std.io.Writer,
    json_mode: bool,
) ![]f32 {
    const have_file = args.from_file != null;
    const have_stdin = args.from_stdin;
    const have_literal = args.literal != null;
    const picked: u2 = @as(u2, @intFromBool(have_file)) + @as(u2, @intFromBool(have_stdin)) + @as(u2, @intFromBool(have_literal));
    if (picked == 0) {
        return writeUsageError(out, json_mode, "vector verbs require one of --from-file, --from-stdin, or --literal");
    }
    if (picked > 1) {
        return writeUsageError(out, json_mode, "--from-file, --from-stdin, and --literal are mutually exclusive");
    }

    const vec = try allocator.alloc(f32, dim);
    errdefer allocator.free(vec);

    if (args.from_file) |path| {
        const file = std.fs.cwd().openFile(path, .{}) catch {
            return writeUsageError(out, json_mode, "--from-file: cannot open file");
        };
        defer file.close();
        const bytes = std.mem.sliceAsBytes(vec);
        const n = file.readAll(bytes) catch {
            return writeUsageError(out, json_mode, "--from-file: read failed");
        };
        if (n != bytes.len) {
            return writeUsageError(out, json_mode, "--from-file: byte count does not match dim * 4");
        }
        return vec;
    }
    if (have_stdin) {
        const bytes = std.mem.sliceAsBytes(vec);
        var read_buf: [4096]u8 = undefined;
        var stdin = std.fs.File.stdin().readerStreaming(&read_buf);
        const r = &stdin.interface;
        r.readSliceAll(bytes) catch {
            return writeUsageError(out, json_mode, "--from-stdin: read failed or ended early");
        };
        return vec;
    }
    // literal: comma-separated floats
    const literal = args.literal.?;
    var i: usize = 0;
    var it = std.mem.splitScalar(u8, literal, ',');
    while (it.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " \t");
        if (trimmed.len == 0) continue;
        if (i >= dim) {
            return writeUsageError(out, json_mode, "--literal has more values than dim");
        }
        vec[i] = std.fmt.parseFloat(f32, trimmed) catch {
            return writeUsageError(out, json_mode, "--literal value is not a valid float");
        };
        i += 1;
    }
    if (i != dim) {
        return writeUsageError(out, json_mode, "--literal count does not match dim");
    }
    return vec;
}

fn writeUsageError(out: *std.io.Writer, json_mode: bool, msg: []const u8) error{ClientVerbFailed} {
    if (json_mode) {
        out.print("{{\"error\":\"", .{}) catch return error.ClientVerbFailed;
        writeJsonEscaped(out, msg) catch return error.ClientVerbFailed;
        out.writeAll("\"}\n") catch return error.ClientVerbFailed;
    } else {
        out.print("error: {s}\n", .{msg}) catch return error.ClientVerbFailed;
    }
    return error.ClientVerbFailed;
}

fn reportVerbError(
    err: anyerror,
    client: *hnswz.client.Client,
    out: *std.io.Writer,
    json_mode: bool,
) error{ClientVerbFailed} {
    if (err != error.ServerError) {
        if (json_mode) {
            out.print("{{\"error\":\"{s}\"}}\n", .{@errorName(err)}) catch {};
        } else {
            out.print("error: {s}\n", .{@errorName(err)}) catch {};
        }
        return error.ClientVerbFailed;
    }
    const status = @intFromEnum(client.last_status);
    const msg = if (client.last_message.len > 0)
        client.last_message
    else
        hnswz.protocol.statusMessage(client.last_status);
    if (json_mode) {
        out.print("{{\"error\":\"", .{}) catch {};
        writeJsonEscaped(out, msg) catch {};
        out.print("\",\"status\":{d}}}\n", .{status}) catch {};
    } else {
        out.print("error (status={d}): {s}\n", .{ status, msg }) catch {};
    }
    return error.ClientVerbFailed;
}

fn writeIdOnly(out: *std.io.Writer, id: u32, json_mode: bool) !void {
    if (json_mode) {
        try out.print("{{\"id\":{d}}}\n", .{id});
    } else {
        try out.print("id={d}\n", .{id});
    }
}

fn writeStats(out: *std.io.Writer, s: hnswz.protocol.StatsResponse, json_mode: bool) !void {
    if (json_mode) {
        try out.print(
            "{{\"proto_version\":{d},\"dim\":{d},\"M\":{d},\"live_count\":{d},\"high_water\":{d},\"upper_used\":{d},\"max_upper_slots\":{d},\"max_level\":{d},\"has_entry_point\":{s}}}\n",
            .{
                s.proto_version, s.dim, s.m, s.live_count, s.high_water,
                s.upper_used,   s.max_upper_slots, s.max_level,
                if (s.has_entry_point != 0) "true" else "false",
            },
        );
    } else {
        try out.print(
            \\proto_version   {d}
            \\dim             {d}
            \\M               {d}
            \\live_count      {d}
            \\high_water      {d}
            \\upper_used      {d}/{d}
            \\max_level       {d}
            \\has_entry_point {s}
            \\
        , .{
            s.proto_version, s.dim, s.m, s.live_count, s.high_water,
            s.upper_used,    s.max_upper_slots, s.max_level,
            if (s.has_entry_point != 0) "yes" else "no",
        });
    }
}

fn writeGet(
    out: *std.io.Writer,
    name: []const u8,
    vec: []const f32,
    full_vec: bool,
    json_mode: bool,
) !void {
    if (json_mode) {
        try out.print("{{\"name\":\"", .{});
        try writeJsonEscaped(out, name);
        try out.writeAll("\",\"vec\":[");
        for (vec, 0..) |x, i| {
            if (i > 0) try out.writeAll(",");
            try out.print("{d}", .{x});
        }
        try out.writeAll("]}\n");
    } else {
        try out.print("name={s}\n", .{name});
        if (full_vec) {
            for (vec, 0..) |x, i| {
                if (i > 0) try out.writeAll(",");
                try out.print("{d}", .{x});
            }
            try out.writeAll("\n");
        } else {
            try out.print("vec=<{d} floats; pass --full-vec to print>\n", .{vec.len});
        }
    }
}

fn writeSearchResults(
    out: *std.io.Writer,
    results: []const hnswz.client.ClientSearchResult,
    json_mode: bool,
) !void {
    if (json_mode) {
        try out.writeAll("{\"results\":[");
        for (results, 0..) |r, i| {
            if (i > 0) try out.writeAll(",");
            try out.print("{{\"id\":{d},\"dist\":{d},\"name\":\"", .{ r.id, r.dist });
            try writeJsonEscaped(out, r.name);
            try out.writeAll("\"}");
        }
        try out.writeAll("]}\n");
    } else {
        try out.writeAll("id\tdist\tname\n");
        for (results) |r| {
            try out.print("{d}\t{d:.6}\t{s}\n", .{ r.id, r.dist, r.name });
        }
    }
}

/// Escape a string for inclusion inside a JSON "..." literal. Handles
/// only the bare minimum (", \, control chars); sufficient for server
/// diagnostics and short metadata names.
fn writeJsonEscaped(out: *std.io.Writer, s: []const u8) !void {
    for (s) |b| {
        switch (b) {
            '"' => try out.writeAll("\\\""),
            '\\' => try out.writeAll("\\\\"),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            0...0x08, 0x0b, 0x0c, 0x0e...0x1f => try out.print("\\u{x:0>4}", .{b}),
            else => try out.writeByte(b),
        }
    }
}

test {
    // Ensure cli.zig's tests get compiled when running the exe test binary.
    _ = @import("cli.zig");
}

// Each test spins up a real `hnswz.server.Server` on an ephemeral port
// with a `FakeEmbedder`, connects a `hnswz.client.Client`, and drives
// the full `runClientVerb` path. Assertions run against a fixed-buffer
// writer so stdout capture isn't needed.

const testing = std.testing;

const ClientTestCtx = struct {
    allocator: std.mem.Allocator,
    cfg: hnswz.config.Config,
    store: hnswz.Store,
    index: hnswz.HnswIndex(M),
    md: hnswz.metadata_mut.MutableMetadata,
    embedder: hnswz.FakeEmbedder,
    srv: hnswz.server.Server(M),
    shutdown: std.atomic.Value(bool),
    thread: ?std.Thread,
    port: u16,

    fn setUp(self: *ClientTestCtx, allocator: std.mem.Allocator, dim: usize, max_vectors: usize) !void {
        self.* = undefined;
        self.allocator = allocator;
        self.shutdown = .init(false);
        self.thread = null;
        self.cfg = .{
            .embedder = .{ .provider = "ollama", .base_url = "http://localhost:11434", .model = "fake", .dim = dim, .max_text_bytes = 1024 },
            .index = .{ .ef_construction = 40, .ef_search = 40, .max_ef = 40 },
            .storage = .{ .data_dir = "./test-client-data", .max_vectors = max_vectors },
        };
        self.store = try hnswz.Store.init(allocator, dim, max_vectors);
        self.index = try hnswz.HnswIndex(M).init(allocator, &self.store, .{
            .max_vectors = max_vectors,
            .max_upper_slots = max_vectors,
            .ef_construction = self.cfg.index.ef_construction,
            .seed = self.cfg.index.seed,
        });
        self.md = hnswz.metadata_mut.MutableMetadata.init();
        self.embedder = hnswz.FakeEmbedder.init(dim, 0xabcd);
        self.srv = try hnswz.server.Server(M).init(
            allocator,
            &self.store,
            &self.index,
            &self.md,
            null, // test harness: WAL off
            self.embedder.embedder(),
            &self.cfg,
            .{
                .listen_addr = "127.0.0.1",
                .listen_port = 0,
                .max_connections = 4,
                .max_frame_bytes = 1 << 20,
                .idle_timeout_secs = 0,
                .auto_snapshot_secs = 0,
                .reuse_address = true,
                .skip_final_snapshot = true,
                .n_workers = 2,
            },
        );
        self.port = self.srv.bound_port;
    }

    fn start(self: *ClientTestCtx) !void {
        self.thread = try std.Thread.spawn(.{}, runner, .{self});
    }

    fn runner(self: *ClientTestCtx) !void {
        try self.srv.run(&self.shutdown);
    }

    fn tearDown(self: *ClientTestCtx) void {
        self.srv.requestShutdown();
        if (self.thread) |t| t.join();
        self.thread = null;
        self.srv.deinit();
        self.md.deinit(self.allocator);
        self.index.deinit();
        self.store.deinit(self.allocator);
    }
};

fn connectClient(allocator: std.mem.Allocator, port: u16) !hnswz.client.Client {
    const addr = try std.net.Address.parseIp("127.0.0.1", port);
    return try hnswz.client.Client.connect(allocator, addr, .{});
}

test "client verb: ping prints OK" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [256]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const args: cli.ClientArgs = .{ .verb = .ping };
    try runClientVerb(testing.allocator, &client, &args, .ping, &w);
    try testing.expectEqualStrings("OK\n", buf[0..w.end]);
}

test "client verb: stats pretty-prints dim and M" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [512]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const args: cli.ClientArgs = .{ .verb = .stats };
    try runClientVerb(testing.allocator, &client, &args, .stats, &w);
    const out = buf[0..w.end];
    try testing.expect(std.mem.indexOf(u8, out, "dim             4") != null);
    try testing.expect(std.mem.indexOf(u8, out, "M               16") != null);
    try testing.expect(std.mem.indexOf(u8, out, "live_count      0") != null);
}

test "client verb: stats --json emits parseable object" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [512]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const args: cli.ClientArgs = .{ .verb = .stats, .json = true };
    try runClientVerb(testing.allocator, &client, &args, .stats, &w);
    const out = buf[0..w.end];
    try testing.expect(std.mem.startsWith(u8, out, "{"));
    try testing.expect(std.mem.indexOf(u8, out, "\"dim\":4") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"M\":16") != null);
}

test "client verb: insert-vec via --literal prints id=0" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [128]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const lit_buf = try testing.allocator.dupe(u8, "1.0,0,0,0");
    defer testing.allocator.free(lit_buf);
    const args: cli.ClientArgs = .{
        .verb = .insert_vec,
        .dim = 4,
        .literal = lit_buf,
    };
    try runClientVerb(testing.allocator, &client, &args, .insert_vec, &w);
    try testing.expectEqualStrings("id=0\n", buf[0..w.end]);
}

test "client verb: insert-vec + search-vec finds inserted vec" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [256]u8 = undefined;

    const lit_buf = try testing.allocator.dupe(u8, "1.0,0,0,0");
    defer testing.allocator.free(lit_buf);
    const insert_args: cli.ClientArgs = .{
        .verb = .insert_vec,
        .dim = 4,
        .literal = lit_buf,
    };
    var w1 = std.io.Writer.fixed(&buf);
    try runClientVerb(testing.allocator, &client, &insert_args, .insert_vec, &w1);

    const search_args: cli.ClientArgs = .{
        .verb = .search_vec,
        .dim = 4,
        .literal = lit_buf[0..9],
        .top_k = 1,
        .ef = 10,
    };
    var w2 = std.io.Writer.fixed(&buf);
    try runClientVerb(testing.allocator, &client, &search_args, .search_vec, &w2);
    const out = buf[0..w2.end];
    try testing.expect(std.mem.startsWith(u8, out, "id\tdist\tname\n"));
    try testing.expect(std.mem.indexOf(u8, out, "0\t0.000000") != null);
}

test "client verb: insert-vec with bad dim surfaces ServerError" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [256]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const lit_buf = try testing.allocator.dupe(u8, "1.0,2.0");
    defer testing.allocator.free(lit_buf);
    const args: cli.ClientArgs = .{
        .verb = .insert_vec,
        .dim = 2, // wrong, server expects 4
        .literal = lit_buf,
    };
    try testing.expectError(error.ClientVerbFailed, runClientVerb(testing.allocator, &client, &args, .insert_vec, &w));
    const out = buf[0..w.end];
    try testing.expect(std.mem.indexOf(u8, out, "status=5") != null); // dim_mismatch
}

test "client verb: delete requires id positional" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [128]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const args: cli.ClientArgs = .{ .verb = .delete };
    try testing.expectError(error.ClientVerbFailed, runClientVerb(testing.allocator, &client, &args, .delete, &w));
    try testing.expect(std.mem.indexOf(u8, buf[0..w.end], "<id>") != null);
}

test "client verb: insert-vec rejects multiple vector sources" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [256]u8 = undefined;
    var w = std.io.Writer.fixed(&buf);

    const lit_buf = try testing.allocator.dupe(u8, "1.0,0,0,0");
    defer testing.allocator.free(lit_buf);
    const args: cli.ClientArgs = .{
        .verb = .insert_vec,
        .dim = 4,
        .literal = lit_buf[0..9],
        .from_stdin = true, // also set — should error
    };
    try testing.expectError(error.ClientVerbFailed, runClientVerb(testing.allocator, &client, &args, .insert_vec, &w));
    try testing.expect(std.mem.indexOf(u8, buf[0..w.end], "mutually exclusive") != null);
}

test "client verb: get returns name and vec" {
    var ctx: ClientTestCtx = undefined;
    try ctx.setUp(testing.allocator, 4, 8);
    defer ctx.tearDown();
    try ctx.start();

    var client = try connectClient(testing.allocator, ctx.port);
    defer client.deinit();

    var buf: [512]u8 = undefined;

    // First insert something
    const lit_buf = try testing.allocator.dupe(u8, "1.0,0,0,0");
    defer testing.allocator.free(lit_buf);
    const insert_args: cli.ClientArgs = .{
        .verb = .insert_vec,
        .dim = 4,
        .literal = lit_buf,
    };
    var w1 = std.io.Writer.fixed(&buf);
    try runClientVerb(testing.allocator, &client, &insert_args, .insert_vec, &w1);

    const pos0_buf = try testing.allocator.dupe(u8, "0");
    defer testing.allocator.free(pos0_buf);
    const get_args: cli.ClientArgs = .{
        .verb = .get,
        .pos0 = pos0_buf,
        .dim = 4,
        .full_vec = true,
    };
    var w2 = std.io.Writer.fixed(&buf);
    try runClientVerb(testing.allocator, &client, &get_args, .get, &w2);
    const out = buf[0..w2.end];
    try testing.expect(std.mem.indexOf(u8, out, "name=") != null);
    try testing.expect(std.mem.indexOf(u8, out, "1") != null); // the vec contains 1.0
}

// End-to-end crash-recovery test: spin a server with a real WAL, send
// writes, stop the server WITHOUT a final snapshot (simulating kill
// -9), then spin a fresh server on the same data_dir and verify the
// WAL replay brought all state back.
test "WAL: crash recovery restores writes across restart" {
    const dim: usize = 4;
    const max_vectors: usize = 16;
    const allocator = testing.allocator;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const data_dir_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(data_dir_path);

    const cfg: hnswz.config.Config = .{
        .embedder = .{ .provider = "ollama", .base_url = "http://localhost:11434", .model = "fake", .dim = dim, .max_text_bytes = 1024 },
        .index = .{ .ef_construction = 16, .ef_search = 16, .max_ef = 16 },
        .storage = .{ .data_dir = data_dir_path, .max_vectors = max_vectors },
    };

    // --- phase 1: server A writes some vectors, then goes away ---
    const first_id: u32, const second_id: u32 = blk: {
        var store = try hnswz.Store.init(allocator, dim, max_vectors);
        defer store.deinit(allocator);
        var index = try hnswz.HnswIndex(M).init(allocator, &store, .{
            .max_vectors = max_vectors,
            .max_upper_slots = max_vectors,
            .ef_construction = cfg.index.ef_construction,
            .seed = cfg.index.seed,
        });
        defer index.deinit();
        var md = hnswz.metadata_mut.MutableMetadata.init();
        defer md.deinit(allocator);

        var wal_res = try hnswz.wal.Wal.open(
            allocator,
            tmp.dir,
            cfg.storage.wal_file,
            @intCast(dim),
            &store,
            &index,
            &md,
        );
        errdefer wal_res.wal.close();

        var embedder = hnswz.FakeEmbedder.init(dim, 0xabc);
        var srv = try hnswz.server.Server(M).init(
            allocator,
            &store,
            &index,
            &md,
            &wal_res.wal,
            embedder.embedder(),
            &cfg,
            .{
                .listen_addr = "127.0.0.1",
                .listen_port = 0,
                .max_connections = 4,
                .max_frame_bytes = 1 << 20,
                .idle_timeout_secs = 0,
                .auto_snapshot_secs = 0,
                .reuse_address = true,
                .skip_final_snapshot = true, // simulates crash: no snapshot on exit
                .n_workers = 2,
            },
        );
        defer srv.deinit();

        var shutdown: std.atomic.Value(bool) = .init(false);
        const Runner = struct {
            fn run(s: *hnswz.server.Server(M), sd: *std.atomic.Value(bool)) !void {
                try s.run(sd);
            }
        };
        const thread = try std.Thread.spawn(.{}, Runner.run, .{ &srv, &shutdown });

        var client = try connectClient(allocator, srv.bound_port);
        defer client.deinit();

        const vec1 = [_]f32{ 1.0, 0, 0, 0 };
        const vec2 = [_]f32{ 0, 1.0, 0, 0 };
        const id1 = try client.insertVec(&vec1);
        const id2 = try client.insertVec(&vec2);

        // Stop the server WITHOUT final snapshot. WAL should still be
        // fully durable; in-memory state is about to be discarded.
        srv.requestShutdown();
        thread.join();
        wal_res.wal.close();

        break :blk .{ id1, id2 };
    };

    // --- phase 2: fresh server on the same data_dir, verify state ---
    {
        var store = try hnswz.Store.init(allocator, dim, max_vectors);
        defer store.deinit(allocator);
        var index = try hnswz.HnswIndex(M).init(allocator, &store, .{
            .max_vectors = max_vectors,
            .max_upper_slots = max_vectors,
            .ef_construction = cfg.index.ef_construction,
            .seed = cfg.index.seed,
        });
        defer index.deinit();
        var md = hnswz.metadata_mut.MutableMetadata.init();
        defer md.deinit(allocator);

        var wal_res = try hnswz.wal.Wal.open(
            allocator,
            tmp.dir,
            cfg.storage.wal_file,
            @intCast(dim),
            &store,
            &index,
            &md,
        );
        defer wal_res.wal.close();

        try testing.expectEqual(@as(usize, 2), wal_res.replayed);
        try testing.expectEqual(@as(usize, 2), store.live_count);
        try testing.expect(!store.isDeleted(first_id));
        try testing.expect(!store.isDeleted(second_id));

        // Search should find the inserted vec at distance 0.
        var ws = try hnswz.HnswIndex(M).Workspace.init(allocator, max_vectors, cfg.index.ef_search);
        defer ws.deinit(allocator);
        var out: [2]hnswz.heap.Entry = undefined;
        const query = [_]f32{ 1.0, 0, 0, 0 };
        const results = try index.search(&ws, &query, 1, cfg.index.ef_search, &out);
        try testing.expectEqual(@as(usize, 1), results.len);
        try testing.expectEqual(first_id, results[0].id);
    }
}

test "lockfile: serve refuses to start if another holder is active" {
    const allocator = testing.allocator;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var lock1 = try hnswz.lockfile.LockFile.acquire(tmp.dir, "hnswz.lock");
    defer lock1.release();

    const res = hnswz.lockfile.LockFile.acquire(tmp.dir, "hnswz.lock");
    try testing.expectError(error.AlreadyLocked, res);

    _ = allocator; // unused — keep parity with other tests
}
