//! Benchmark harness for the HNSW index.
//!
//! Runs a build + search workload on a synthetic, deterministic dataset and
//! reports throughput plus latency percentiles. The goal is to be the
//! regression signal for the project — each run prints either a human
//! summary or a machine-readable JSON blob (`--json`) that external tooling
//! can diff across commits.
//!
//! Embedding is NOT exercised here. We bypass Ollama entirely and generate
//! random unit-normalized vectors from a seeded PRNG. That keeps the
//! measurement scoped to the index's own work and keeps runs reproducible.

const std = @import("std");
const builtin = @import("builtin");

const Store = @import("store.zig").Store;
const HnswIndexFn = @import("hnsw.zig").HnswIndex;
const bruteforce = @import("bruteforce.zig");
const heap = @import("heap.zig");
const server_mod = @import("server.zig");
const client_mod = @import("client.zig");
const MutableMetadata = @import("metadata_mut.zig").MutableMetadata;
const config_mod = @import("config.zig");
const ollama = @import("ollama.zig");

/// Must match the M baked into the shipped binary (see main.zig).
pub const M: usize = 16;

// Linear histograms; last bucket is the overflow catch-all.
// Search latency is typically sub-ms → 1 µs buckets × 20 000 = 20 ms range.
// Insert latency can spike into ms → 10 µs buckets × 20 000 = 200 ms range.
const SEARCH_BUCKET_NS: u64 = 1_000;
const SEARCH_BUCKETS: usize = 20_000;
const INSERT_BUCKET_NS: u64 = 10_000;
const INSERT_BUCKETS: usize = 20_000;

pub const Transport = enum { in_process, tcp };

pub const Options = struct {
    num_vectors: usize,
    num_queries: usize,
    dim: usize,
    ef_construction: usize,
    ef_search: usize,
    max_ef: usize,
    top_k: usize,
    seed: u64,
    warmup: usize,
    validate: bool,
    json: bool,
    /// Upper-layer pool slots. A generous default is `num_vectors` (matches
    /// the pattern in hnsw tests); the expected mean is N / (M-1).
    upper_pool_slots: usize,
    /// `.in_process` calls directly into `HnswIndex`; `.tcp` spawns a
    /// `server.Server` on a worker thread and drives the workload through
    /// a `client.Client` so the delta is the protocol overhead.
    transport: Transport = .in_process,
    /// When true, skip the build+search workload entirely and instead
    /// measure PING RTT and 1-vector SEARCH_VEC RTT via TCP. Implies
    /// `transport == .tcp`.
    bench_protocol: bool = false,
    /// For `.tcp`: how many clients drive load in parallel during the
    /// search phase (and PING/SEARCH RTT phases of `--bench-protocol`).
    /// Build phase is always single-client (inserts serialize on the
    /// writer anyway). Default 1 = old behaviour.
    concurrent_clients: usize = 1,
    /// For `.tcp`: worker-pool size passed to the server. 0 = auto
    /// (cpu_count - 2). Only meaningful under `transport == .tcp`.
    server_n_workers: usize = 0,
    /// For `.tcp`: size of the server's connection table. Must be >=
    /// `concurrent_clients` to avoid connection rejections. Default 64.
    tcp_max_connections: u16 = 64,
};

/// Fixed-capacity linear-bucket latency histogram.
///
/// `record(ns)` is O(1); `percentileNs(p)` is O(n_buckets) so percentile
/// queries are intended to be called once per phase at report time.
pub const Histogram = struct {
    buckets: []u64,
    bucket_ns: u64,
    samples: u64,
    sum_ns: u128,
    max_sample_ns: u64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_buckets: usize, bucket_ns: u64) !Histogram {
        std.debug.assert(n_buckets >= 2);
        std.debug.assert(bucket_ns >= 1);
        const buckets = try allocator.alloc(u64, n_buckets);
        @memset(buckets, 0);
        return .{
            .buckets = buckets,
            .bucket_ns = bucket_ns,
            .samples = 0,
            .sum_ns = 0,
            .max_sample_ns = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Histogram) void {
        self.allocator.free(self.buckets);
    }

    pub fn record(self: *Histogram, ns: u64) void {
        const raw_idx = ns / self.bucket_ns;
        const max_idx = @as(u64, @intCast(self.buckets.len - 1));
        const idx: usize = @intCast(@min(raw_idx, max_idx));
        self.buckets[idx] += 1;
        self.samples += 1;
        self.sum_ns += ns;
        if (ns > self.max_sample_ns) self.max_sample_ns = ns;
    }

    /// Return the lower-bound ns of the bucket containing the given
    /// percentile. `p` is clamped to [0, 1]. Callers should use
    /// `max_sample_ns` directly for p100 if they want an exact value.
    pub fn percentileNs(self: *const Histogram, p: f32) u64 {
        std.debug.assert(self.samples > 0);
        const pp = std.math.clamp(p, 0.0, 1.0);
        const target_f = @as(f64, @floatFromInt(self.samples)) * @as(f64, pp);
        var target: u64 = @intFromFloat(@ceil(target_f));
        if (target == 0) target = 1;

        var acc: u64 = 0;
        for (self.buckets, 0..) |c, i| {
            acc += c;
            if (acc >= target) return @as(u64, @intCast(i)) * self.bucket_ns;
        }
        return @as(u64, @intCast(self.buckets.len - 1)) * self.bucket_ns;
    }

    pub fn meanNs(self: *const Histogram) u64 {
        if (self.samples == 0) return 0;
        return @intCast(self.sum_ns / @as(u128, self.samples));
    }

    /// True if any sample landed past the tracked range → the final bucket
    /// is polluted with over-range values and tail percentiles understate.
    pub fn overflowed(self: *const Histogram) bool {
        return self.max_sample_ns >= self.buckets.len * self.bucket_ns;
    }

    /// Fold `other` into `self`. Both histograms must have identical
    /// `buckets.len` and `bucket_ns`. Used by the concurrent-clients
    /// search driver to merge per-thread histograms.
    pub fn merge(self: *Histogram, other: *const Histogram) void {
        std.debug.assert(self.buckets.len == other.buckets.len);
        std.debug.assert(self.bucket_ns == other.bucket_ns);
        for (self.buckets, other.buckets) |*a, b| a.* += b;
        self.samples += other.samples;
        self.sum_ns += other.sum_ns;
        if (other.max_sample_ns > self.max_sample_ns) self.max_sample_ns = other.max_sample_ns;
    }
};

pub const PhaseStats = struct {
    wall_ns: u64,
    ops: u64,
    p50_ns: u64,
    p90_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    p100_ns: u64,
    mean_ns: u64,
    overflowed: bool,

    pub fn opsPerSec(self: PhaseStats) f64 {
        if (self.wall_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.ops)) * 1_000_000_000.0 /
            @as(f64, @floatFromInt(self.wall_ns));
    }
};

fn phaseStatsFrom(h: *const Histogram, wall_ns: u64) PhaseStats {
    return .{
        .wall_ns = wall_ns,
        .ops = h.samples,
        .p50_ns = h.percentileNs(0.50),
        .p90_ns = h.percentileNs(0.90),
        .p95_ns = h.percentileNs(0.95),
        .p99_ns = h.percentileNs(0.99),
        .p100_ns = h.max_sample_ns, // exact, not bucketed
        .mean_ns = h.meanNs(),
        .overflowed = h.overflowed(),
    };
}

pub const Report = struct {
    opts: Options,
    build_mode: []const u8,
    build: PhaseStats,
    upper_used: usize,
    search: PhaseStats,
    recall_at_k: ?f32,
};

pub fn run(allocator: std.mem.Allocator, opts: Options, out: *std.io.Writer) !void {
    if (opts.bench_protocol) return runProtocolFloor(allocator, opts, out);
    if (opts.transport == .tcp) return runTcp(allocator, opts, out);
    return runInProcess(allocator, opts, out);
}

fn runInProcess(allocator: std.mem.Allocator, opts: Options, out: *std.io.Writer) !void {
    std.debug.assert(opts.num_vectors > 0);
    std.debug.assert(opts.num_queries > 0);
    std.debug.assert(opts.dim > 0);
    std.debug.assert(opts.ef_construction > 0);
    std.debug.assert(opts.ef_search > 0);
    std.debug.assert(opts.top_k > 0);
    std.debug.assert(opts.top_k <= opts.ef_search);
    std.debug.assert(opts.max_ef >= opts.ef_construction);
    std.debug.assert(opts.max_ef >= opts.ef_search);
    std.debug.assert(opts.upper_pool_slots > 0);
    std.debug.assert(opts.num_vectors >= opts.top_k);

    if (builtin.mode == .Debug) {
        // Loud stderr warning — does not gate the run so the test binary
        // can still smoke-run in Debug, but the numbers are meaningless.
        std.debug.print(
            \\WARNING: benchmark compiled in Debug mode. Results are NOT meaningful.
            \\         Rebuild with `zig build -Doptimize=ReleaseFast` for real numbers.
            \\
        , .{});
    }

    var store = try Store.init(allocator, opts.dim, opts.num_vectors);
    defer store.deinit(allocator);

    const query_buf = try allocator.alloc(f32, opts.num_queries * opts.dim);
    defer allocator.free(query_buf);

    {
        var prng = std.Random.DefaultPrng.init(opts.seed);
        const random = prng.random();
        const scratch = try allocator.alloc(f32, opts.dim);
        defer allocator.free(scratch);

        for (0..opts.num_vectors) |_| {
            for (scratch) |*x| x.* = random.float(f32) * 2.0 - 1.0;
            bruteforce.normalize(scratch);
            _ = try store.add(scratch);
        }
        for (0..opts.num_queries) |qi| {
            const slot = query_buf[qi * opts.dim ..][0..opts.dim];
            for (slot) |*x| x.* = random.float(f32) * 2.0 - 1.0;
            bruteforce.normalize(slot);
        }
    }

    // Ground truth (only if --validate). Flat [num_queries × top_k] layout.
    var truth_ids: ?[]u32 = null;
    defer if (truth_ids) |t| allocator.free(t);
    if (opts.validate) {
        const truth = try allocator.alloc(u32, opts.num_queries * opts.top_k);
        truth_ids = truth;
        for (0..opts.num_queries) |qi| {
            const q = query_buf[qi * opts.dim ..][0..opts.dim];
            const bf = try bruteforce.search(&store, q, opts.top_k, allocator);
            defer allocator.free(bf);
            const got = @min(opts.top_k, bf.len);
            for (0..got) |i| truth[qi * opts.top_k + i] = bf[i].id;
            // Shouldn't happen (num_vectors >= top_k asserted above), but
            // pad with a sentinel so recall computations stay well-defined.
            for (got..opts.top_k) |i| truth[qi * opts.top_k + i] = std.math.maxInt(u32);
        }
    }

    const Index = HnswIndexFn(M);
    var index = try Index.init(allocator, &store, .{
        .max_vectors = opts.num_vectors,
        .max_upper_slots = opts.upper_pool_slots,
        .ef_construction = opts.ef_construction,
        .seed = opts.seed,
    });
    defer index.deinit();

    var ws = try Index.Workspace.init(allocator, opts.num_vectors, opts.max_ef);
    defer ws.deinit(allocator);

    var build_hist = try Histogram.init(allocator, INSERT_BUCKETS, INSERT_BUCKET_NS);
    defer build_hist.deinit();

    var build_wall = try std.time.Timer.start();
    for (0..opts.num_vectors) |i| {
        var op = try std.time.Timer.start();
        try index.insert(&ws, @intCast(i));
        build_hist.record(op.read());
    }
    const build_wall_ns = build_wall.read();
    const build_stats = phaseStatsFrom(&build_hist, build_wall_ns);

    const results_buf = try allocator.alloc(heap.Entry, opts.top_k);
    defer allocator.free(results_buf);
    const got_ids = try allocator.alloc(u32, opts.top_k);
    defer allocator.free(got_ids);

    // Warmup — untimed, warms caches + branch predictor.
    const warmup_n = @min(opts.warmup, opts.num_queries);
    for (0..warmup_n) |qi| {
        const q = query_buf[qi * opts.dim ..][0..opts.dim];
        _ = try index.search(&ws, q, opts.top_k, opts.ef_search, results_buf);
    }

    var search_hist = try Histogram.init(allocator, SEARCH_BUCKETS, SEARCH_BUCKET_NS);
    defer search_hist.deinit();

    var recall_sum: f64 = 0;
    var search_wall = try std.time.Timer.start();
    for (0..opts.num_queries) |qi| {
        const q = query_buf[qi * opts.dim ..][0..opts.dim];
        var op = try std.time.Timer.start();
        const got = try index.search(&ws, q, opts.top_k, opts.ef_search, results_buf);
        search_hist.record(op.read());

        if (truth_ids) |truth| {
            for (0..got.len) |i| got_ids[i] = got[i].id;
            const this_truth = truth[qi * opts.top_k .. (qi + 1) * opts.top_k];
            recall_sum += bruteforce.computeRecall(this_truth, got_ids[0..got.len]);
        }
    }
    const search_wall_ns = search_wall.read();
    const search_stats = phaseStatsFrom(&search_hist, search_wall_ns);

    const recall: ?f32 = if (opts.validate)
        @as(f32, @floatCast(recall_sum / @as(f64, @floatFromInt(opts.num_queries))))
    else
        null;

    const report: Report = .{
        .opts = opts,
        .build_mode = @tagName(builtin.mode),
        .build = build_stats,
        .upper_used = index.upper_used,
        .search = search_stats,
        .recall_at_k = recall,
    };

    if (opts.json) try writeJson(out, report) else try writePretty(out, report);
    try out.flush();
}

fn writePretty(w: *std.io.Writer, r: Report) !void {
    try w.print(
        \\hnswz benchmark
        \\  mode       {s}
        \\  transport  {s}
        \\  seed       {d}
        \\  dim        {d}
        \\  M          {d}
        \\  n          {d}
        \\  q          {d}
        \\  ef_cons    {d}
        \\  ef_search  {d}
        \\  top_k      {d}
        \\  warmup     {d}
        \\
        \\
    , .{
        r.build_mode, @tagName(r.opts.transport), r.opts.seed, r.opts.dim, M,
        r.opts.num_vectors, r.opts.num_queries,
        r.opts.ef_construction, r.opts.ef_search,
        r.opts.top_k, r.opts.warmup,
    });

    try w.writeAll("Build phase\n");
    try writePhase(w, r.build, "inserts");
    try w.print("  upper_used {d}/{d}\n", .{ r.upper_used, r.opts.upper_pool_slots });

    try w.writeAll("\nSearch phase\n");
    try writePhase(w, r.search, "queries");
    if (r.recall_at_k) |rk| {
        try w.print("  recall@{d}  {d:.4}  (validated vs brute force)\n", .{ r.opts.top_k, rk });
    }
}

fn writePhase(w: *std.io.Writer, s: PhaseStats, unit: []const u8) !void {
    const wall_s = @as(f64, @floatFromInt(s.wall_ns)) / 1_000_000_000.0;
    try w.print("  wall       {d:.3} s\n", .{wall_s});
    try w.print("  {s}/s      {d:.1}\n", .{ unit, s.opsPerSec() });
    try w.print(
        "  latency µs p50={d} p90={d} p95={d} p99={d} p100={d} mean={d}{s}\n",
        .{
            s.p50_ns / 1000, s.p90_ns / 1000, s.p95_ns / 1000,
            s.p99_ns / 1000, s.p100_ns / 1000, s.mean_ns / 1000,
            if (s.overflowed) "  (+ exceeds histogram range)" else "",
        },
    );
}

fn writeJson(w: *std.io.Writer, r: Report) !void {
    try w.print(
        \\{{
        \\  "schema_version": 1,
        \\  "build_mode": "{s}",
        \\  "params": {{
        \\    "seed": {d},
        \\    "dim": {d},
        \\    "M": {d},
        \\    "num_vectors": {d},
        \\    "num_queries": {d},
        \\    "ef_construction": {d},
        \\    "ef_search": {d},
        \\    "top_k": {d},
        \\    "warmup": {d},
        \\    "upper_pool_slots": {d}
        \\  }},
        \\
    , .{
        r.build_mode, r.opts.seed, r.opts.dim, M,
        r.opts.num_vectors, r.opts.num_queries,
        r.opts.ef_construction, r.opts.ef_search,
        r.opts.top_k, r.opts.warmup, r.opts.upper_pool_slots,
    });
    try w.writeAll("  \"build\": ");
    try writePhaseJson(w, r.build);
    try w.print(",\n  \"upper_used\": {d},\n  \"search\": ", .{r.upper_used});
    try writePhaseJson(w, r.search);
    try w.writeAll(",\n");
    if (r.recall_at_k) |rk| {
        try w.print("  \"recall_at_k\": {d:.6}\n", .{rk});
    } else {
        try w.writeAll("  \"recall_at_k\": null\n");
    }
    try w.writeAll("}\n");
}

fn writePhaseJson(w: *std.io.Writer, s: PhaseStats) !void {
    try w.print(
        "{{\"wall_ns\": {d}, \"ops\": {d}, \"ops_per_sec\": {d:.3}, " ++
            "\"latency_ns\": {{\"p50\": {d}, \"p90\": {d}, \"p95\": {d}, " ++
            "\"p99\": {d}, \"p100\": {d}, \"mean\": {d}}}, " ++
            "\"overflowed\": {s}}}",
        .{
            s.wall_ns, s.ops, s.opsPerSec(),
            s.p50_ns, s.p90_ns, s.p95_ns, s.p99_ns, s.p100_ns, s.mean_ns,
            if (s.overflowed) "true" else "false",
        },
    );
}

// Spins up a Server on 127.0.0.1:0 in a worker thread, connects a
// Client, and drives the same deterministic workload as the in-process
// path. Report shape is unchanged so JSON diffs across transports are
// trivial: the delta is the protocol overhead.

fn tcpTransportConfig(opts: Options) config_mod.Config {
    return .{
        .embedder = .{
            .provider = "ollama", // unused — no Embedder passed in
            .base_url = "http://localhost:11434",
            .model = "unused",
            .dim = opts.dim,
            .max_text_bytes = 1024,
        },
        .index = .{
            .ef_construction = opts.ef_construction,
            .ef_search = opts.ef_search,
            .max_ef = opts.max_ef,
            .seed = opts.seed,
        },
        .storage = .{
            .data_dir = "./benchmark-serve-data",
            .max_vectors = opts.num_vectors,
            .upper_pool_slots = opts.upper_pool_slots,
        },
    };
}

const TcpHarness = struct {
    allocator: std.mem.Allocator,
    cfg: config_mod.Config,
    store: Store,
    index: HnswIndexFn(M),
    md: MutableMetadata,
    srv: server_mod.Server(M),
    shutdown: std.atomic.Value(bool),
    thread: ?std.Thread,

    fn setUp(self: *TcpHarness, allocator: std.mem.Allocator, opts: Options) !void {
        self.* = undefined;
        self.allocator = allocator;
        self.shutdown = .init(false);
        self.thread = null;
        self.cfg = tcpTransportConfig(opts);

        self.store = try Store.init(allocator, opts.dim, opts.num_vectors);
        self.index = try HnswIndexFn(M).init(allocator, &self.store, .{
            .max_vectors = opts.num_vectors,
            .max_upper_slots = opts.upper_pool_slots,
            .ef_construction = opts.ef_construction,
            .seed = opts.seed,
        });
        self.md = MutableMetadata.init();

        self.srv = try server_mod.Server(M).init(
            allocator,
            &self.store,
            &self.index,
            &self.md,
            null, // WAL disabled — benchmark is in-process, no durability needed
            null, // no embedder — benchmark uses *_VEC opcodes only
            &self.cfg,
            .{
                .listen_addr = "127.0.0.1",
                .listen_port = 0,
                .max_connections = opts.tcp_max_connections,
                .max_frame_bytes = protocol_mod.MAX_FRAME_BYTES_DEFAULT,
                .idle_timeout_secs = 0,
                .auto_snapshot_secs = 0,
                .reuse_address = true,
                .skip_final_snapshot = true,
                .n_workers = opts.server_n_workers,
            },
        );
    }

    fn runner(self: *TcpHarness) !void {
        try self.srv.run(&self.shutdown);
    }

    fn start(self: *TcpHarness) !void {
        self.thread = try std.Thread.spawn(.{}, runner, .{self});
    }

    fn tearDown(self: *TcpHarness) void {
        self.srv.requestShutdown();
        if (self.thread) |t| t.join();
        self.thread = null;
        self.srv.deinit();
        self.md.deinit(self.allocator);
        self.index.deinit();
        self.store.deinit(self.allocator);
    }
};

const protocol_mod = @import("protocol.zig");

fn runTcp(allocator: std.mem.Allocator, opts: Options, out: *std.io.Writer) !void {
    std.debug.assert(opts.num_vectors > 0);
    std.debug.assert(opts.num_queries > 0);
    std.debug.assert(opts.dim > 0);
    std.debug.assert(opts.top_k > 0);
    std.debug.assert(opts.top_k <= opts.ef_search);
    std.debug.assert(opts.max_ef >= opts.ef_construction);
    std.debug.assert(opts.max_ef >= opts.ef_search);
    std.debug.assert(opts.num_vectors >= opts.top_k);

    if (builtin.mode == .Debug) {
        std.debug.print(
            \\WARNING: benchmark compiled in Debug mode. Results are NOT meaningful.
            \\         Rebuild with `zig build -Doptimize=ReleaseFast` for real numbers.
            \\
        , .{});
    }

    var h: TcpHarness = undefined;
    try h.setUp(allocator, opts);
    defer h.tearDown();
    try h.start();

    const addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port);
    var client = try client_mod.Client.connect(allocator, addr, .{
        .recv_buf_size = 4 * 1024 * 1024,
        .send_buf_size = 4 * 1024 * 1024,
    });
    defer client.deinit();

    // Deterministic vector / query generation — same algorithm, same
    // seed as in_process.
    const query_buf = try allocator.alloc(f32, opts.num_queries * opts.dim);
    defer allocator.free(query_buf);
    const vec_buf = try allocator.alloc(f32, opts.num_vectors * opts.dim);
    defer allocator.free(vec_buf);

    {
        var prng = std.Random.DefaultPrng.init(opts.seed);
        const random = prng.random();
        for (0..opts.num_vectors) |i| {
            const slot = vec_buf[i * opts.dim ..][0..opts.dim];
            for (slot) |*x| x.* = random.float(f32) * 2.0 - 1.0;
            bruteforce.normalize(slot);
        }
        for (0..opts.num_queries) |qi| {
            const slot = query_buf[qi * opts.dim ..][0..opts.dim];
            for (slot) |*x| x.* = random.float(f32) * 2.0 - 1.0;
            bruteforce.normalize(slot);
        }
    }

    var build_hist = try Histogram.init(allocator, INSERT_BUCKETS, INSERT_BUCKET_NS);
    defer build_hist.deinit();

    var build_wall = try std.time.Timer.start();
    for (0..opts.num_vectors) |i| {
        const v = vec_buf[i * opts.dim ..][0..opts.dim];
        var op = try std.time.Timer.start();
        _ = try client.insertVec(v);
        build_hist.record(op.read());
    }
    const build_wall_ns = build_wall.read();
    const build_stats = phaseStatsFrom(&build_hist, build_wall_ns);

    var truth_ids: ?[]u32 = null;
    defer if (truth_ids) |t| allocator.free(t);
    if (opts.validate) {
        const truth = try allocator.alloc(u32, opts.num_queries * opts.top_k);
        truth_ids = truth;
        for (0..opts.num_queries) |qi| {
            const q = query_buf[qi * opts.dim ..][0..opts.dim];
            const bf = try bruteforce.search(&h.store, q, opts.top_k, allocator);
            defer allocator.free(bf);
            const got = @min(opts.top_k, bf.len);
            for (0..got) |i| truth[qi * opts.top_k + i] = bf[i].id;
            for (got..opts.top_k) |i| truth[qi * opts.top_k + i] = std.math.maxInt(u32);
        }
    }

    // `concurrent_clients` controls how many client connections drive
    // the workload in parallel. 1 is the old sequential path. >1
    // spawns N threads that pull query indices from an atomic counter,
    // each with their own Client + Histogram + recall accumulator.
    // Per-thread histograms merge at the end.

    // Warmup still uses the main-thread client so the threads don't
    // race on a shared client during warmup.
    const warmup_n = @min(opts.warmup, opts.num_queries);
    for (0..warmup_n) |qi| {
        const q = query_buf[qi * opts.dim ..][0..opts.dim];
        const r = try client.searchVec(q, @intCast(opts.top_k), @intCast(opts.ef_search));
        client.freeSearchResults(r);
    }

    var search_hist = try Histogram.init(allocator, SEARCH_BUCKETS, SEARCH_BUCKET_NS);
    defer search_hist.deinit();

    const n_clients = @max(@as(usize, 1), opts.concurrent_clients);

    const SearchDriver = struct {
        allocator: std.mem.Allocator,
        addr: std.net.Address,
        query_buf: []f32,
        dim: usize,
        top_k: usize,
        ef_search: usize,
        num_queries: usize,
        cursor: std.atomic.Value(usize),
        validate: bool,
        truth_ids: ?[]u32,

        // Aggregated across threads.
        merge_mutex: std.Thread.Mutex = .{},
        merged_hist: *Histogram,
        merged_recall_sum: f64 = 0,
        first_err: ?anyerror = null,
    };

    var driver: SearchDriver = .{
        .allocator = allocator,
        .addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port),
        .query_buf = query_buf,
        .dim = opts.dim,
        .top_k = opts.top_k,
        .ef_search = opts.ef_search,
        .num_queries = opts.num_queries,
        .cursor = .init(0),
        .validate = opts.validate,
        .truth_ids = truth_ids,
        .merged_hist = &search_hist,
    };

    const WorkerFn = struct {
        fn run(d: *SearchDriver) void {
            var local_hist = Histogram.init(d.allocator, SEARCH_BUCKETS, SEARCH_BUCKET_NS) catch |e| {
                d.merge_mutex.lock();
                if (d.first_err == null) d.first_err = e;
                d.merge_mutex.unlock();
                return;
            };
            defer local_hist.deinit();

            var c = client_mod.Client.connect(d.allocator, d.addr, .{
                .recv_buf_size = 4 * 1024 * 1024,
                .send_buf_size = 4 * 1024 * 1024,
            }) catch |e| {
                d.merge_mutex.lock();
                if (d.first_err == null) d.first_err = e;
                d.merge_mutex.unlock();
                return;
            };
            defer c.deinit();

            const got_ids_local = d.allocator.alloc(u32, d.top_k) catch |e| {
                d.merge_mutex.lock();
                if (d.first_err == null) d.first_err = e;
                d.merge_mutex.unlock();
                return;
            };
            defer d.allocator.free(got_ids_local);

            var local_recall: f64 = 0;

            while (true) {
                const qi = d.cursor.fetchAdd(1, .monotonic);
                if (qi >= d.num_queries) break;

                const q = d.query_buf[qi * d.dim ..][0..d.dim];
                var op = std.time.Timer.start() catch unreachable;
                const results = c.searchVec(q, @intCast(d.top_k), @intCast(d.ef_search)) catch |e| {
                    d.merge_mutex.lock();
                    if (d.first_err == null) d.first_err = e;
                    d.merge_mutex.unlock();
                    return;
                };
                local_hist.record(op.read());

                if (d.truth_ids) |truth| {
                    for (results, 0..) |r, i| got_ids_local[i] = r.id;
                    const this_truth = truth[qi * d.top_k .. (qi + 1) * d.top_k];
                    local_recall += bruteforce.computeRecall(this_truth, got_ids_local[0..results.len]);
                }
                c.freeSearchResults(results);
            }

            d.merge_mutex.lock();
            defer d.merge_mutex.unlock();
            d.merged_hist.merge(&local_hist);
            d.merged_recall_sum += local_recall;
        }
    };

    var search_wall = try std.time.Timer.start();

    const threads = try allocator.alloc(std.Thread, n_clients);
    defer allocator.free(threads);
    for (threads) |*t| {
        t.* = try std.Thread.spawn(.{}, WorkerFn.run, .{&driver});
    }
    for (threads) |t| t.join();

    const search_wall_ns = search_wall.read();
    if (driver.first_err) |e| return e;

    const recall_sum = driver.merged_recall_sum;
    const search_stats = phaseStatsFrom(&search_hist, search_wall_ns);

    const recall: ?f32 = if (opts.validate)
        @as(f32, @floatCast(recall_sum / @as(f64, @floatFromInt(opts.num_queries))))
    else
        null;

    const report: Report = .{
        .opts = opts,
        .build_mode = @tagName(builtin.mode),
        .build = build_stats,
        .upper_used = h.index.upper_used,
        .search = search_stats,
        .recall_at_k = recall,
    };

    if (opts.json) try writeJson(out, report) else try writePretty(out, report);
    try out.flush();
}

// Two phases, both reported with the same PhaseStats shape:
//   1. PING RTT  — bounds the protocol floor (syscalls + framing).
//   2. SEARCH_VEC RTT with 1 stored vector — bounds floor + minimal
//      HNSW work.

fn runProtocolFloor(allocator: std.mem.Allocator, opts: Options, out: *std.io.Writer) !void {
    std.debug.assert(opts.dim > 0);
    std.debug.assert(opts.num_queries > 0);

    if (builtin.mode == .Debug) {
        std.debug.print(
            \\WARNING: benchmark compiled in Debug mode. Results are NOT meaningful.
            \\         Rebuild with `zig build -Doptimize=ReleaseFast` for real numbers.
            \\
        , .{});
    }

    var h: TcpHarness = undefined;
    var probe_opts = opts;
    probe_opts.num_vectors = @max(opts.num_vectors, 1);
    try h.setUp(allocator, probe_opts);
    defer h.tearDown();
    try h.start();

    const addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port);
    var client = try client_mod.Client.connect(allocator, addr, .{});
    defer client.deinit();

    // Prime the store so SEARCH_VEC has something to return.
    const vec = try allocator.alloc(f32, opts.dim);
    defer allocator.free(vec);
    {
        var prng = std.Random.DefaultPrng.init(opts.seed);
        const random = prng.random();
        for (vec) |*x| x.* = random.float(f32) * 2.0 - 1.0;
        bruteforce.normalize(vec);
    }
    _ = try client.insertVec(vec);

    // Warmup
    var k: usize = 0;
    while (k < opts.warmup) : (k += 1) try client.ping();

    var ping_hist = try Histogram.init(allocator, SEARCH_BUCKETS, SEARCH_BUCKET_NS);
    defer ping_hist.deinit();
    var ping_wall = try std.time.Timer.start();
    var i: usize = 0;
    while (i < opts.num_queries) : (i += 1) {
        var op = try std.time.Timer.start();
        try client.ping();
        ping_hist.record(op.read());
    }
    const ping_wall_ns = ping_wall.read();
    const ping_stats = phaseStatsFrom(&ping_hist, ping_wall_ns);

    var search_hist = try Histogram.init(allocator, SEARCH_BUCKETS, SEARCH_BUCKET_NS);
    defer search_hist.deinit();
    var search_wall = try std.time.Timer.start();
    i = 0;
    while (i < opts.num_queries) : (i += 1) {
        var op = try std.time.Timer.start();
        const r = try client.searchVec(vec, 1, @intCast(opts.ef_search));
        search_hist.record(op.read());
        client.freeSearchResults(r);
    }
    const search_wall_ns = search_wall.read();
    const search_stats = phaseStatsFrom(&search_hist, search_wall_ns);

    // Build a Report that reuses the existing formatters. `build`
    // carries the PING stats; `search` carries the SEARCH stats.
    var floor_opts = probe_opts;
    floor_opts.num_vectors = 1; // PhaseStats report 'inserts' as PING count in this mode
    floor_opts.transport = .tcp;
    const report: Report = .{
        .opts = floor_opts,
        .build_mode = @tagName(builtin.mode),
        .build = ping_stats,
        .upper_used = h.index.upper_used,
        .search = search_stats,
        .recall_at_k = null,
    };

    if (opts.json) {
        try out.writeAll("# NOTE: bench_protocol — 'build' phase is PING RTT, 'search' is SEARCH_VEC RTT.\n");
        try writeJson(out, report);
    } else {
        try out.writeAll("hnswz benchmark — protocol floor (transport=tcp)\n");
        try out.writeAll("  'Build phase' below is PING RTT; 'Search phase' is SEARCH_VEC RTT.\n\n");
        try writePretty(out, report);
    }
    try out.flush();
}

const testing = std.testing;

test "Histogram records and returns percentiles" {
    var h = try Histogram.init(testing.allocator, 1000, 1);
    defer h.deinit();

    // Record values 1..100 ns.
    for (1..101) |v| h.record(@intCast(v));
    try testing.expectEqual(@as(u64, 100), h.samples);

    // p50 should fall around the 50th sample (bucket index 49 or 50).
    const p50 = h.percentileNs(0.50);
    try testing.expect(p50 >= 49 and p50 <= 51);

    const p99 = h.percentileNs(0.99);
    try testing.expect(p99 >= 98 and p99 <= 100);

    try testing.expect(!h.overflowed());
    try testing.expectEqual(@as(u64, 100), h.max_sample_ns);
}

test "Histogram overflow detected when sample exceeds range" {
    var h = try Histogram.init(testing.allocator, 10, 1_000); // 0..10 µs
    defer h.deinit();
    h.record(5_000);
    try testing.expect(!h.overflowed());
    h.record(50_000);
    try testing.expect(h.overflowed());
}

test "run benchmark with tiny params produces JSON" {
    var buf: [8192]u8 = undefined;
    var fixed = std.io.Writer.fixed(&buf);

    const opts: Options = .{
        .num_vectors = 50,
        .num_queries = 10,
        .dim = 16,
        .ef_construction = 20,
        .ef_search = 20,
        .max_ef = 20,
        .top_k = 5,
        .seed = 1,
        .warmup = 2,
        .validate = true,
        .json = true,
        .upper_pool_slots = 50,
    };
    try run(testing.allocator, opts, &fixed);

    const out = buf[0..fixed.end];
    try testing.expect(out.len > 0);
    try testing.expectEqual(@as(u8, '{'), out[0]);
    try testing.expect(std.mem.indexOf(u8, out, "\"schema_version\": 1") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"recall_at_k\":") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"build\":") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"search\":") != null);
}

test "run benchmark with tiny params produces pretty output" {
    var buf: [8192]u8 = undefined;
    var fixed = std.io.Writer.fixed(&buf);

    const opts: Options = .{
        .num_vectors = 50,
        .num_queries = 10,
        .dim = 16,
        .ef_construction = 20,
        .ef_search = 20,
        .max_ef = 20,
        .top_k = 5,
        .seed = 1,
        .warmup = 0,
        .validate = false,
        .json = false,
        .upper_pool_slots = 50,
    };
    try run(testing.allocator, opts, &fixed);

    const out = buf[0..fixed.end];
    try testing.expect(std.mem.indexOf(u8, out, "Build phase") != null);
    try testing.expect(std.mem.indexOf(u8, out, "Search phase") != null);
    try testing.expect(std.mem.indexOf(u8, out, "recall@") == null); // validate=false
}

test "run benchmark over TCP transport with tiny params" {
    var buf: [16384]u8 = undefined;
    var fixed = std.io.Writer.fixed(&buf);

    const opts: Options = .{
        .num_vectors = 30,
        .num_queries = 10,
        .dim = 16,
        .ef_construction = 20,
        .ef_search = 20,
        .max_ef = 20,
        .top_k = 5,
        .seed = 7,
        .warmup = 2,
        .validate = true,
        .json = false,
        .upper_pool_slots = 30,
        .transport = .tcp,
        .bench_protocol = false,
    };
    try run(testing.allocator, opts, &fixed);

    const out = buf[0..fixed.end];
    try testing.expect(std.mem.indexOf(u8, out, "transport  tcp") != null);
    try testing.expect(std.mem.indexOf(u8, out, "Build phase") != null);
    try testing.expect(std.mem.indexOf(u8, out, "Search phase") != null);
    try testing.expect(std.mem.indexOf(u8, out, "recall@") != null);
}

test "bench_protocol produces PING and SEARCH phase stats" {
    var buf: [16384]u8 = undefined;
    var fixed = std.io.Writer.fixed(&buf);

    const opts: Options = .{
        .num_vectors = 1,
        .num_queries = 20,
        .dim = 8,
        .ef_construction = 10,
        .ef_search = 10,
        .max_ef = 10,
        .top_k = 1,
        .seed = 9,
        .warmup = 2,
        .validate = false,
        .json = false,
        .upper_pool_slots = 2,
        .transport = .tcp,
        .bench_protocol = true,
    };
    try run(testing.allocator, opts, &fixed);

    const out = buf[0..fixed.end];
    try testing.expect(std.mem.indexOf(u8, out, "protocol floor") != null);
    try testing.expect(std.mem.indexOf(u8, out, "PING RTT") != null);
    try testing.expect(std.mem.indexOf(u8, out, "SEARCH_VEC RTT") != null);
}
