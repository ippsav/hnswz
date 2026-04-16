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

/// Must match the M baked into the shipped binary (see main.zig).
pub const M: usize = 16;

// Linear histograms; last bucket is the overflow catch-all.
// Search latency is typically sub-ms → 1 µs buckets × 20 000 = 20 ms range.
// Insert latency can spike into ms → 10 µs buckets × 20 000 = 200 ms range.
const SEARCH_BUCKET_NS: u64 = 1_000;
const SEARCH_BUCKETS: usize = 20_000;
const INSERT_BUCKET_NS: u64 = 10_000;
const INSERT_BUCKETS: usize = 20_000;

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

    // ── setup (untimed) ─────────────────────────────────────────────

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

    // ── build phase ─────────────────────────────────────────────────

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

    // ── search phase ────────────────────────────────────────────────

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

// ── output formats ────────────────────────────────────────────────────

fn writePretty(w: *std.io.Writer, r: Report) !void {
    try w.print(
        \\hnswz benchmark
        \\  mode       {s}
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
        r.build_mode, r.opts.seed, r.opts.dim,   M,
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

// ── tests ──────────────────────────────────────────────────────────────

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
