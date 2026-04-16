const std = @import("std");
const testing = std.testing;

const distance = @import("distance.zig");
const Store = @import("store.zig").Store;

pub const Result = struct {
    id: u32,
    dist: f32,
};

fn resultLessThan(_: void, a: Result, b: Result) bool {
    return a.dist < b.dist;
}

/// Brute-force top-k nearest neighbor search. Computes cosine distance
/// (for pre-normalized vectors) against every vector in the store,
/// returns the k nearest sorted by distance ascending.
/// Caller owns the returned slice.
pub fn search(
    store: *const Store,
    query: []const f32,
    k: usize,
    allocator: std.mem.Allocator,
) ![]Result {
    std.debug.assert(query.len == store.dim);
    const n = store.count;

    const all = try allocator.alloc(Result, n);
    defer allocator.free(all);

    for (0..n) |i| {
        const id: u32 = @intCast(i);
        all[i] = .{
            .id = id,
            .dist = distance.cosineNormalized(query, store.get(id)),
        };
    }

    std.mem.sort(Result, all, {}, resultLessThan);

    const actual_k = @min(k, n);
    const top_k = try allocator.alloc(Result, actual_k);
    @memcpy(top_k, all[0..actual_k]);
    return top_k;
}

/// Normalize a vector to unit length (L2 norm = 1).
pub fn normalize(v: []f32) void {
    const len = @sqrt(distance.dot(v, v));
    if (len == 0) return;
    for (v) |*x| {
        x.* /= len;
    }
}

/// Compute recall: fraction of truth IDs present in predicted IDs.
pub fn computeRecall(truth: []const u32, predicted: []const u32) f32 {
    var hits: u32 = 0;
    for (predicted) |p| {
        for (truth) |t| {
            if (p == t) {
                hits += 1;
                break;
            }
        }
    }
    return @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(truth.len));
}

test "brute force returns nearest vectors in order" {
    var store = try Store.init(testing.allocator, 4, 10);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 0.0, 0.0, 0.0 }); // id 0  -- identical to query
    _ = try store.add(&[_]f32{ 0.0, 1.0, 0.0, 0.0 }); // id 1  -- orthogonal
    _ = try store.add(&[_]f32{ 0.707, 0.707, 0.0, 0.0 }); // id 2  -- 45 degrees
    _ = try store.add(&[_]f32{ 0.0, 0.0, 1.0, 0.0 }); // id 3  -- orthogonal
    _ = try store.add(&[_]f32{ 0.0, 0.0, 0.0, 1.0 }); // id 4  -- orthogonal

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try search(&store, &query, 3, testing.allocator);
    defer testing.allocator.free(results);

    try testing.expectEqual(3, results.len);
    try testing.expectEqual(0, results[0].id); // identical vector
    try testing.expectEqual(2, results[1].id); // 0.707 dot product
}

test "brute force with k > n returns all vectors" {
    var store = try Store.init(testing.allocator, 2, 3);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 0.0 });
    _ = try store.add(&[_]f32{ 0.0, 1.0 });

    const query = [_]f32{ 1.0, 0.0 };
    const results = try search(&store, &query, 10, testing.allocator);
    defer testing.allocator.free(results);

    try testing.expectEqual(2, results.len);
}

test "recall@10: brute force vs itself = 1.0" {
    const dim = 1024;
    const num_vectors = 10_000;
    const num_queries = 100;
    const k = 10;
    const seed: u64 = 42;

    var store = try Store.init(testing.allocator, dim, num_vectors);
    defer store.deinit(testing.allocator);

    // Deterministic PRNG — same seed = same dataset every run
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    // Generate random unit vectors and insert
    for (0..num_vectors) |_| {
        var vec: [dim]f32 = undefined;
        for (&vec) |*x| {
            x.* = random.float(f32) * 2.0 - 1.0;
        }
        normalize(&vec);
        _ = try store.add(&vec);
    }

    // Phase 1: compute ground truth (brute-force top-k for each query)
    var truth_ids: [num_queries][k]u32 = undefined;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const truth = try search(&store, query, k, testing.allocator);
        defer testing.allocator.free(truth);
        for (0..k) |i| {
            truth_ids[qi][i] = truth[i].id;
        }
    }

    // Phase 2: run algorithm under test (brute force again → recall must be 1.0)
    var total_recall: f32 = 0;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const results = try search(&store, query, k, testing.allocator);
        defer testing.allocator.free(results);

        var result_ids: [k]u32 = undefined;
        for (0..k) |i| {
            result_ids[i] = results[i].id;
        }

        total_recall += computeRecall(&truth_ids[qi], &result_ids);
    }

    const avg_recall = total_recall / @as(f32, @floatFromInt(num_queries));
    std.debug.print("\nrecall@{d}: {d:.4}\n", .{ k, avg_recall });
    try testing.expectApproxEqAbs(1.0, avg_recall, 1e-6);
}
