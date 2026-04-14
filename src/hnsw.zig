const std = @import("std");
const distance = @import("distance.zig");
const Store = @import("store.zig").Store;
const heap = @import("heap.zig");
const bruteforce = @import("bruteforce.zig");

pub fn HnswIndex(comptime dim: usize, comptime M: usize) type {
    const M0 = 2 * M;
    const ml: f64 = 1.0 / @log(@as(f64, @floatFromInt(M)));

    const Neighbors = struct {
        buf: [M0]u32 = @splat(0),
        list: std.ArrayList(u32) = .{},

        /// Must be called via pointer after final placement in heap memory.
        fn initInPlace(self: *@This()) void {
            self.list = std.ArrayList(u32).initBuffer(&self.buf);
        }

        fn append(self: *@This(), id: u32, cap: usize) void {
            if (self.list.items.len < cap) {
                self.list.appendAssumeCapacity(id);
            }
        }

        fn count(self: *const @This()) usize {
            return self.list.items.len;
        }

        fn slice(self: *const @This()) []const u32 {
            return self.list.items;
        }

        fn clear(self: *@This()) void {
            self.list.items.len = 0;
        }
    };

    const Node = struct {
        level: u8,
        layers: []Neighbors,

        fn init(allocator: std.mem.Allocator, level: u8) !@This() {
            const num_layers: usize = @as(usize, level) + 1;
            const layers = try allocator.alloc(Neighbors, num_layers);
            for (layers) |*layer| {
                layer.* = .{};
                layer.initInPlace();
            }
            return .{ .level = level, .layers = layers };
        }

        fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.layers);
        }
    };

    return struct {
        const Self = @This();

        nodes: std.ArrayList(Node),
        entry_point: ?u32 = null,
        max_level: u8 = 0,
        store: *Store(dim),
        ef_construction: usize,
        rng: std.Random.DefaultPrng,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, store: *Store(dim), ef_construction: usize, seed: u64) Self {
            return .{
                .nodes = .{},
                .store = store,
                .ef_construction = ef_construction,
                .rng = std.Random.DefaultPrng.init(seed),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.nodes.items) |*node| {
                node.deinit(self.allocator);
            }
            self.nodes.deinit(self.allocator);
        }

        pub fn randomLevel(self: *Self) u8 {
            const random = self.rng.random();
            const uni = 1.0 - random.float(f64); // (0, 1]
            const l: f64 = @floor(-@log(uni) * ml);
            const capped: u8 = @intFromFloat(@min(l, 15.0));
            return capped;
        }

        fn distanceTo(self: *Self, a: u32, b: []const f32) f32 {
            return distance.cosineNormalized(self.store.get(a), b);
        }

        fn distanceBetween(self: *Self, a: u32, b: u32) f32 {
            return distance.cosineNormalized(self.store.get(a), self.store.get(b));
        }

        fn layerCap(layer: usize) usize {
            return if (layer == 0) M0 else M;
        }

        fn greedyClosest(self: *Self, query: []const f32, entry_id: u32, layer: usize) u32 {
            var current = entry_id;
            var current_dist = self.distanceTo(current, query);

            while (true) {
                var changed = false;
                const neighbors = self.nodes.items[current].layers[layer].slice();
                for (neighbors) |neighbor| {
                    const d = self.distanceTo(neighbor, query);
                    if (d < current_dist) {
                        current = neighbor;
                        current_dist = d;
                        changed = true;
                    }
                }
                if (!changed) break;
            }
            return current;
        }

        fn beamSearch(self: *Self, query: []const f32, entry_id: u32, layer: usize, ef: usize) ![]heap.Entry {
            const n = self.nodes.items.len;
            const visited = try self.allocator.alloc(bool, n);
            defer self.allocator.free(visited);
            @memset(visited, false);

            var candidates = heap.MinHeap.init(self.allocator, {});
            defer candidates.deinit();
            var results = heap.MaxHeap.init(self.allocator, {});
            defer results.deinit();

            const entry_dist = self.distanceTo(entry_id, query);
            try candidates.add(.{ .id = entry_id, .dist = entry_dist });
            try results.add(.{ .id = entry_id, .dist = entry_dist });
            visited[entry_id] = true;

            while (candidates.count() > 0) {
                const closest = candidates.remove();

                // If closest candidate is farther than the farthest result, stop
                if (results.count() >= ef) {
                    if (closest.dist > results.peek().?.dist) break;
                }

                const neighbors = self.nodes.items[closest.id].layers[layer].slice();
                for (neighbors) |neighbor| {
                    if (visited[neighbor]) continue;
                    visited[neighbor] = true;

                    const d = self.distanceTo(neighbor, query);

                    if (results.count() < ef or d < results.peek().?.dist) {
                        try candidates.add(.{ .id = neighbor, .dist = d });
                        try results.add(.{ .id = neighbor, .dist = d });

                        if (results.count() > ef) {
                            _ = results.remove();
                        }
                    }
                }
            }

            // Extract results sorted by distance ascending
            const count = results.count();
            const sorted = try self.allocator.alloc(heap.Entry, count);
            var i: usize = count;
            while (results.count() > 0) {
                i -= 1;
                sorted[i] = results.remove();
            }
            return sorted;
        }

        fn selectNeighborsHeuristic(self: *Self, target: u32, candidates: []const heap.Entry, m: usize) ![]heap.Entry {
            var selected: std.ArrayList(heap.Entry) = .{};
            defer selected.deinit(self.allocator);

            for (candidates) |c| {
                if (selected.items.len >= m) break;
                if (c.id == target) continue;

                var accept = true;
                for (selected.items) |s| {
                    const dist_s_c = self.distanceBetween(s.id, c.id);
                    if (dist_s_c < c.dist) {
                        accept = false;
                        break;
                    }
                }
                if (accept) {
                    try selected.append(self.allocator, c);
                }
            }

            return try selected.toOwnedSlice(self.allocator);
        }

        fn connectNeighbors(self: *Self, node_id: u32, neighbors: []const heap.Entry, layer: usize) !void {
            const cap = layerCap(layer);

            // Forward edges: node_id -> each neighbor
            for (neighbors) |n| {
                self.nodes.items[node_id].layers[layer].append(n.id, cap);
            }

            // Reverse edges: each neighbor -> node_id
            for (neighbors) |n| {
                var neighbor_list = &self.nodes.items[n.id].layers[layer];
                if (neighbor_list.count() < cap) {
                    neighbor_list.append(node_id, cap);
                } else {
                    // Prune: rebuild neighbor's list using heuristic with neighbor as center
                    try self.pruneNeighbor(n.id, node_id, layer, cap);
                }
            }
        }

        fn pruneNeighbor(self: *Self, neighbor_id: u32, new_id: u32, layer: usize, cap: usize) !void {
            const neighbor_node = &self.nodes.items[neighbor_id];
            const old_neighbors = neighbor_node.layers[layer].slice();

            // Build candidate list: old neighbors + new_id, with distances relative to neighbor_id
            var cands = try self.allocator.alloc(heap.Entry, old_neighbors.len + 1);
            defer self.allocator.free(cands);

            for (old_neighbors, 0..) |nb, i| {
                cands[i] = .{ .id = nb, .dist = self.distanceBetween(neighbor_id, nb) };
            }
            cands[old_neighbors.len] = .{ .id = new_id, .dist = self.distanceBetween(neighbor_id, new_id) };

            // Sort by distance ascending
            std.mem.sort(heap.Entry, cands, {}, entryLessThan);

            // Select using heuristic
            const selected = try self.selectNeighborsHeuristic(neighbor_id, cands, cap);
            defer self.allocator.free(selected);

            // Rebuild the neighbor list
            neighbor_node.layers[layer].clear();
            for (selected) |s| {
                neighbor_node.layers[layer].append(s.id, cap);
            }
        }

        fn entryLessThan(_: void, a: heap.Entry, b: heap.Entry) bool {
            return a.dist < b.dist;
        }

        pub fn insert(self: *Self, id: u32) !void {
            const level = self.randomLevel();
            var node = try Node.init(self.allocator, level);
            errdefer node.deinit(self.allocator);

            // Ensure nodes list is large enough (ids must be sequential)
            while (self.nodes.items.len <= id) {
                try self.nodes.append(self.allocator, try Node.init(self.allocator, 0));
            }
            // Replace the placeholder at position id
            if (self.nodes.items[id].layers.ptr != node.layers.ptr) {
                self.nodes.items[id].deinit(self.allocator);
            }
            self.nodes.items[id] = node;

            if (self.entry_point == null) {
                self.entry_point = id;
                self.max_level = level;
                return;
            }

            var ep = self.entry_point.?;

            // Greedy descent from max_level down to level+1
            var l: usize = self.max_level;
            while (l > @as(usize, level) + 1) : (l -= 1) {
                ep = self.greedyClosest(self.store.get(id), ep, l);
            }
            if (l > @as(usize, level)) {
                ep = self.greedyClosest(self.store.get(id), ep, l);
            }

            // Build edges from min(level, max_level) down to 0
            const top_layer = @min(@as(usize, level), @as(usize, self.max_level));
            var layer_i: usize = top_layer + 1;
            while (layer_i > 0) {
                layer_i -= 1;

                const results = try self.beamSearch(self.store.get(id), ep, layer_i, self.ef_construction);
                defer self.allocator.free(results);

                const cap = layerCap(layer_i);
                const selected = try self.selectNeighborsHeuristic(id, results, cap);
                defer self.allocator.free(selected);

                try self.connectNeighbors(id, selected, layer_i);

                // Update entry point for next layer to closest found
                if (results.len > 0) {
                    ep = results[0].id;
                }
            }

            // Update entry point if new node has higher level
            if (level > self.max_level) {
                self.entry_point = id;
                self.max_level = level;
            }
        }

        pub fn search(self: *Self, query: *const [dim]f32, k: usize, ef_search: usize) ![]heap.Entry {
            if (self.entry_point == null) return try self.allocator.alloc(heap.Entry, 0);

            var ep = self.entry_point.?;

            // Greedy descent from top layer to layer 1
            var l: usize = self.max_level;
            while (l > 0) : (l -= 1) {
                ep = self.greedyClosest(query, ep, l);
            }

            // Beam search at layer 0
            const results = try self.beamSearch(query, ep, 0, ef_search);

            // Return top-k (results already sorted ascending)
            const actual_k = @min(k, results.len);
            if (actual_k == results.len) return results;

            // Shrink to top-k
            const top_k = try self.allocator.alloc(heap.Entry, actual_k);
            @memcpy(top_k, results[0..actual_k]);
            self.allocator.free(results);
            return top_k;
        }
    };
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "randomLevel distribution" {
    const Index = HnswIndex(4, 16);
    var store = try Store(4).init(testing.allocator, 1);
    defer store.deinit(testing.allocator);

    var idx = Index.init(testing.allocator, &store, 200, 42);
    defer idx.deinit();

    const samples = 100_000;
    var counts = [_]u32{0} ** 16;

    for (0..samples) |_| {
        const level = idx.randomLevel();
        counts[level] += 1;
    }

    const total: f32 = @floatFromInt(samples);

    // Layer 0: ~93-95%
    const pct0 = @as(f32, @floatFromInt(counts[0])) / total;
    try testing.expect(pct0 > 0.90 and pct0 < 0.97);

    // Layer 1: ~5-7%
    const pct1 = @as(f32, @floatFromInt(counts[1])) / total;
    try testing.expect(pct1 > 0.03 and pct1 < 0.10);

    // Layer 2: ~0.1-1%
    const pct2 = @as(f32, @floatFromInt(counts[2])) / total;
    try testing.expect(pct2 > 0.001 and pct2 < 0.01);

    // Layer 3+: < 0.1%
    var sum_3plus: u32 = 0;
    for (3..16) |i| {
        sum_3plus += counts[i];
    }
    const pct3plus = @as(f32, @floatFromInt(sum_3plus)) / total;
    try testing.expect(pct3plus < 0.001);

    std.debug.print("\nLevel distribution: L0={d:.2}% L1={d:.2}% L2={d:.4}% L3+={d:.4}%\n", .{
        pct0 * 100, pct1 * 100, pct2 * 100, pct3plus * 100,
    });
}

test "recall@10 with heuristic neighbor selection (ef_search=50)" {
    const dim = 128;
    const num_vectors = 10_000;
    const num_queries = 100;
    const k = 10;
    const seed: u64 = 42;
    const M_val = 16;

    const S = Store(dim);
    var store = try S.init(testing.allocator, num_vectors);
    defer store.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_vectors) |_| {
        var vec: [dim]f32 = undefined;
        for (&vec) |*x| {
            x.* = random.float(f32) * 2.0 - 1.0;
        }
        bruteforce.normalize(&vec);
        _ = try store.add(vec);
    }

    // Build HNSW index
    var idx = HnswIndex(dim, M_val).init(testing.allocator, &store, 200, 123);
    defer idx.deinit();

    for (0..num_vectors) |i| {
        try idx.insert(@intCast(i));
    }

    // Compute ground truth
    var truth_ids: [num_queries][k]u32 = undefined;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const truth = try bruteforce.search(dim, &store, query, k, testing.allocator);
        defer testing.allocator.free(truth);
        for (0..k) |i| {
            truth_ids[qi][i] = truth[i].id;
        }
    }

    // Test recall@10 with ef_search=50
    var total_recall: f32 = 0;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const results = try idx.search(query, k, 50);
        defer testing.allocator.free(results);

        var result_ids: [k]u32 = undefined;
        const actual_k = @min(k, results.len);
        for (0..actual_k) |i| {
            result_ids[i] = results[i].id;
        }

        total_recall += bruteforce.computeRecall(&truth_ids[qi], result_ids[0..actual_k]);
    }

    const avg_recall = total_recall / @as(f32, @floatFromInt(num_queries));
    std.debug.print("\nrecall@{d} (ef_search=50): {d:.4}\n", .{ k, avg_recall });
    try testing.expect(avg_recall >= 0.85);
}

test "recall@10 with heuristic neighbor selection (ef_search=100)" {
    const dim = 128;
    const num_vectors = 10_000;
    const num_queries = 100;
    const k = 10;
    const seed: u64 = 42;
    const M_val = 16;

    const S = Store(dim);
    var store = try S.init(testing.allocator, num_vectors);
    defer store.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_vectors) |_| {
        var vec: [dim]f32 = undefined;
        for (&vec) |*x| {
            x.* = random.float(f32) * 2.0 - 1.0;
        }
        bruteforce.normalize(&vec);
        _ = try store.add(vec);
    }

    var idx = HnswIndex(dim, M_val).init(testing.allocator, &store, 200, 123);
    defer idx.deinit();

    for (0..num_vectors) |i| {
        try idx.insert(@intCast(i));
    }

    var truth_ids: [num_queries][k]u32 = undefined;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const truth = try bruteforce.search(dim, &store, query, k, testing.allocator);
        defer testing.allocator.free(truth);
        for (0..k) |i| {
            truth_ids[qi][i] = truth[i].id;
        }
    }

    var total_recall: f32 = 0;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const results = try idx.search(query, k, 100);
        defer testing.allocator.free(results);

        var result_ids: [k]u32 = undefined;
        const actual_k = @min(k, results.len);
        for (0..actual_k) |i| {
            result_ids[i] = results[i].id;
        }

        total_recall += bruteforce.computeRecall(&truth_ids[qi], result_ids[0..actual_k]);
    }

    const avg_recall = total_recall / @as(f32, @floatFromInt(num_queries));
    std.debug.print("\nrecall@{d} (ef_search=100): {d:.4}\n", .{ k, avg_recall });
    try testing.expect(avg_recall >= 0.95);
}
