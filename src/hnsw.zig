const std = @import("std");
const distance = @import("distance.zig");
const Store = @import("store.zig").Store;
const heap = @import("heap.zig");
const bruteforce = @import("bruteforce.zig");

pub const Params = struct {
    /// Capacity for the per-node flat arrays. Must be >= the highest node id
    /// ever inserted. For `load`, must be >= file's node_count.
    max_vectors: usize,
    /// Capacity for the upper-layer neighbor pool in "slots" (one slot == M neighbors).
    /// Each inserted node with level=L consumes L slots. Sum across all nodes
    /// must fit. Expected mean is N / (M-1); be generous.
    max_upper_slots: usize,
    ef_construction: usize,
    seed: u64 = 42,
};

pub const LoadSize = struct {
    node_count: usize,
    upper_slots: usize,
};

pub fn HnswIndex(comptime M: usize) type {
    const M0 = 2 * M;
    const ml: f64 = 1.0 / @log(@as(f64, @floatFromInt(M)));
    comptime std.debug.assert(M > 0);

    return struct {
        const Self = @This();

        /// Fixed-capacity scratch space used by insert/search. Not owned by
        /// the index — callers check one out of a pool per request.
        pub const Workspace = struct {
            visited: std.DynamicBitSetUnmanaged,
            candidates_buf: []heap.Entry, // sized to max_vectors — peak candidates ≤ |visited|
            results_buf: []heap.Entry, // sized to max_ef — strictly capped
            output: []heap.Entry, // sized to max_ef
            select_scratch: []heap.Entry, // sized to M0
            prune_scratch: []heap.Entry, // sized to M0 + 1

            pub fn init(allocator: std.mem.Allocator, max_vectors: usize, max_ef: usize) !Workspace {
                std.debug.assert(max_ef > 0);
                std.debug.assert(max_vectors > 0);
                var vis = try std.DynamicBitSetUnmanaged.initEmpty(allocator, max_vectors);
                errdefer vis.deinit(allocator);

                const candidates_buf = try allocator.alloc(heap.Entry, max_vectors);
                errdefer allocator.free(candidates_buf);
                const results_buf = try allocator.alloc(heap.Entry, max_ef);
                errdefer allocator.free(results_buf);
                const output = try allocator.alloc(heap.Entry, max_ef);
                errdefer allocator.free(output);
                const select_scratch = try allocator.alloc(heap.Entry, M0);
                errdefer allocator.free(select_scratch);
                const prune_scratch = try allocator.alloc(heap.Entry, M0 + 1);

                return .{
                    .visited = vis,
                    .candidates_buf = candidates_buf,
                    .results_buf = results_buf,
                    .output = output,
                    .select_scratch = select_scratch,
                    .prune_scratch = prune_scratch,
                };
            }

            pub fn deinit(self: *Workspace, allocator: std.mem.Allocator) void {
                self.visited.deinit(allocator);
                allocator.free(self.candidates_buf);
                allocator.free(self.results_buf);
                allocator.free(self.output);
                allocator.free(self.select_scratch);
                allocator.free(self.prune_scratch);
            }

            fn resetVisited(self: *Workspace) void {
                self.visited.unsetAll();
            }
        };

        // Per-node flat arrays (length = max_vectors).
        levels: []u8,
        layer0_neighbors: []u32, // max_vectors × M0
        layer0_lens: []u16, // max_vectors
        upper_offset: []u32, // max_vectors — index (in slots) into upper_neighbors
        upper_count: []u8, // max_vectors — number of upper layers (== level)

        // Upper-layer pool (shared across all nodes, bump-allocated).
        upper_neighbors: []u32, // max_upper_slots × M
        upper_lens: []u16, // max_upper_slots

        entry_point: ?u32 = null,
        max_level: u8 = 0,
        node_count: usize = 0,
        upper_used: usize = 0,

        store: *const Store,
        ef_construction: usize,
        rng: std.Random.DefaultPrng,
        allocator: std.mem.Allocator,
        max_vectors: usize,
        max_upper_slots: usize,

        pub fn init(allocator: std.mem.Allocator, store: *const Store, params: Params) !Self {
            std.debug.assert(params.max_vectors > 0);

            const levels = try allocator.alloc(u8, params.max_vectors);
            errdefer allocator.free(levels);
            const layer0_neighbors = try allocator.alloc(u32, params.max_vectors * M0);
            errdefer allocator.free(layer0_neighbors);
            const layer0_lens = try allocator.alloc(u16, params.max_vectors);
            errdefer allocator.free(layer0_lens);
            const upper_offset = try allocator.alloc(u32, params.max_vectors);
            errdefer allocator.free(upper_offset);
            const upper_count = try allocator.alloc(u8, params.max_vectors);
            errdefer allocator.free(upper_count);

            const upper_neighbors = if (params.max_upper_slots == 0)
                try allocator.alloc(u32, 0)
            else
                try allocator.alloc(u32, params.max_upper_slots * M);
            errdefer allocator.free(upper_neighbors);
            const upper_lens = try allocator.alloc(u16, params.max_upper_slots);

            @memset(levels, 0);
            @memset(layer0_lens, 0);
            @memset(upper_offset, 0);
            @memset(upper_count, 0);
            @memset(upper_lens, 0);

            return .{
                .levels = levels,
                .layer0_neighbors = layer0_neighbors,
                .layer0_lens = layer0_lens,
                .upper_offset = upper_offset,
                .upper_count = upper_count,
                .upper_neighbors = upper_neighbors,
                .upper_lens = upper_lens,
                .store = store,
                .ef_construction = params.ef_construction,
                .rng = std.Random.DefaultPrng.init(params.seed),
                .allocator = allocator,
                .max_vectors = params.max_vectors,
                .max_upper_slots = params.max_upper_slots,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.levels);
            self.allocator.free(self.layer0_neighbors);
            self.allocator.free(self.layer0_lens);
            self.allocator.free(self.upper_offset);
            self.allocator.free(self.upper_count);
            self.allocator.free(self.upper_neighbors);
            self.allocator.free(self.upper_lens);
        }

        // ── accessor helpers ────────────────────────────────────────────

        const LayerView = struct {
            neighbors: []u32, // full slice of size M0 (layer 0) or M (upper)
            len: *u16,
        };

        fn layer(self: *Self, id: u32, layer_i: usize) LayerView {
            if (layer_i == 0) {
                return .{
                    .neighbors = self.layer0_neighbors[@as(usize, id) * M0 ..][0..M0],
                    .len = &self.layer0_lens[id],
                };
            }
            const slot = self.upper_offset[id] + @as(u32, @intCast(layer_i - 1));
            return .{
                .neighbors = self.upper_neighbors[@as(usize, slot) * M ..][0..M],
                .len = &self.upper_lens[slot],
            };
        }

        fn neighborsOf(self: *const Self, id: u32, layer_i: usize) []const u32 {
            if (layer_i == 0) {
                const len = self.layer0_lens[id];
                return self.layer0_neighbors[@as(usize, id) * M0 ..][0..len];
            }
            const slot = self.upper_offset[id] + @as(u32, @intCast(layer_i - 1));
            const len = self.upper_lens[slot];
            return self.upper_neighbors[@as(usize, slot) * M ..][0..len];
        }

        inline fn layerCap(layer_i: usize) usize {
            return if (layer_i == 0) M0 else M;
        }

        fn appendNeighbor(lv: LayerView, id: u32, cap: usize) void {
            const n = lv.len.*;
            if (@as(usize, n) < cap) {
                lv.neighbors[n] = id;
                lv.len.* = n + 1;
            }
        }

        fn clearNeighbors(lv: LayerView) void {
            lv.len.* = 0;
        }

        // ── distance shims ──────────────────────────────────────────────

        fn distanceTo(self: *const Self, a: u32, b: []const f32) f32 {
            return distance.cosineNormalized(self.store.get(a), b);
        }

        fn distanceBetween(self: *const Self, a: u32, b: u32) f32 {
            return distance.cosineNormalized(self.store.get(a), self.store.get(b));
        }

        // ── search primitives ───────────────────────────────────────────

        pub fn randomLevel(self: *Self) u8 {
            const random = self.rng.random();
            const uni = 1.0 - random.float(f64); // (0, 1]
            const l: f64 = @floor(-@log(uni) * ml);
            const capped: u8 = @intFromFloat(@min(l, 15.0));
            return capped;
        }

        fn greedyClosest(self: *const Self, query: []const f32, entry_id: u32, layer_i: usize) u32 {
            var current = entry_id;
            var current_dist = self.distanceTo(current, query);

            while (true) {
                var changed = false;
                const neighbors = self.neighborsOf(current, layer_i);
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

        /// Beam search on a given layer. Writes sorted-ascending results into
        /// `ws.output` and returns that slice.
        fn beamSearch(
            self: *const Self,
            ws: *Workspace,
            query: []const f32,
            entry_id: u32,
            layer_i: usize,
            ef: usize,
        ) ![]heap.Entry {
            std.debug.assert(ef <= ws.results_buf.len);
            std.debug.assert(ef <= ws.output.len);

            ws.resetVisited();
            // candidates is bounded by |visited| = max_vectors; use the full buffer.
            var candidates = heap.MinHeap.init(ws.candidates_buf);
            var results = heap.MaxHeap.init(ws.results_buf[0..ef]);

            const entry_dist = self.distanceTo(entry_id, query);
            try candidates.push(.{ .id = entry_id, .dist = entry_dist });
            try results.push(.{ .id = entry_id, .dist = entry_dist });
            ws.visited.set(entry_id);

            while (candidates.count() > 0) {
                const closest = candidates.pop();

                if (results.count() >= ef) {
                    if (closest.dist > results.peek().?.dist) break;
                }

                const neighbors = self.neighborsOf(closest.id, layer_i);
                for (neighbors) |neighbor| {
                    if (ws.visited.isSet(neighbor)) continue;
                    ws.visited.set(neighbor);

                    const d = self.distanceTo(neighbor, query);

                    if (results.count() < ef or d < results.peek().?.dist) {
                        // candidates must have room; it mirrors visits into
                        // results, which is bounded by ef.
                        try candidates.push(.{ .id = neighbor, .dist = d });
                        if (results.count() < ef) {
                            try results.push(.{ .id = neighbor, .dist = d });
                        } else {
                            // At capacity: replace root (largest) with the
                            // new, better entry.
                            _ = results.pop();
                            try results.push(.{ .id = neighbor, .dist = d });
                        }
                    }
                }
            }

            // Drain results (max-heap) into the output buffer sorted ascending.
            const n = results.count();
            var i: usize = n;
            while (results.count() > 0) {
                i -= 1;
                ws.output[i] = results.pop();
            }
            return ws.output[0..n];
        }

        fn entryLessThan(_: void, a: heap.Entry, b: heap.Entry) bool {
            return a.dist < b.dist;
        }

        /// Select up to `m` neighbors from `candidates` (already sorted
        /// ascending by distance to `target`) using the paper's heuristic.
        /// Writes into `ws.select_scratch` and returns that slice.
        fn selectNeighborsHeuristic(
            self: *const Self,
            ws: *Workspace,
            target: u32,
            candidates: []const heap.Entry,
            m: usize,
        ) []heap.Entry {
            std.debug.assert(m <= ws.select_scratch.len);
            var selected_len: usize = 0;

            for (candidates) |c| {
                if (selected_len >= m) break;
                if (c.id == target) continue;

                var accept = true;
                for (ws.select_scratch[0..selected_len]) |s| {
                    const dist_s_c = self.distanceBetween(s.id, c.id);
                    if (dist_s_c < c.dist) {
                        accept = false;
                        break;
                    }
                }
                if (accept) {
                    ws.select_scratch[selected_len] = c;
                    selected_len += 1;
                }
            }
            return ws.select_scratch[0..selected_len];
        }

        fn connectNeighbors(
            self: *Self,
            ws: *Workspace,
            node_id: u32,
            neighbors: []const heap.Entry,
            layer_i: usize,
        ) void {
            const cap = layerCap(layer_i);

            // Forward: node_id -> each neighbor
            const nv = self.layer(node_id, layer_i);
            for (neighbors) |n| appendNeighbor(nv, n.id, cap);

            // Reverse: each neighbor -> node_id, pruning if over cap
            for (neighbors) |n| {
                const nbr_view = self.layer(n.id, layer_i);
                if (@as(usize, nbr_view.len.*) < cap) {
                    appendNeighbor(nbr_view, node_id, cap);
                } else {
                    self.pruneNeighbor(ws, n.id, node_id, layer_i, cap);
                }
            }
        }

        fn pruneNeighbor(
            self: *Self,
            ws: *Workspace,
            neighbor_id: u32,
            new_id: u32,
            layer_i: usize,
            cap: usize,
        ) void {
            const nbr_view = self.layer(neighbor_id, layer_i);
            const old_len: usize = nbr_view.len.*;
            std.debug.assert(old_len + 1 <= ws.prune_scratch.len);

            for (nbr_view.neighbors[0..old_len], 0..) |nb, i| {
                ws.prune_scratch[i] = .{ .id = nb, .dist = self.distanceBetween(neighbor_id, nb) };
            }
            ws.prune_scratch[old_len] = .{ .id = new_id, .dist = self.distanceBetween(neighbor_id, new_id) };

            std.mem.sort(heap.Entry, ws.prune_scratch[0 .. old_len + 1], {}, entryLessThan);

            const selected = self.selectNeighborsHeuristic(ws, neighbor_id, ws.prune_scratch[0 .. old_len + 1], cap);

            clearNeighbors(nbr_view);
            for (selected) |s| appendNeighbor(nbr_view, s.id, cap);
        }

        // ── insert & search (public) ────────────────────────────────────

        pub fn insert(self: *Self, ws: *Workspace, id: u32) !void {
            // Sequential insertion only — ids must be assigned by store.add().
            std.debug.assert(@as(usize, id) == self.node_count);
            if (self.node_count >= self.max_vectors) return error.IndexFull;

            const level = self.randomLevel();
            // Reserve upper-layer slots for this node's upper layers.
            const upper_needed: usize = level;
            if (self.upper_used + upper_needed > self.max_upper_slots) return error.UpperPoolFull;

            self.levels[id] = level;
            self.upper_offset[id] = @intCast(self.upper_used);
            self.upper_count[id] = level;
            self.layer0_lens[id] = 0;
            for (0..upper_needed) |k| {
                self.upper_lens[self.upper_used + k] = 0;
            }
            self.upper_used += upper_needed;
            self.node_count += 1;

            if (self.entry_point == null) {
                self.entry_point = id;
                self.max_level = level;
                return;
            }

            var ep = self.entry_point.?;

            // Greedy descent from max_level down to level+1.
            var l: usize = self.max_level;
            while (l > @as(usize, level) + 1) : (l -= 1) {
                ep = self.greedyClosest(self.store.get(id), ep, l);
            }
            if (l > @as(usize, level)) {
                ep = self.greedyClosest(self.store.get(id), ep, l);
            }

            // Build edges from min(level, max_level) down to 0.
            const top_layer = @min(@as(usize, level), @as(usize, self.max_level));
            var layer_i: usize = top_layer + 1;
            // Stable copy of `selected` — connectNeighbors→pruneNeighbor
            // reuses ws.select_scratch for its own heuristic call.
            var selected_copy: [M0]heap.Entry = undefined;
            while (layer_i > 0) {
                layer_i -= 1;

                const results = try self.beamSearch(ws, self.store.get(id), ep, layer_i, self.ef_construction);
                const cap = layerCap(layer_i);
                const selected = self.selectNeighborsHeuristic(ws, id, results, cap);
                @memcpy(selected_copy[0..selected.len], selected);
                const stable = selected_copy[0..selected.len];
                self.connectNeighbors(ws, id, stable, layer_i);

                if (results.len > 0) ep = results[0].id;
            }

            if (level > self.max_level) {
                self.entry_point = id;
                self.max_level = level;
            }
        }

        /// Top-k search. Writes results into `out` and returns the populated slice.
        /// `out.len` must be >= k.
        pub fn search(
            self: *const Self,
            ws: *Workspace,
            query: []const f32,
            k: usize,
            ef_search: usize,
            out: []heap.Entry,
        ) ![]heap.Entry {
            std.debug.assert(query.len == self.store.dim);
            std.debug.assert(out.len >= k);
            std.debug.assert(ef_search >= k);
            std.debug.assert(ef_search <= ws.output.len);

            if (self.entry_point == null) return out[0..0];

            var ep = self.entry_point.?;
            var l: usize = self.max_level;
            while (l > 0) : (l -= 1) {
                ep = self.greedyClosest(query, ep, l);
            }

            const results = try self.beamSearch(ws, query, ep, 0, ef_search);
            const actual_k = @min(k, results.len);
            @memcpy(out[0..actual_k], results[0..actual_k]);
            return out[0..actual_k];
        }

        // ── persistence ─────────────────────────────────────────────────

        /// Format: "HGRF" | version u32 | M u32 | entry_point u32 | max_level u8
        ///         | ef_construction u32 | node_count u32
        ///         | per-node: level u8, per-layer: neighbor_count u16 + neighbor IDs u32…
        pub fn save(self: *const Self, dir: std.fs.Dir, sub_path: []const u8) !void {
            const file = try dir.createFile(sub_path, .{});
            defer file.close();

            var wbuf: [8192]u8 = undefined;
            var bw = file.writer(&wbuf);
            const w = &bw.interface;

            var hdr: [25]u8 = undefined;
            @memcpy(hdr[0..4], "HGRF");
            std.mem.writeInt(u32, hdr[4..8], 1, .little);
            std.mem.writeInt(u32, hdr[8..12], @intCast(M), .little);
            std.mem.writeInt(u32, hdr[12..16], self.entry_point orelse 0xFFFFFFFF, .little);
            hdr[16] = self.max_level;
            std.mem.writeInt(u32, hdr[17..21], @intCast(self.ef_construction), .little);
            std.mem.writeInt(u32, hdr[21..25], @intCast(self.node_count), .little);
            try w.writeAll(&hdr);

            var tmp4: [4]u8 = undefined;
            var tmp2: [2]u8 = undefined;
            for (0..self.node_count) |node_i| {
                const id: u32 = @intCast(node_i);
                const level = self.levels[id];
                try w.writeByte(level);

                for (0..@as(usize, level) + 1) |layer_i| {
                    const nbrs = self.neighborsOf(id, layer_i);
                    std.mem.writeInt(u16, &tmp2, @intCast(nbrs.len), .little);
                    try w.writeAll(&tmp2);
                    for (nbrs) |neighbor_id| {
                        std.mem.writeInt(u32, &tmp4, neighbor_id, .little);
                        try w.writeAll(&tmp4);
                    }
                }
            }

            try w.flush();
        }

        /// Scan a graph file and report the sizes needed to load it.
        /// Useful for sizing the `Params` passed to `load` exactly.
        pub fn scanSize(dir: std.fs.Dir, sub_path: []const u8) !LoadSize {
            const file = try dir.openFile(sub_path, .{});
            defer file.close();

            var rbuf: [8192]u8 = undefined;
            var br = file.readerStreaming(&rbuf);
            const r = &br.interface;

            var hdr: [25]u8 = undefined;
            try r.readSliceAll(&hdr);
            if (!std.mem.eql(u8, hdr[0..4], "HGRF")) return error.InvalidMagic;
            const version = std.mem.readInt(u32, hdr[4..8], .little);
            if (version != 1) return error.UnsupportedVersion;
            const file_m = std.mem.readInt(u32, hdr[8..12], .little);
            if (file_m != M) return error.MParameterMismatch;
            const node_count: usize = @intCast(std.mem.readInt(u32, hdr[21..25], .little));

            var upper_slots: usize = 0;
            var tmp4: [4]u8 = undefined;
            var tmp2: [2]u8 = undefined;
            var tmp1: [1]u8 = undefined;
            for (0..node_count) |_| {
                try r.readSliceAll(&tmp1);
                const level = tmp1[0];
                upper_slots += level;
                for (0..@as(usize, level) + 1) |_| {
                    try r.readSliceAll(&tmp2);
                    const nbr_count: usize = @intCast(std.mem.readInt(u16, &tmp2, .little));
                    // Skip neighbor IDs
                    for (0..nbr_count) |_| try r.readSliceAll(&tmp4);
                }
            }

            return .{ .node_count = node_count, .upper_slots = upper_slots };
        }

        pub fn load(
            allocator: std.mem.Allocator,
            store: *const Store,
            dir: std.fs.Dir,
            sub_path: []const u8,
            params: Params,
        ) !Self {
            const file = try dir.openFile(sub_path, .{});
            defer file.close();

            var rbuf: [8192]u8 = undefined;
            var br = file.readerStreaming(&rbuf);
            const r = &br.interface;

            var hdr: [25]u8 = undefined;
            try r.readSliceAll(&hdr);

            if (!std.mem.eql(u8, hdr[0..4], "HGRF")) return error.InvalidMagic;
            const version = std.mem.readInt(u32, hdr[4..8], .little);
            if (version != 1) return error.UnsupportedVersion;
            const file_m = std.mem.readInt(u32, hdr[8..12], .little);
            if (file_m != M) return error.MParameterMismatch;
            const raw_ep = std.mem.readInt(u32, hdr[12..16], .little);
            const entry_point: ?u32 = if (raw_ep == 0xFFFFFFFF) null else raw_ep;
            const max_level = hdr[16];
            const file_ef = @as(usize, @intCast(std.mem.readInt(u32, hdr[17..21], .little)));
            const node_count: usize = @intCast(std.mem.readInt(u32, hdr[21..25], .little));

            if (node_count > params.max_vectors) return error.CapacityTooSmall;

            // File's ef_construction wins over params.ef_construction if differ —
            // keeps round-trip save/load semantics identical.
            var adjusted = params;
            adjusted.ef_construction = file_ef;

            var self = try Self.init(allocator, store, adjusted);
            errdefer self.deinit();

            var tmp4: [4]u8 = undefined;
            var tmp2: [2]u8 = undefined;
            var tmp1: [1]u8 = undefined;
            for (0..node_count) |node_i| {
                const id: u32 = @intCast(node_i);
                try r.readSliceAll(&tmp1);
                const level = tmp1[0];

                if (self.upper_used + level > params.max_upper_slots) return error.UpperPoolFull;
                self.levels[id] = level;
                self.upper_offset[id] = @intCast(self.upper_used);
                self.upper_count[id] = level;
                self.upper_used += level;

                for (0..@as(usize, level) + 1) |layer_i| {
                    try r.readSliceAll(&tmp2);
                    const nbr_count: usize = @intCast(std.mem.readInt(u16, &tmp2, .little));
                    const cap = layerCap(layer_i);
                    std.debug.assert(nbr_count <= cap);
                    const view = self.layer(id, layer_i);
                    for (0..nbr_count) |_| {
                        try r.readSliceAll(&tmp4);
                        appendNeighbor(view, std.mem.readInt(u32, &tmp4, .little), cap);
                    }
                }
            }

            self.entry_point = entry_point;
            self.max_level = max_level;
            self.node_count = node_count;
            return self;
        }
    };
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "randomLevel distribution" {
    const Index = HnswIndex(16);
    var store = try Store.init(testing.allocator, 4, 1);
    defer store.deinit(testing.allocator);

    var idx = try Index.init(testing.allocator, &store, .{
        .max_vectors = 1,
        .max_upper_slots = 1,
        .ef_construction = 200,
        .seed = 42,
    });
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
    for (3..16) |i| sum_3plus += counts[i];
    const pct3plus = @as(f32, @floatFromInt(sum_3plus)) / total;
    try testing.expect(pct3plus < 0.001);

    std.debug.print("\nLevel distribution: L0={d:.2}% L1={d:.2}% L2={d:.4}% L3+={d:.4}%\n", .{
        pct0 * 100, pct1 * 100, pct2 * 100, pct3plus * 100,
    });
}

fn fillRandomStore(store: *Store, prng: *std.Random.DefaultPrng) !void {
    const random = prng.random();
    var buf: [4096]f32 = undefined; // hardcoded max test dim
    std.debug.assert(store.dim <= buf.len);
    const vec = buf[0..store.dim];
    for (0..store.capacity) |_| {
        for (vec) |*x| x.* = random.float(f32) * 2.0 - 1.0;
        bruteforce.normalize(vec);
        _ = try store.add(vec);
    }
}

test "recall@10 with heuristic neighbor selection (ef_search=50)" {
    const dim = 128;
    const num_vectors = 10_000;
    const num_queries = 100;
    const k = 10;
    const Index = HnswIndex(16);

    var store = try Store.init(testing.allocator, dim, num_vectors);
    defer store.deinit(testing.allocator);
    var prng = std.Random.DefaultPrng.init(42);
    try fillRandomStore(&store, &prng);

    var idx = try Index.init(testing.allocator, &store, .{
        .max_vectors = num_vectors,
        .max_upper_slots = num_vectors,
        .ef_construction = 200,
        .seed = 123,
    });
    defer idx.deinit();

    var ws = try Index.Workspace.init(testing.allocator, num_vectors, 200);
    defer ws.deinit(testing.allocator);

    for (0..num_vectors) |i| try idx.insert(&ws, @intCast(i));

    // Ground truth
    var truth_ids: [num_queries][k]u32 = undefined;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const truth = try bruteforce.search(&store, query, k, testing.allocator);
        defer testing.allocator.free(truth);
        for (0..k) |i| truth_ids[qi][i] = truth[i].id;
    }

    var out: [k]heap.Entry = undefined;
    var total_recall: f32 = 0;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const results = try idx.search(&ws, query, k, 50, &out);
        var result_ids: [k]u32 = undefined;
        const actual_k = @min(k, results.len);
        for (0..actual_k) |i| result_ids[i] = results[i].id;
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
    const Index = HnswIndex(16);

    var store = try Store.init(testing.allocator, dim, num_vectors);
    defer store.deinit(testing.allocator);
    var prng = std.Random.DefaultPrng.init(42);
    try fillRandomStore(&store, &prng);

    var idx = try Index.init(testing.allocator, &store, .{
        .max_vectors = num_vectors,
        .max_upper_slots = num_vectors,
        .ef_construction = 200,
        .seed = 123,
    });
    defer idx.deinit();

    var ws = try Index.Workspace.init(testing.allocator, num_vectors, 200);
    defer ws.deinit(testing.allocator);

    for (0..num_vectors) |i| try idx.insert(&ws, @intCast(i));

    var truth_ids: [num_queries][k]u32 = undefined;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const truth = try bruteforce.search(&store, query, k, testing.allocator);
        defer testing.allocator.free(truth);
        for (0..k) |i| truth_ids[qi][i] = truth[i].id;
    }

    var out: [k]heap.Entry = undefined;
    var total_recall: f32 = 0;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const results = try idx.search(&ws, query, k, 100, &out);
        var result_ids: [k]u32 = undefined;
        const actual_k = @min(k, results.len);
        for (0..actual_k) |i| result_ids[i] = results[i].id;
        total_recall += bruteforce.computeRecall(&truth_ids[qi], result_ids[0..actual_k]);
    }

    const avg_recall = total_recall / @as(f32, @floatFromInt(num_queries));
    std.debug.print("\nrecall@{d} (ef_search=100): {d:.4}\n", .{ k, avg_recall });
    try testing.expect(avg_recall >= 0.95);
}

test "save/load round-trip preserves recall@10" {
    const dim = 128;
    const num_vectors = 10_000;
    const num_queries = 100;
    const k = 10;
    const Index = HnswIndex(16);

    var store = try Store.init(testing.allocator, dim, num_vectors);
    defer store.deinit(testing.allocator);
    var prng = std.Random.DefaultPrng.init(42);
    try fillRandomStore(&store, &prng);

    var idx = try Index.init(testing.allocator, &store, .{
        .max_vectors = num_vectors,
        .max_upper_slots = num_vectors,
        .ef_construction = 200,
        .seed = 123,
    });
    defer idx.deinit();

    var ws = try Index.Workspace.init(testing.allocator, num_vectors, 200);
    defer ws.deinit(testing.allocator);

    for (0..num_vectors) |i| try idx.insert(&ws, @intCast(i));

    var truth_ids: [num_queries][k]u32 = undefined;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const truth = try bruteforce.search(&store, query, k, testing.allocator);
        defer testing.allocator.free(truth);
        for (0..k) |i| truth_ids[qi][i] = truth[i].id;
    }

    var out: [k]heap.Entry = undefined;
    var recall_before: f32 = 0;
    for (0..num_queries) |qi| {
        const query = store.get(@intCast(qi));
        const results = try idx.search(&ws, query, k, 100, &out);
        var result_ids: [k]u32 = undefined;
        const actual_k = @min(k, results.len);
        for (0..actual_k) |i| result_ids[i] = results[i].id;
        recall_before += bruteforce.computeRecall(&truth_ids[qi], result_ids[0..actual_k]);
    }
    recall_before /= @as(f32, @floatFromInt(num_queries));

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try store.save(tmp.dir, "vectors.hvsf");
    try idx.save(tmp.dir, "graph.hgrf");

    var loaded_store = try Store.load(testing.allocator, tmp.dir, "vectors.hvsf", null);
    defer loaded_store.deinit(testing.allocator);

    const sz = try Index.scanSize(tmp.dir, "graph.hgrf");

    var loaded_idx = try Index.load(testing.allocator, &loaded_store, tmp.dir, "graph.hgrf", .{
        .max_vectors = sz.node_count,
        .max_upper_slots = sz.upper_slots,
        .ef_construction = 200,
        .seed = 0,
    });
    defer loaded_idx.deinit();

    var loaded_ws = try Index.Workspace.init(testing.allocator, num_vectors, 200);
    defer loaded_ws.deinit(testing.allocator);

    var recall_after: f32 = 0;
    for (0..num_queries) |qi| {
        const query = loaded_store.get(@intCast(qi));
        const results = try loaded_idx.search(&loaded_ws, query, k, 100, &out);
        var result_ids: [k]u32 = undefined;
        const actual_k = @min(k, results.len);
        for (0..actual_k) |i| result_ids[i] = results[i].id;
        recall_after += bruteforce.computeRecall(&truth_ids[qi], result_ids[0..actual_k]);
    }
    recall_after /= @as(f32, @floatFromInt(num_queries));

    std.debug.print("\nrecall@{d} before save: {d:.4}\n", .{ k, recall_before });
    std.debug.print("recall@{d} after load:  {d:.4}\n", .{ k, recall_after });
    try testing.expectApproxEqAbs(recall_before, recall_after, 1e-6);
}
