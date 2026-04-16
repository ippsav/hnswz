const std = @import("std");
const assert = std.debug.assert;

/// Flat, contiguous []f32 vector store. All memory is allocated up front at
/// `init(allocator, dim, capacity)` and never grown. `dim` is a runtime
/// field (configured at server startup), not a comptime parameter.
///
/// Slots can be tombstoned via `delete` and reused by later `add` calls.
/// `count` is the high-water mark (max slot index + 1 ever used);
/// `live_count` is the current non-deleted count.
pub const Store = struct {
    const Self = @This();

    dim: usize,
    data: []align(64) f32,
    deleted: std.DynamicBitSetUnmanaged,
    free_list: []u32,
    free_len: usize,
    count: usize,
    live_count: usize,
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, dim: usize, cap: usize) !Self {
        const alignment = comptime std.mem.Alignment.fromByteUnits(64);
        const buf = try allocator.alignedAlloc(f32, alignment, cap * dim);
        errdefer allocator.free(buf);

        var deleted = try std.DynamicBitSetUnmanaged.initEmpty(allocator, cap);
        errdefer deleted.deinit(allocator);

        const free_list = try allocator.alloc(u32, cap);

        return .{
            .dim = dim,
            .capacity = cap,
            .count = 0,
            .live_count = 0,
            .data = buf,
            .deleted = deleted,
            .free_list = free_list,
            .free_len = 0,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.deleted.deinit(allocator);
        allocator.free(self.free_list);
    }

    pub fn isDeleted(self: *const Self, id: u32) bool {
        return self.deleted.isSet(id);
    }

    /// Append a vector. Reuses the most recently deleted slot if one is
    /// available, otherwise bumps `count`. Returns the assigned id.
    pub fn add(self: *Self, v: []const f32) !u32 {
        assert(v.len == self.dim);
        if (self.live_count >= self.capacity) return error.OutOfCapacity;

        const id: u32 = if (self.free_len > 0) blk: {
            self.free_len -= 1;
            const reused = self.free_list[self.free_len];
            self.deleted.unset(reused);
            break :blk reused;
        } else blk: {
            const new_id: u32 = @intCast(self.count);
            self.count += 1;
            break :blk new_id;
        };

        @memcpy(self.data[id * self.dim ..][0..self.dim], v);
        self.live_count += 1;
        return id;
    }

    /// Re-add a vector at a specific previously-deleted slot. Used by
    /// `replaceVector` to keep an id stable across an update, so external
    /// metadata (id -> filename etc.) does not need remapping.
    pub fn addAt(self: *Self, id: u32, v: []const f32) !void {
        assert(v.len == self.dim);
        assert(id < self.count);
        if (!self.deleted.isSet(id)) return error.SlotOccupied;

        var found = false;
        var i: usize = self.free_len;
        while (i > 0) {
            i -= 1;
            if (self.free_list[i] == id) {
                self.free_list[i] = self.free_list[self.free_len - 1];
                self.free_len -= 1;
                found = true;
                break;
            }
        }
        if (!found) return error.NotInFreeList;

        self.deleted.unset(id);
        @memcpy(self.data[id * self.dim ..][0..self.dim], v);
        self.live_count += 1;
    }

    /// Mark a slot deleted and push it onto the free list. Idempotent:
    /// deleting an already-deleted id is a no-op (returns without error).
    pub fn delete(self: *Self, id: u32) void {
        assert(id < self.count);
        if (self.deleted.isSet(id)) return;
        self.deleted.set(id);
        self.free_list[self.free_len] = id;
        self.free_len += 1;
        self.live_count -= 1;
    }

    pub fn get(self: *const Self, id: u32) []const f32 {
        return self.data[id * self.dim ..][0..self.dim];
    }

    /// old_to_new[old_id] = new dense id, or 0xFFFFFFFF for deleted slots.
    /// Length equals `self.count`. Used by HnswIndex.save and metadata save
    /// to rewrite id references when persisting a compacted snapshot.
    pub fn buildRemap(self: *const Self, allocator: std.mem.Allocator) ![]u32 {
        const remap = try allocator.alloc(u32, self.count);
        var new_id: u32 = 0;
        for (0..self.count) |i| {
            const oid: u32 = @intCast(i);
            if (self.deleted.isSet(oid)) {
                remap[i] = 0xFFFFFFFF;
            } else {
                remap[i] = new_id;
                new_id += 1;
            }
        }
        return remap;
    }

    /// Persist vectors to disk in dense (compacted) order.
    /// Format: "HVSF" | version u32 | dim u32 | count u64 | raw f32 bytes
    /// The on-disk `count` is the live count; deleted slots are skipped.
    pub fn save(self: *const Self, dir: std.fs.Dir, sub_path: []const u8) !void {
        const file = try dir.createFile(sub_path, .{});
        defer file.close();

        var wbuf: [8192]u8 = undefined;
        var bw = file.writer(&wbuf);
        const w = &bw.interface;

        var hdr: [20]u8 = undefined;
        @memcpy(hdr[0..4], "HVSF");
        std.mem.writeInt(u32, hdr[4..8], 1, .little);
        std.mem.writeInt(u32, hdr[8..12], @intCast(self.dim), .little);
        std.mem.writeInt(u64, hdr[12..20], @intCast(self.live_count), .little);
        try w.writeAll(&hdr);

        if (self.live_count == self.count) {
            try w.writeAll(std.mem.sliceAsBytes(self.data[0 .. self.count * self.dim]));
        } else {
            for (0..self.count) |i| {
                const id: u32 = @intCast(i);
                if (self.deleted.isSet(id)) continue;
                try w.writeAll(std.mem.sliceAsBytes(self.data[id * self.dim ..][0..self.dim]));
            }
        }
        try w.flush();
    }

    /// Load vectors from disk. Caller owns the returned store.
    /// Allocates exactly `count * dim` f32 (no slack capacity) unless
    /// `cap_override` is non-null, in which case the store has `cap_override`
    /// capacity and `count` filled entries.
    pub fn load(
        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        sub_path: []const u8,
        cap_override: ?usize,
    ) !Self {
        const file = try dir.openFile(sub_path, .{});
        defer file.close();

        var rbuf: [8192]u8 = undefined;
        var br = file.readerStreaming(&rbuf);
        const r = &br.interface;

        var hdr: [20]u8 = undefined;
        try r.readSliceAll(&hdr);

        if (!std.mem.eql(u8, hdr[0..4], "HVSF")) return error.InvalidMagic;

        const version = std.mem.readInt(u32, hdr[4..8], .little);
        if (version != 1) return error.UnsupportedVersion;

        const file_dim: usize = @intCast(std.mem.readInt(u32, hdr[8..12], .little));
        const count: usize = @intCast(std.mem.readInt(u64, hdr[12..20], .little));

        const cap = cap_override orelse count;
        if (cap < count) return error.CapacityTooSmall;

        var store = try Self.init(allocator, file_dim, cap);
        errdefer store.deinit(allocator);

        try r.readSliceAll(std.mem.sliceAsBytes(store.data[0 .. count * file_dim]));
        store.count = count;
        store.live_count = count;

        return store;
    }
};

const testing = std.testing;

test "add and get round-trips correctly" {
    var store = try Store.init(testing.allocator, 3, 10);
    defer store.deinit(testing.allocator);

    const vec = [_]f32{ 1.0, 2.0, 3.0 };
    const id = try store.add(&vec);

    try testing.expectEqual(@as(u32, 0), id);
    try testing.expectEqualSlices(f32, &vec, store.get(id));
}

test "sequential adds return incrementing ids" {
    var store = try Store.init(testing.allocator, 2, 10);
    defer store.deinit(testing.allocator);

    const id0 = try store.add(&[_]f32{ 1.0, 2.0 });
    const id1 = try store.add(&[_]f32{ 3.0, 4.0 });
    const id2 = try store.add(&[_]f32{ 5.0, 6.0 });

    try testing.expectEqual(@as(u32, 0), id0);
    try testing.expectEqual(@as(u32, 1), id1);
    try testing.expectEqual(@as(u32, 2), id2);
    try testing.expectEqual(@as(usize, 3), store.count);
    try testing.expectEqual(@as(usize, 3), store.live_count);
}

test "vectors are stored independently" {
    var store = try Store.init(testing.allocator, 2, 10);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 4.0 });
    _ = try store.add(&[_]f32{ 5.0, 6.0 });

    try testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, store.get(0));
    try testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, store.get(1));
    try testing.expectEqualSlices(f32, &.{ 5.0, 6.0 }, store.get(2));
}

test "add returns OutOfCapacity when full" {
    var store = try Store.init(testing.allocator, 2, 2);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 4.0 });

    const result = store.add(&[_]f32{ 5.0, 6.0 });
    try testing.expectError(error.OutOfCapacity, result);
}

test "delete tombstones and decrements live_count" {
    var store = try Store.init(testing.allocator, 2, 4);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 4.0 });
    _ = try store.add(&[_]f32{ 5.0, 6.0 });

    store.delete(1);

    try testing.expectEqual(@as(usize, 3), store.count);
    try testing.expectEqual(@as(usize, 2), store.live_count);
    try testing.expect(store.isDeleted(1));
    try testing.expect(!store.isDeleted(0));
    try testing.expect(!store.isDeleted(2));
}

test "delete is idempotent on already-deleted slot" {
    var store = try Store.init(testing.allocator, 2, 4);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    store.delete(0);
    store.delete(0);

    try testing.expectEqual(@as(usize, 0), store.live_count);
    try testing.expectEqual(@as(usize, 1), store.free_len);
}

test "add reuses freed slot (LIFO)" {
    var store = try Store.init(testing.allocator, 2, 4);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 4.0 });
    _ = try store.add(&[_]f32{ 5.0, 6.0 });

    store.delete(1);
    const reused_id = try store.add(&[_]f32{ 7.0, 8.0 });

    try testing.expectEqual(@as(u32, 1), reused_id);
    try testing.expectEqual(@as(usize, 3), store.count);
    try testing.expectEqual(@as(usize, 3), store.live_count);
    try testing.expectEqualSlices(f32, &.{ 7.0, 8.0 }, store.get(1));
    try testing.expect(!store.isDeleted(1));
}

test "add falls back to bump when free list is empty" {
    var store = try Store.init(testing.allocator, 2, 4);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    store.delete(0);
    const reused = try store.add(&[_]f32{ 3.0, 4.0 });
    try testing.expectEqual(@as(u32, 0), reused);

    const fresh = try store.add(&[_]f32{ 5.0, 6.0 });
    try testing.expectEqual(@as(u32, 1), fresh);
    try testing.expectEqual(@as(usize, 2), store.count);
}

test "addAt reuses a specific slot" {
    var store = try Store.init(testing.allocator, 2, 4);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 4.0 });
    _ = try store.add(&[_]f32{ 5.0, 6.0 });

    store.delete(0);
    store.delete(2);
    // free_list now holds [0, 2] in some order; addAt picks the exact id.
    try store.addAt(0, &[_]f32{ 9.0, 9.0 });

    try testing.expect(!store.isDeleted(0));
    try testing.expect(store.isDeleted(2));
    try testing.expectEqualSlices(f32, &.{ 9.0, 9.0 }, store.get(0));
    try testing.expectEqual(@as(usize, 1), store.free_len);
}

test "addAt rejects an occupied slot" {
    var store = try Store.init(testing.allocator, 2, 4);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    try testing.expectError(error.SlotOccupied, store.addAt(0, &[_]f32{ 9.0, 9.0 }));
}

test "save and load round-trip" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var store = try Store.init(testing.allocator, 3, 5);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{ 1.0, 2.0, 3.0 });
    _ = try store.add(&[_]f32{ 4.0, 5.0, 6.0 });

    try store.save(tmp.dir, "v.hvsf");

    var loaded = try Store.load(testing.allocator, tmp.dir, "v.hvsf", null);
    defer loaded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), loaded.dim);
    try testing.expectEqual(@as(usize, 2), loaded.count);
    try testing.expectEqual(@as(usize, 2), loaded.live_count);
    try testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0 }, loaded.get(0));
    try testing.expectEqualSlices(f32, &.{ 4.0, 5.0, 6.0 }, loaded.get(1));
}

test "load with cap_override preserves growth room" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var store = try Store.init(testing.allocator, 2, 2);
    defer store.deinit(testing.allocator);
    _ = try store.add(&[_]f32{ 1.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 4.0 });
    try store.save(tmp.dir, "v.hvsf");

    var loaded = try Store.load(testing.allocator, tmp.dir, "v.hvsf", 100);
    defer loaded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), loaded.count);
    try testing.expectEqual(@as(usize, 2), loaded.live_count);
    try testing.expectEqual(@as(usize, 100), loaded.capacity);
}

test "save compacts over deleted slots" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var store = try Store.init(testing.allocator, 2, 5);
    defer store.deinit(testing.allocator);
    _ = try store.add(&[_]f32{ 1.0, 1.0 });
    _ = try store.add(&[_]f32{ 2.0, 2.0 });
    _ = try store.add(&[_]f32{ 3.0, 3.0 });
    _ = try store.add(&[_]f32{ 4.0, 4.0 });

    store.delete(1);
    store.delete(2);
    try store.save(tmp.dir, "v.hvsf");

    var loaded = try Store.load(testing.allocator, tmp.dir, "v.hvsf", null);
    defer loaded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), loaded.count);
    try testing.expectEqual(@as(usize, 2), loaded.live_count);
    try testing.expectEqualSlices(f32, &.{ 1.0, 1.0 }, loaded.get(0));
    try testing.expectEqualSlices(f32, &.{ 4.0, 4.0 }, loaded.get(1));
}

test "buildRemap maps live ids densely and sentinels deleted ones" {
    var store = try Store.init(testing.allocator, 1, 5);
    defer store.deinit(testing.allocator);

    _ = try store.add(&[_]f32{1.0});
    _ = try store.add(&[_]f32{2.0});
    _ = try store.add(&[_]f32{3.0});
    _ = try store.add(&[_]f32{4.0});

    store.delete(1);
    store.delete(3);

    const remap = try store.buildRemap(testing.allocator);
    defer testing.allocator.free(remap);

    try testing.expectEqual(@as(u32, 0), remap[0]);
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), remap[1]);
    try testing.expectEqual(@as(u32, 1), remap[2]);
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), remap[3]);
}
