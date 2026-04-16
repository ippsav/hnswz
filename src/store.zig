const std = @import("std");
const assert = std.debug.assert;

/// Flat, contiguous []f32 vector store. All memory is allocated up front at
/// `init(allocator, dim, capacity)` and never grown. `dim` is a runtime
/// field (configured at server startup), not a comptime parameter.
pub const Store = struct {
    const Self = @This();

    dim: usize,
    data: []align(64) f32,
    count: usize,
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, dim: usize, cap: usize) !Self {
        const alignment = comptime std.mem.Alignment.fromByteUnits(64);
        const buf = try allocator.alignedAlloc(f32, alignment, cap * dim);
        return .{
            .dim = dim,
            .capacity = cap,
            .count = 0,
            .data = buf,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    /// Append a vector. `v.len` must equal `self.dim`.
    pub fn add(self: *Self, v: []const f32) !u32 {
        assert(v.len == self.dim);
        if (self.count >= self.capacity) return error.OutOfCapacity;

        const id: u32 = @intCast(self.count);
        @memcpy(self.data[id * self.dim ..][0..self.dim], v);
        self.count += 1;

        return id;
    }

    pub fn get(self: *const Self, id: u32) []const f32 {
        return self.data[id * self.dim ..][0..self.dim];
    }

    /// Persist vectors to disk.
    /// Format: "HVSF" | version u32 | dim u32 | count u64 | raw f32 bytes
    pub fn save(self: *Self, dir: std.fs.Dir, sub_path: []const u8) !void {
        const file = try dir.createFile(sub_path, .{});
        defer file.close();

        var wbuf: [8192]u8 = undefined;
        var bw = file.writer(&wbuf);
        const w = &bw.interface;

        // Header: magic(4) + version(4) + dim(4) + count(8) = 20 bytes
        var hdr: [20]u8 = undefined;
        @memcpy(hdr[0..4], "HVSF");
        std.mem.writeInt(u32, hdr[4..8], 1, .little);
        std.mem.writeInt(u32, hdr[8..12], @intCast(self.dim), .little);
        std.mem.writeInt(u64, hdr[12..20], @intCast(self.count), .little);
        try w.writeAll(&hdr);

        // Raw vector data
        try w.writeAll(std.mem.sliceAsBytes(self.data[0 .. self.count * self.dim]));
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
    try testing.expectEqual(@as(usize, 100), loaded.capacity);
}
