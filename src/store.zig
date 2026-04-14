const std = @import("std");
const assert = std.debug.assert;

pub fn Store(comptime dim_: usize) type {
    return struct {
        const Self = @This();

        pub const dim = dim_;
        data: []align(64) f32,
        count: usize,
        capacity: usize,

        pub fn init(allocator: std.mem.Allocator, cap: usize) !Self {
            const alignment = comptime std.mem.Alignment.fromByteUnits(64);
            const buf = try allocator.alignedAlloc(f32, alignment, cap * dim);
            return .{
                .capacity = cap,
                .count = 0,
                .data = buf,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
        }

        pub fn add(self: *Self, v: [dim]f32) !u32 {
            if (self.count >= self.capacity) return error.OutOfCapacity;

            const id: u32 = @intCast(self.count);
            @memcpy(self.data[id * dim ..][0..dim], &v);
            self.count += 1;

            return id;
        }

        pub fn get(self: *Self, id: u32) *const [dim]f32 {
            return self.data[id * dim ..][0..dim];
        }
    };
}

const testing = std.testing;

test "add and get round-trips correctly" {
    const S = Store(3);
    var store = try S.init(testing.allocator, 10);
    defer store.deinit(testing.allocator);

    const vec = [_]f32{ 1.0, 2.0, 3.0 };
    const id = try store.add(vec);

    try testing.expectEqual(@as(u32, 0), id);
    try testing.expectEqualSlices(f32, &vec, store.get(id));
}

test "sequential adds return incrementing ids" {
    const S = Store(2);
    var store = try S.init(testing.allocator, 10);
    defer store.deinit(testing.allocator);

    const id0 = try store.add(.{ 1.0, 2.0 });
    const id1 = try store.add(.{ 3.0, 4.0 });
    const id2 = try store.add(.{ 5.0, 6.0 });

    try testing.expectEqual(@as(u32, 0), id0);
    try testing.expectEqual(@as(u32, 1), id1);
    try testing.expectEqual(@as(u32, 2), id2);
    try testing.expectEqual(@as(usize, 3), store.count);
}

test "vectors are stored independently" {
    const S = Store(2);
    var store = try S.init(testing.allocator, 10);
    defer store.deinit(testing.allocator);

    _ = try store.add(.{ 1.0, 2.0 });
    _ = try store.add(.{ 3.0, 4.0 });
    _ = try store.add(.{ 5.0, 6.0 });

    // earlier vectors must not be overwritten by later adds
    try testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, store.get(0));
    try testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, store.get(1));
    try testing.expectEqualSlices(f32, &.{ 5.0, 6.0 }, store.get(2));
}

test "add returns OutOfCapacity when full" {
    const S = Store(2);
    var store = try S.init(testing.allocator, 2);
    defer store.deinit(testing.allocator);

    _ = try store.add(.{ 1.0, 2.0 });
    _ = try store.add(.{ 3.0, 4.0 });

    const result = store.add(.{ 5.0, 6.0 });
    try testing.expectError(error.OutOfCapacity, result);
}
