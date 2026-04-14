const std = @import("std");

pub const Entry = struct {
    id: u32,
    dist: f32,
};

fn minOrder(_: void, a: Entry, b: Entry) std.math.Order {
    return std.math.order(a.dist, b.dist);
}

fn maxOrder(_: void, a: Entry, b: Entry) std.math.Order {
    return std.math.order(b.dist, a.dist);
}

/// Min-heap: pop returns the entry with the smallest distance.
/// Used as the candidates queue during beam search.
pub const MinHeap = std.PriorityQueue(Entry, void, minOrder);

/// Max-heap: pop returns the entry with the largest distance.
/// Used as the results set during beam search — the top is the farthest
/// "good" result, so a new candidate can be compared against it cheaply.
pub const MaxHeap = std.PriorityQueue(Entry, void, maxOrder);

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "MinHeap pops in ascending distance order" {
    var h = MinHeap.init(testing.allocator, {});
    defer h.deinit();

    try h.add(.{ .id = 0, .dist = 0.5 });
    try h.add(.{ .id = 1, .dist = 0.1 });
    try h.add(.{ .id = 2, .dist = 0.9 });
    try h.add(.{ .id = 3, .dist = 0.3 });

    try testing.expectEqual(@as(u32, 1), h.remove().id); // 0.1
    try testing.expectEqual(@as(u32, 3), h.remove().id); // 0.3
    try testing.expectEqual(@as(u32, 0), h.remove().id); // 0.5
    try testing.expectEqual(@as(u32, 2), h.remove().id); // 0.9
}

test "MaxHeap pops in descending distance order" {
    var h = MaxHeap.init(testing.allocator, {});
    defer h.deinit();

    try h.add(.{ .id = 0, .dist = 0.5 });
    try h.add(.{ .id = 1, .dist = 0.1 });
    try h.add(.{ .id = 2, .dist = 0.9 });
    try h.add(.{ .id = 3, .dist = 0.3 });

    try testing.expectEqual(@as(u32, 2), h.remove().id); // 0.9
    try testing.expectEqual(@as(u32, 0), h.remove().id); // 0.5
    try testing.expectEqual(@as(u32, 3), h.remove().id); // 0.3
    try testing.expectEqual(@as(u32, 1), h.remove().id); // 0.1
}

test "MinHeap peek returns smallest without removing" {
    var h = MinHeap.init(testing.allocator, {});
    defer h.deinit();

    try h.add(.{ .id = 0, .dist = 0.7 });
    try h.add(.{ .id = 1, .dist = 0.2 });

    try testing.expectEqual(@as(f32, 0.2), h.peek().?.dist);
    try testing.expectEqual(@as(usize, 2), h.count());
}

test "MaxHeap peek returns largest without removing" {
    var h = MaxHeap.init(testing.allocator, {});
    defer h.deinit();

    try h.add(.{ .id = 0, .dist = 0.2 });
    try h.add(.{ .id = 1, .dist = 0.7 });

    try testing.expectEqual(@as(f32, 0.7), h.peek().?.dist);
    try testing.expectEqual(@as(usize, 2), h.count());
}

test "empty heaps return null on peek" {
    var min = MinHeap.init(testing.allocator, {});
    defer min.deinit();
    var max = MaxHeap.init(testing.allocator, {});
    defer max.deinit();

    try testing.expect(min.peek() == null);
    try testing.expect(max.peek() == null);
}
