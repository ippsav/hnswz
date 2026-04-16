const std = @import("std");

pub const Entry = struct {
    id: u32,
    dist: f32,
};

/// Fixed-capacity binary heap backed by a caller-provided buffer.
/// No allocations. The comptime `is_min` flag picks min- vs max-heap ordering.
pub fn FixedHeap(comptime is_min: bool) type {
    return struct {
        const Self = @This();

        buf: []Entry,
        len: usize,

        /// Wrap a caller-owned buffer as an empty heap.
        pub fn init(buf: []Entry) Self {
            return .{ .buf = buf, .len = 0 };
        }

        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        pub fn count(self: *const Self) usize {
            return self.len;
        }

        pub fn capacity(self: *const Self) usize {
            return self.buf.len;
        }

        pub fn peek(self: *const Self) ?Entry {
            if (self.len == 0) return null;
            return self.buf[0];
        }

        /// Push an entry. Returns `error.HeapFull` if the backing buffer is
        /// full — callers must size `buf` to the worst-case high-water mark.
        pub fn push(self: *Self, e: Entry) error{HeapFull}!void {
            if (self.len == self.buf.len) return error.HeapFull;
            self.buf[self.len] = e;
            self.len += 1;
            self.siftUp(self.len - 1);
        }

        /// Pop the root (min for is_min, max otherwise). Caller must ensure non-empty.
        pub fn pop(self: *Self) Entry {
            std.debug.assert(self.len > 0);
            const root = self.buf[0];
            self.len -= 1;
            if (self.len > 0) {
                self.buf[0] = self.buf[self.len];
                self.siftDown(0);
            }
            return root;
        }

        fn less(a: Entry, b: Entry) bool {
            return if (is_min) a.dist < b.dist else a.dist > b.dist;
        }

        fn siftUp(self: *Self, idx: usize) void {
            var i = idx;
            while (i > 0) {
                const parent = (i - 1) / 2;
                if (less(self.buf[i], self.buf[parent])) {
                    const tmp = self.buf[i];
                    self.buf[i] = self.buf[parent];
                    self.buf[parent] = tmp;
                    i = parent;
                } else break;
            }
        }

        fn siftDown(self: *Self, idx: usize) void {
            var i = idx;
            while (true) {
                const l = 2 * i + 1;
                const r = 2 * i + 2;
                var best = i;
                if (l < self.len and less(self.buf[l], self.buf[best])) best = l;
                if (r < self.len and less(self.buf[r], self.buf[best])) best = r;
                if (best == i) break;
                const tmp = self.buf[i];
                self.buf[i] = self.buf[best];
                self.buf[best] = tmp;
                i = best;
            }
        }
    };
}

/// Min-heap: pop returns the smallest distance. Used as the candidate queue.
pub const MinHeap = FixedHeap(true);

/// Max-heap: pop returns the largest distance. Used as the results set.
pub const MaxHeap = FixedHeap(false);

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "MinHeap pops in ascending distance order" {
    var buf: [8]Entry = undefined;
    var h = MinHeap.init(&buf);

    try h.push(.{ .id = 0, .dist = 0.5 });
    try h.push(.{ .id = 1, .dist = 0.1 });
    try h.push(.{ .id = 2, .dist = 0.9 });
    try h.push(.{ .id = 3, .dist = 0.3 });

    try testing.expectEqual(@as(u32, 1), h.pop().id); // 0.1
    try testing.expectEqual(@as(u32, 3), h.pop().id); // 0.3
    try testing.expectEqual(@as(u32, 0), h.pop().id); // 0.5
    try testing.expectEqual(@as(u32, 2), h.pop().id); // 0.9
}

test "MaxHeap pops in descending distance order" {
    var buf: [8]Entry = undefined;
    var h = MaxHeap.init(&buf);

    try h.push(.{ .id = 0, .dist = 0.5 });
    try h.push(.{ .id = 1, .dist = 0.1 });
    try h.push(.{ .id = 2, .dist = 0.9 });
    try h.push(.{ .id = 3, .dist = 0.3 });

    try testing.expectEqual(@as(u32, 2), h.pop().id); // 0.9
    try testing.expectEqual(@as(u32, 0), h.pop().id); // 0.5
    try testing.expectEqual(@as(u32, 3), h.pop().id); // 0.3
    try testing.expectEqual(@as(u32, 1), h.pop().id); // 0.1
}

test "MinHeap peek returns smallest without removing" {
    var buf: [4]Entry = undefined;
    var h = MinHeap.init(&buf);

    try h.push(.{ .id = 0, .dist = 0.7 });
    try h.push(.{ .id = 1, .dist = 0.2 });

    try testing.expectEqual(@as(f32, 0.2), h.peek().?.dist);
    try testing.expectEqual(@as(usize, 2), h.count());
}

test "MaxHeap peek returns largest without removing" {
    var buf: [4]Entry = undefined;
    var h = MaxHeap.init(&buf);

    try h.push(.{ .id = 0, .dist = 0.2 });
    try h.push(.{ .id = 1, .dist = 0.7 });

    try testing.expectEqual(@as(f32, 0.7), h.peek().?.dist);
    try testing.expectEqual(@as(usize, 2), h.count());
}

test "empty heaps return null on peek" {
    var buf_min: [4]Entry = undefined;
    var buf_max: [4]Entry = undefined;
    var min = MinHeap.init(&buf_min);
    var max = MaxHeap.init(&buf_max);

    try testing.expect(min.peek() == null);
    try testing.expect(max.peek() == null);
}

test "push returns HeapFull when at capacity" {
    var buf: [2]Entry = undefined;
    var h = MinHeap.init(&buf);

    try h.push(.{ .id = 0, .dist = 1.0 });
    try h.push(.{ .id = 1, .dist = 2.0 });
    try testing.expectError(error.HeapFull, h.push(.{ .id = 2, .dist = 3.0 }));
}

test "clear resets length to zero" {
    var buf: [4]Entry = undefined;
    var h = MinHeap.init(&buf);

    try h.push(.{ .id = 0, .dist = 1.0 });
    try h.push(.{ .id = 1, .dist = 2.0 });
    h.clear();

    try testing.expectEqual(@as(usize, 0), h.count());
    try testing.expect(h.peek() == null);
}
