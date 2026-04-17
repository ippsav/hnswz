//! Server-side mutable analogue of `metadata.Metadata`. Stores the id →
//! filename mapping for a live (long-running) server that inserts and
//! deletes vectors incrementally.
//!
//! Each slot is an `?[]u8` owned by this module: `null` for tombstoned /
//! never-assigned ids, populated otherwise. IDs passed to `setAt` are
//! assigned by the `Store` (LIFO free-list or bump counter) so we simply
//! follow along; this module does not mint ids.
//!
//! `save` writes a sparse HMTF v2 file whose layout matches the in-memory
//! slot array 1:1 — tombstones are preserved so that file ids and
//! in-memory ids stay in one id space across snapshot/restart boundaries.
const std = @import("std");
const metadata = @import("metadata.zig");

pub const MutableMetadata = struct {
    const Self = @This();

    /// slot `i` is the name for vector id `i`. `null` = tombstoned or
    /// never assigned.
    slots: std.ArrayListUnmanaged(?[]u8),
    /// Number of non-null slots. Kept in sync with Store.live_count by the
    /// server — not validated by this module.
    live_count: usize,

    pub fn init() Self {
        return .{ .slots = .{}, .live_count = 0 };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.slots.items) |maybe| {
            if (maybe) |owned| allocator.free(owned);
        }
        self.slots.deinit(allocator);
        self.* = undefined;
    }

    /// Pre-grow to exactly `cap` slots (all null). Cheap up-front allocation
    /// matching `Store.capacity` so the server doesn't resize under churn.
    pub fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, cap: usize) !void {
        try self.slots.ensureTotalCapacity(allocator, cap);
    }

    /// Write `name` at slot `id`, growing the backing array with `null`
    /// placeholders if needed. Any existing name at that slot is freed.
    pub fn setAt(self: *Self, allocator: std.mem.Allocator, id: u32, name: []const u8) !void {
        const idx: usize = id;
        while (self.slots.items.len <= idx) {
            try self.slots.append(allocator, null);
        }
        if (self.slots.items[idx]) |old| {
            allocator.free(old);
        } else {
            self.live_count += 1;
        }
        self.slots.items[idx] = try allocator.dupe(u8, name);
    }

    pub fn deleteAt(self: *Self, allocator: std.mem.Allocator, id: u32) void {
        const idx: usize = id;
        if (idx >= self.slots.items.len) return;
        if (self.slots.items[idx]) |old| {
            allocator.free(old);
            self.slots.items[idx] = null;
            self.live_count -= 1;
        }
    }

    pub fn get(self: *const Self, id: u32) ?[]const u8 {
        const idx: usize = id;
        if (idx >= self.slots.items.len) return null;
        return self.slots.items[idx];
    }

    pub fn highWater(self: *const Self) usize {
        return self.slots.items.len;
    }

    /// Write an HMTF v2 file that mirrors this module's in-memory slots
    /// 1:1. `total_slots` must equal `store.count` — callers pass it
    /// explicitly so we can emit trailing tombstones for slots the store
    /// allocated but that this module has never seen (the HNSW graph
    /// stays in sync by id even when metadata is never assigned).
    pub fn save(
        self: *const Self,
        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        sub_path: []const u8,
        total_slots: usize,
    ) !void {
        std.debug.assert(total_slots >= self.slots.items.len);

        var slots = try allocator.alloc(?[]const u8, total_slots);
        defer allocator.free(slots);

        var live_seen: usize = 0;
        for (0..total_slots) |i| {
            if (i < self.slots.items.len) {
                if (self.slots.items[i]) |name| {
                    slots[i] = name;
                    live_seen += 1;
                } else {
                    slots[i] = null;
                }
            } else {
                slots[i] = null;
            }
        }
        std.debug.assert(live_seen == self.live_count);

        try metadata.save(dir, sub_path, slots, self.live_count);
    }
};



const testing = std.testing;

test "init is empty" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), md.live_count);
    try testing.expectEqual(@as(?[]const u8, null), md.get(0));
}

test "setAt stores and grows" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "first.txt");
    try md.setAt(testing.allocator, 2, "third.txt"); // skips 1

    try testing.expectEqual(@as(usize, 2), md.live_count);
    try testing.expectEqualStrings("first.txt", md.get(0).?);
    try testing.expect(md.get(1) == null);
    try testing.expectEqualStrings("third.txt", md.get(2).?);
}

test "setAt on same id replaces without leaking" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "original.txt");
    try md.setAt(testing.allocator, 0, "replaced.txt");

    try testing.expectEqual(@as(usize, 1), md.live_count);
    try testing.expectEqualStrings("replaced.txt", md.get(0).?);
}

test "deleteAt frees and decrements" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "a.txt");
    try md.setAt(testing.allocator, 1, "b.txt");
    md.deleteAt(testing.allocator, 0);

    try testing.expectEqual(@as(usize, 1), md.live_count);
    try testing.expect(md.get(0) == null);
    try testing.expectEqualStrings("b.txt", md.get(1).?);
}

test "deleteAt of non-existent id is a no-op" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    md.deleteAt(testing.allocator, 7);
    try testing.expectEqual(@as(usize, 0), md.live_count);
}

test "setAt then deleteAt then setAt again" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "first");
    md.deleteAt(testing.allocator, 0);
    try md.setAt(testing.allocator, 0, "second");

    try testing.expectEqual(@as(usize, 1), md.live_count);
    try testing.expectEqualStrings("second", md.get(0).?);
}

test "save preserves ids and tombstones sparsely" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "a.txt"); // live
    try md.setAt(testing.allocator, 1, "b.txt"); // tombstoned
    try md.setAt(testing.allocator, 2, "c.txt"); // live
    try md.setAt(testing.allocator, 3, "d.txt"); // live
    md.deleteAt(testing.allocator, 1);

    try md.save(testing.allocator, tmp.dir, "m.hmtf", 4);

    var loaded = try metadata.load(testing.allocator, tmp.dir, "m.hmtf");
    defer loaded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), loaded.count);
    try testing.expectEqual(@as(usize, 3), loaded.live_count);
    try testing.expect(!loaded.isTombstone(0));
    try testing.expect(loaded.isTombstone(1));
    try testing.expect(!loaded.isTombstone(2));
    try testing.expect(!loaded.isTombstone(3));
    try testing.expectEqualStrings("a.txt", loaded.get(0));
    try testing.expectEqualStrings("c.txt", loaded.get(2));
    try testing.expectEqualStrings("d.txt", loaded.get(3));
}

test "save pads trailing tombstones when total_slots > slots.len" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "a.txt");
    // Store may have allocated id 1 without ever calling setAt (e.g. the
    // server writes the WAL record and inserts before setAt). Save still
    // has to keep slot 1 present as a tombstone so the graph's id 1 lines
    // up with the file.
    try md.save(testing.allocator, tmp.dir, "m.hmtf", 2);

    var loaded = try metadata.load(testing.allocator, tmp.dir, "m.hmtf");
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), loaded.count);
    try testing.expectEqual(@as(usize, 1), loaded.live_count);
    try testing.expect(!loaded.isTombstone(0));
    try testing.expect(loaded.isTombstone(1));
}

test "save with empty store writes empty file that round-trips" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.save(testing.allocator, tmp.dir, "empty.hmtf", 0);

    var loaded = try metadata.load(testing.allocator, tmp.dir, "empty.hmtf");
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), loaded.count);
}

test "ensureCapacity does not add live slots" {
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.ensureCapacity(testing.allocator, 1000);
    try testing.expectEqual(@as(usize, 0), md.live_count);
    try testing.expectEqual(@as(usize, 0), md.highWater());
}
