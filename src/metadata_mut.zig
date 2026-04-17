//! Server-side mutable analogue of `metadata.Metadata`. Stores the id →
//! filename mapping for a live (long-running) server that inserts and
//! deletes vectors incrementally.
//!
//! Each slot is an `?[]u8` owned by this module: `null` for tombstoned /
//! never-assigned ids, populated otherwise. IDs passed to `setAt` are
//! assigned by the `Store` (LIFO free-list or bump counter) so we simply
//! follow along; this module does not mint ids.
//!
//! On `save`, we accept a `remap[old_id] -> new_id | 0xFFFFFFFF` produced
//! by `Store.buildRemap` and write a compacted HMTF V1 file that the
//! existing `metadata.load` can read back without format changes.
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

    /// Write an HMTF V1 file compacted according to `remap`. `remap[i] ==
    /// 0xFFFFFFFF` means old id `i` is deleted; otherwise `remap[i]` is
    /// the new dense id. Slots are emitted in new-id order so the file
    /// round-trips through the static `metadata.load` unchanged.
    ///
    /// The caller is expected to pass a remap built from the same Store
    /// that drove the mutations — inconsistency is a programming error,
    /// not a runtime one, and is caught by assertions.
    pub fn save(
        self: *const Self,
        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        sub_path: []const u8,
        remap: []const u32,
    ) !void {
        // Gather non-deleted slots into new-id order.
        var names = try allocator.alloc([]const u8, self.live_count);
        defer allocator.free(names);

        var live_seen: usize = 0;
        for (remap, 0..) |new_id, old_id| {
            if (new_id == 0xFFFFFFFF) continue;
            std.debug.assert(new_id < self.live_count);
            std.debug.assert(old_id < self.slots.items.len);
            const slot = self.slots.items[old_id] orelse {
                // store says this id is live but metadata has no entry.
                // Emit an empty name; callers that care set a name on
                // every insert.
                names[new_id] = "";
                live_seen += 1;
                continue;
            };
            names[new_id] = slot;
            live_seen += 1;
        }
        std.debug.assert(live_seen == self.live_count);

        try metadata.save(dir, sub_path, names);
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

test "save compacts via remap and round-trips through metadata.load" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    try md.setAt(testing.allocator, 0, "a.txt"); // live
    try md.setAt(testing.allocator, 1, "b.txt"); // tombstoned
    try md.setAt(testing.allocator, 2, "c.txt"); // live
    try md.setAt(testing.allocator, 3, "d.txt"); // live
    md.deleteAt(testing.allocator, 1);

    // Simulates Store.buildRemap with id 1 deleted:
    //   old 0 -> new 0, old 1 -> deleted, old 2 -> new 1, old 3 -> new 2
    const remap = [_]u32{ 0, 0xFFFFFFFF, 1, 2 };
    try md.save(testing.allocator, tmp.dir, "m.hmtf", &remap);

    var loaded = try metadata.load(testing.allocator, tmp.dir, "m.hmtf");
    defer loaded.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), loaded.count);
    try testing.expectEqualStrings("a.txt", loaded.get(0));
    try testing.expectEqualStrings("c.txt", loaded.get(1));
    try testing.expectEqualStrings("d.txt", loaded.get(2));
}

test "save with empty store writes empty file that round-trips" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    const remap: []const u32 = &.{};
    try md.save(testing.allocator, tmp.dir, "empty.hmtf", remap);

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
