//! Sidecar file mapping vector ids to the source filenames they were
//! embedded from.
//!
//! Disk format (HMTF v2):
//!   magic:      [4]u8 = "HMTF"
//!   version:    u32   = 2
//!   count:      u64   total slot count (matches store.count, incl. tombstones)
//!   live_count: u64   non-tombstoned slot count (matches store.live_count)
//!   entries[count]:
//!     flags:    u8    bit0: 1 = live, 0 = tombstone
//!     if live:
//!       name_len: u16
//!       name:     [name_len]u8
//!
//! In-memory layout keeps a packed names buffer + an offsets table so
//! lookup is O(1). Tombstoned slots point at a zero-length range; the
//! `tombstones` bitset is the authoritative liveness source.
const std = @import("std");
const assert = std.debug.assert;

pub const Metadata = struct {
    const Self = @This();

    names_buf: []u8,
    offsets: []u32, // len == count + 1; name i is names_buf[offsets[i]..offsets[i+1]]
    tombstones: std.DynamicBitSetUnmanaged,
    count: usize,
    live_count: usize,

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.names_buf);
        allocator.free(self.offsets);
        self.tombstones.deinit(allocator);
    }

    pub fn isTombstone(self: *const Self, id: u32) bool {
        return self.tombstones.isSet(id);
    }

    pub fn get(self: *const Self, id: u32) []const u8 {
        assert(id < self.count);
        return self.names_buf[self.offsets[id]..self.offsets[id + 1]];
    }
};

/// Persist `slots[i]` as the filename for id `i`; `null` means tombstone.
/// `live_count` must match the number of non-null slots (checked in
/// debug builds).
pub fn save(
    dir: std.fs.Dir,
    sub_path: []const u8,
    slots: []const ?[]const u8,
    live_count: usize,
) !void {
    const file = try dir.createFile(sub_path, .{});
    defer file.close();

    var wbuf: [8192]u8 = undefined;
    var bw = file.writer(&wbuf);
    const w = &bw.interface;

    var hdr: [24]u8 = undefined;
    @memcpy(hdr[0..4], "HMTF");
    std.mem.writeInt(u32, hdr[4..8], 2, .little);
    std.mem.writeInt(u64, hdr[8..16], @intCast(slots.len), .little);
    std.mem.writeInt(u64, hdr[16..24], @intCast(live_count), .little);
    try w.writeAll(&hdr);

    var seen_live: usize = 0;
    for (slots) |maybe_name| {
        if (maybe_name) |name| {
            if (name.len > std.math.maxInt(u16)) return error.NameTooLong;
            try w.writeByte(1);
            var len_buf: [2]u8 = undefined;
            std.mem.writeInt(u16, &len_buf, @intCast(name.len), .little);
            try w.writeAll(&len_buf);
            try w.writeAll(name);
            seen_live += 1;
        } else {
            try w.writeByte(0);
        }
    }
    assert(seen_live == live_count);

    try w.flush();
    try file.sync();
}

/// Load metadata. Caller owns the returned value.
pub fn load(allocator: std.mem.Allocator, dir: std.fs.Dir, sub_path: []const u8) !Metadata {
    const file = try dir.openFile(sub_path, .{});
    defer file.close();

    var rbuf: [8192]u8 = undefined;
    var br = file.readerStreaming(&rbuf);
    const r = &br.interface;

    var hdr: [24]u8 = undefined;
    try r.readSliceAll(&hdr);

    if (!std.mem.eql(u8, hdr[0..4], "HMTF")) return error.InvalidMagic;
    const version = std.mem.readInt(u32, hdr[4..8], .little);
    if (version != 2) return error.UnsupportedVersion;

    const count: usize = @intCast(std.mem.readInt(u64, hdr[8..16], .little));
    const live_count: usize = @intCast(std.mem.readInt(u64, hdr[16..24], .little));
    if (live_count > count) return error.InvalidFile;

    // File bytes after the header hold (flag + optional name_len + name)
    // per slot. We need the name bytes total, which is: file size - header
    // - flag bytes - (2 bytes per live slot for name_len).
    const stat = try file.stat();
    const body_size: u64 = @as(u64, stat.size) - @sizeOf(@TypeOf(hdr));
    const flag_and_len_overhead: u64 = count + @as(u64, live_count) * 2;
    if (body_size < flag_and_len_overhead) return error.CorruptSizes;
    const names_bytes: usize = @intCast(body_size - flag_and_len_overhead);

    const names_buf = try allocator.alloc(u8, names_bytes);
    errdefer allocator.free(names_buf);
    const offsets = try allocator.alloc(u32, count + 1);
    errdefer allocator.free(offsets);
    var tombstones = try std.DynamicBitSetUnmanaged.initEmpty(allocator, count);
    errdefer tombstones.deinit(allocator);

    var cursor: usize = 0;
    var seen_live: usize = 0;
    for (0..count) |i| {
        var flag_buf: [1]u8 = undefined;
        try r.readSliceAll(&flag_buf);
        offsets[i] = @intCast(cursor);
        if (flag_buf[0] == 0) {
            tombstones.set(@intCast(i));
            continue;
        }
        if (flag_buf[0] != 1) return error.InvalidFile;
        var len_buf: [2]u8 = undefined;
        try r.readSliceAll(&len_buf);
        const name_len: usize = @intCast(std.mem.readInt(u16, &len_buf, .little));
        if (cursor + name_len > names_buf.len) return error.CorruptSizes;
        try r.readSliceAll(names_buf[cursor..][0..name_len]);
        cursor += name_len;
        seen_live += 1;
    }
    offsets[count] = @intCast(cursor);

    if (cursor != names_buf.len) return error.CorruptSizes;
    if (seen_live != live_count) return error.InvalidFile;

    return .{
        .names_buf = names_buf,
        .offsets = offsets,
        .tombstones = tombstones,
        .count = count,
        .live_count = live_count,
    };
}



const testing = std.testing;

test "save and load round-trip" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const slots = [_]?[]const u8{ "a.txt", "bravo.txt", "c-three.txt" };
    try save(tmp.dir, "m.hmtf", &slots, 3);

    var md = try load(testing.allocator, tmp.dir, "m.hmtf");
    defer md.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), md.count);
    try testing.expectEqual(@as(usize, 3), md.live_count);
    try testing.expectEqualStrings("a.txt", md.get(0));
    try testing.expectEqualStrings("bravo.txt", md.get(1));
    try testing.expectEqualStrings("c-three.txt", md.get(2));
    try testing.expect(!md.isTombstone(0));
}

test "save preserves tombstones" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const slots = [_]?[]const u8{ "a.txt", null, "c.txt", null, "e.txt" };
    try save(tmp.dir, "m.hmtf", &slots, 3);

    var md = try load(testing.allocator, tmp.dir, "m.hmtf");
    defer md.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 5), md.count);
    try testing.expectEqual(@as(usize, 3), md.live_count);
    try testing.expect(!md.isTombstone(0));
    try testing.expect(md.isTombstone(1));
    try testing.expect(!md.isTombstone(2));
    try testing.expect(md.isTombstone(3));
    try testing.expect(!md.isTombstone(4));
    try testing.expectEqualStrings("a.txt", md.get(0));
    try testing.expectEqualStrings("c.txt", md.get(2));
    try testing.expectEqualStrings("e.txt", md.get(4));
}

test "load rejects bad magic" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "bad.hmtf", .data = "XXXX\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" });
    try testing.expectError(error.InvalidMagic, load(testing.allocator, tmp.dir, "bad.hmtf"));
}

test "empty slots list round-trips" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const empty: []const ?[]const u8 = &.{};
    try save(tmp.dir, "e.hmtf", empty, 0);

    var md = try load(testing.allocator, tmp.dir, "e.hmtf");
    defer md.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), md.count);
    try testing.expectEqual(@as(usize, 0), md.live_count);
}
