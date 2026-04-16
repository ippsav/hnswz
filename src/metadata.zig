//! Sidecar file mapping vector ids to the source filenames they were
//! embedded from. Written once at build time, loaded once at query time.
//!
//! Disk format (HMTF):
//!   magic:    [4]u8 = "HMTF"
//!   version:  u32   = 1
//!   count:    u64
//!   entries[count]:
//!     name_len: u16
//!     name:     [name_len]u8
//!
//! In-memory layout uses a single packed names buffer + an offsets table
//! so lookup is O(1) and there are no per-id allocations.
const std = @import("std");
const assert = std.debug.assert;

pub const Metadata = struct {
    const Self = @This();

    names_buf: []u8,
    offsets: []u32, // len == count + 1; name i is names_buf[offsets[i]..offsets[i+1]]
    count: usize,

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.names_buf);
        allocator.free(self.offsets);
    }

    pub fn get(self: *const Self, id: u32) []const u8 {
        assert(id < self.count);
        return self.names_buf[self.offsets[id]..self.offsets[id + 1]];
    }
};

/// Persist `names[i]` as the filename for id `i` (index = id).
pub fn save(dir: std.fs.Dir, sub_path: []const u8, names: []const []const u8) !void {
    const file = try dir.createFile(sub_path, .{});
    defer file.close();

    var wbuf: [8192]u8 = undefined;
    var bw = file.writer(&wbuf);
    const w = &bw.interface;

    // Header: magic(4) + version(4) + count(8) = 16 bytes
    var hdr: [16]u8 = undefined;
    @memcpy(hdr[0..4], "HMTF");
    std.mem.writeInt(u32, hdr[4..8], 1, .little);
    std.mem.writeInt(u64, hdr[8..16], @intCast(names.len), .little);
    try w.writeAll(&hdr);

    for (names) |name| {
        if (name.len > std.math.maxInt(u16)) return error.NameTooLong;
        var len_buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &len_buf, @intCast(name.len), .little);
        try w.writeAll(&len_buf);
        try w.writeAll(name);
    }

    try w.flush();
}

/// Load metadata. Caller owns the returned value.
pub fn load(allocator: std.mem.Allocator, dir: std.fs.Dir, sub_path: []const u8) !Metadata {
    const file = try dir.openFile(sub_path, .{});
    defer file.close();

    var rbuf: [8192]u8 = undefined;
    var br = file.readerStreaming(&rbuf);
    const r = &br.interface;

    var hdr: [16]u8 = undefined;
    try r.readSliceAll(&hdr);

    if (!std.mem.eql(u8, hdr[0..4], "HMTF")) return error.InvalidMagic;
    const version = std.mem.readInt(u32, hdr[4..8], .little);
    if (version != 1) return error.UnsupportedVersion;

    const count: usize = @intCast(std.mem.readInt(u64, hdr[8..16], .little));

    // Stat the file so we know how many bytes to allocate for names_buf
    // without pre-summing. The names section is everything after the
    // header minus 2 bytes per entry for the u16 length prefixes.
    const stat = try file.stat();
    const total_entries_size: u64 = @as(u64, stat.size) - @sizeOf(@TypeOf(hdr));
    const names_bytes: usize = @intCast(total_entries_size - count * @sizeOf(u16));

    const names_buf = try allocator.alloc(u8, names_bytes);
    errdefer allocator.free(names_buf);
    const offsets = try allocator.alloc(u32, count + 1);
    errdefer allocator.free(offsets);

    var cursor: usize = 0;
    for (0..count) |i| {
        var len_buf: [2]u8 = undefined;
        try r.readSliceAll(&len_buf);
        const name_len: usize = @intCast(std.mem.readInt(u16, &len_buf, .little));
        if (cursor + name_len > names_buf.len) return error.CorruptSizes;

        offsets[i] = @intCast(cursor);
        try r.readSliceAll(names_buf[cursor..][0..name_len]);
        cursor += name_len;
    }
    offsets[count] = @intCast(cursor);

    if (cursor != names_buf.len) return error.CorruptSizes;

    return .{
        .names_buf = names_buf,
        .offsets = offsets,
        .count = count,
    };
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "save and load round-trip" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const names = [_][]const u8{ "a.txt", "bravo.txt", "c-three.txt" };
    try save(tmp.dir, "m.hmtf", &names);

    var md = try load(testing.allocator, tmp.dir, "m.hmtf");
    defer md.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), md.count);
    try testing.expectEqualStrings("a.txt", md.get(0));
    try testing.expectEqualStrings("bravo.txt", md.get(1));
    try testing.expectEqualStrings("c-three.txt", md.get(2));
}

test "load rejects bad magic" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "bad.hmtf", .data = "XXXX\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" });
    try testing.expectError(error.InvalidMagic, load(testing.allocator, tmp.dir, "bad.hmtf"));
}

test "empty names list round-trips" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const empty: []const []const u8 = &.{};
    try save(tmp.dir, "e.hmtf", empty);

    var md = try load(testing.allocator, tmp.dir, "e.hmtf");
    defer md.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), md.count);
}
