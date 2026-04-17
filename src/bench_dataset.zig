//! Loader for the SIFT-family .fvecs / .ivecs ANN benchmark corpora.
//!
//! Files are a flat stream of records. Each record is:
//!   * .fvecs: i32 little-endian dim, then dim * f32 little-endian
//!   * .ivecs: i32 little-endian dim, then dim * i32 little-endian
//!
//! Every record in a single file shares the same dim — we assert that and
//! read the dim from the first record. The `_base.fvecs`, `_query.fvecs`,
//! `_groundtruth.ivecs` triple is produced by the standard SIFT1M /
//! siftsmall / GIST1M distributions; we accept either the prefixed form
//! (e.g. `sift_base.fvecs`) or the bare form (`base.fvecs`).
//!
//! Base and query vectors are L2-normalized on load so they play nicely
//! with the cosine-on-unit-vectors distance the index ships. For unit
//! vectors, L2 and cosine induce the same nearest-neighbor ordering, so
//! SIFT's L2-ground-truth remains valid.

const std = @import("std");
const bruteforce = @import("bruteforce.zig");

pub const DatasetError = error{
    DatasetDirNotFound,
    BaseFileMissing,
    QueryFileMissing,
    InconsistentDim,
    MalformedVecsFile,
    DimMismatch,
};

pub const Dataset = struct {
    allocator: std.mem.Allocator,
    dim: usize,
    num_base: usize,
    num_queries: usize,
    /// Flat row-major [num_base × dim], L2-normalized.
    base: []f32,
    /// Flat row-major [num_queries × dim], L2-normalized.
    queries: []f32,
    /// Flat row-major [num_queries × truth_k] if `<dir>/*groundtruth.ivecs`
    /// was present; else null. Always represents the *top-k* nearest by
    /// L2 on the original (un-normalized) SIFT vectors — consistent with
    /// cosine ordering on the normalized copies we store.
    truth: ?[]u32,
    truth_k: usize,

    pub fn deinit(self: *Dataset) void {
        self.allocator.free(self.base);
        self.allocator.free(self.queries);
        if (self.truth) |t| self.allocator.free(t);
        self.* = undefined;
    }
};

/// Scan `dir_path` for a `*base.fvecs` / `*query.fvecs` / `*groundtruth.ivecs`
/// triple, load and normalize the vectors, and return the resulting
/// `Dataset`. Groundtruth is optional. Fails fast if base or query is
/// absent.
pub fn load(allocator: std.mem.Allocator, dir_path: []const u8) !Dataset {
    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |e| switch (e) {
        error.FileNotFound, error.NotDir => return error.DatasetDirNotFound,
        else => return e,
    };
    defer dir.close();

    var base_name: ?[]u8 = null;
    defer if (base_name) |n| allocator.free(n);
    var query_name: ?[]u8 = null;
    defer if (query_name) |n| allocator.free(n);
    var truth_name: ?[]u8 = null;
    defer if (truth_name) |n| allocator.free(n);

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, "base.fvecs") and base_name == null) {
            base_name = try allocator.dupe(u8, entry.name);
        } else if (std.mem.endsWith(u8, entry.name, "query.fvecs") and query_name == null) {
            query_name = try allocator.dupe(u8, entry.name);
        } else if (std.mem.endsWith(u8, entry.name, "groundtruth.ivecs") and truth_name == null) {
            truth_name = try allocator.dupe(u8, entry.name);
        }
    }

    const base_n = base_name orelse return error.BaseFileMissing;
    const query_n = query_name orelse return error.QueryFileMissing;

    const base_result = try readFvecs(allocator, dir, base_n);
    errdefer allocator.free(base_result.data);
    const query_result = try readFvecs(allocator, dir, query_n);
    errdefer allocator.free(query_result.data);

    if (base_result.dim != query_result.dim) return error.DimMismatch;

    // Normalize in place — cosine-on-unit-vectors preserves L2 ordering,
    // and the index's distance kernel assumes unit inputs.
    normalizeAll(base_result.data, base_result.dim);
    normalizeAll(query_result.data, query_result.dim);

    var truth_data: ?[]u32 = null;
    var truth_k: usize = 0;
    if (truth_name) |tn| {
        const ivecs = try readIvecs(allocator, dir, tn);
        errdefer allocator.free(ivecs.data);
        if (ivecs.n != query_result.n) {
            allocator.free(ivecs.data);
            return error.InconsistentDim;
        }
        truth_data = ivecs.data;
        truth_k = ivecs.dim;
    }

    return .{
        .allocator = allocator,
        .dim = base_result.dim,
        .num_base = base_result.n,
        .num_queries = query_result.n,
        .base = base_result.data,
        .queries = query_result.data,
        .truth = truth_data,
        .truth_k = truth_k,
    };
}

fn normalizeAll(buf: []f32, dim: usize) void {
    var i: usize = 0;
    while (i + dim <= buf.len) : (i += dim) {
        bruteforce.normalize(buf[i..][0..dim]);
    }
}

const FvecsResult = struct { data: []f32, n: usize, dim: usize };
const IvecsResult = struct { data: []u32, n: usize, dim: usize };

fn readFvecs(allocator: std.mem.Allocator, dir: std.fs.Dir, name: []const u8) !FvecsResult {
    var file = try dir.openFile(name, .{});
    defer file.close();
    const stat = try file.stat();
    const file_size = stat.size;

    if (file_size < 4) return error.MalformedVecsFile;

    var reader_buf: [64 * 1024]u8 = undefined;
    var file_reader = file.reader(&reader_buf);
    const r = &file_reader.interface;

    // First record's dim header doubles as the file-wide dim.
    var first_hdr: [4]u8 = undefined;
    try r.readSliceAll(&first_hdr);
    const dim_i32: i32 = @bitCast(std.mem.readInt(i32, &first_hdr, .little));
    if (dim_i32 <= 0) return error.MalformedVecsFile;
    const dim: usize = @intCast(dim_i32);

    const record_bytes = 4 + dim * 4;
    if (file_size % record_bytes != 0) return error.MalformedVecsFile;
    const n: usize = @intCast(file_size / record_bytes);

    const data = try allocator.alloc(f32, n * dim);
    errdefer allocator.free(data);

    // First record: the dim header is already consumed; read payload.
    try r.readSliceAll(std.mem.sliceAsBytes(data[0..dim]));

    // Remaining records: header + payload.
    var vi: usize = 1;
    while (vi < n) : (vi += 1) {
        var hdr: [4]u8 = undefined;
        try r.readSliceAll(&hdr);
        const this_dim: i32 = @bitCast(std.mem.readInt(i32, &hdr, .little));
        if (this_dim != dim_i32) return error.InconsistentDim;

        const slot = data[vi * dim ..][0..dim];
        try r.readSliceAll(std.mem.sliceAsBytes(slot));
    }

    return .{ .data = data, .n = n, .dim = dim };
}

fn readIvecs(allocator: std.mem.Allocator, dir: std.fs.Dir, name: []const u8) !IvecsResult {
    var file = try dir.openFile(name, .{});
    defer file.close();
    const stat = try file.stat();
    const file_size = stat.size;

    if (file_size < 4) return error.MalformedVecsFile;

    var reader_buf: [64 * 1024]u8 = undefined;
    var file_reader = file.reader(&reader_buf);
    const r = &file_reader.interface;

    var first_hdr: [4]u8 = undefined;
    try r.readSliceAll(&first_hdr);
    const dim_i32: i32 = @bitCast(std.mem.readInt(i32, &first_hdr, .little));
    if (dim_i32 <= 0) return error.MalformedVecsFile;
    const dim: usize = @intCast(dim_i32);

    const record_bytes = 4 + dim * 4;
    if (file_size % record_bytes != 0) return error.MalformedVecsFile;
    const n: usize = @intCast(file_size / record_bytes);

    const data = try allocator.alloc(u32, n * dim);
    errdefer allocator.free(data);

    try r.readSliceAll(std.mem.sliceAsBytes(data[0..dim]));

    var vi: usize = 1;
    while (vi < n) : (vi += 1) {
        var hdr: [4]u8 = undefined;
        try r.readSliceAll(&hdr);
        const this_dim: i32 = @bitCast(std.mem.readInt(i32, &hdr, .little));
        if (this_dim != dim_i32) return error.InconsistentDim;

        const slot = data[vi * dim ..][0..dim];
        try r.readSliceAll(std.mem.sliceAsBytes(slot));
    }

    return .{ .data = data, .n = n, .dim = dim };
}

const testing = std.testing;

test "readFvecs round-trips a hand-written file" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Two records of dim=3.
    const f = try tmp.dir.createFile("sample_base.fvecs", .{});
    var w_buf: [64]u8 = undefined;
    var fw = f.writer(&w_buf);
    const w = &fw.interface;
    const Record = extern struct { dim: i32, v: [3]f32 };
    const r0: Record = .{ .dim = 3, .v = .{ 1.0, 2.0, 2.0 } };
    const r1: Record = .{ .dim = 3, .v = .{ 0.0, 0.0, 1.0 } };
    try w.writeAll(std.mem.asBytes(&r0));
    try w.writeAll(std.mem.asBytes(&r1));
    try w.flush();
    f.close();

    // Minimal queries file so load() succeeds.
    const fq = try tmp.dir.createFile("sample_query.fvecs", .{});
    var q_buf: [64]u8 = undefined;
    var fqw = fq.writer(&q_buf);
    const qw = &fqw.interface;
    const q0: Record = .{ .dim = 3, .v = .{ 1.0, 0.0, 0.0 } };
    try qw.writeAll(std.mem.asBytes(&q0));
    try qw.flush();
    fq.close();

    const sub_path = try tmp.dir.realpathAlloc(testing.allocator, ".");
    defer testing.allocator.free(sub_path);

    var ds = try load(testing.allocator, sub_path);
    defer ds.deinit();

    try testing.expectEqual(@as(usize, 3), ds.dim);
    try testing.expectEqual(@as(usize, 2), ds.num_base);
    try testing.expectEqual(@as(usize, 1), ds.num_queries);
    try testing.expect(ds.truth == null);

    // First base vector was (1,2,2) → unit-normalized to (1/3, 2/3, 2/3).
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), ds.base[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), ds.base[1], 1e-5);
}

test "readFvecs rejects malformed file" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // 5 bytes — not a valid record.
    const f = try tmp.dir.createFile("bad_base.fvecs", .{});
    try f.writeAll("\x03\x00\x00\x00\x00");
    f.close();
    const fq = try tmp.dir.createFile("bad_query.fvecs", .{});
    try fq.writeAll("\x03\x00\x00\x00\x00");
    fq.close();

    const sub_path = try tmp.dir.realpathAlloc(testing.allocator, ".");
    defer testing.allocator.free(sub_path);

    try testing.expectError(error.MalformedVecsFile, load(testing.allocator, sub_path));
}
