//! Wire format for the `hnswz serve` TCP protocol.
//!
//! Every frame — request and response — is a 9-byte header followed by an
//! opcode-specific payload. All multi-byte integers are little-endian,
//! matching the on-disk HVSF/HGRF/HMTF formats so serialization is uniform
//! across the project. `req_id` is client-chosen and echoed by the server
//! so future async/pipelined clients work without a format break.
//!
//! ```
//! Request frame:
//!   off  size  field
//!     0    4   body_len   u32   bytes that follow this field (= 5 + payload)
//!     4    1   opcode     u8
//!     5    4   req_id     u32
//!     9    N   payload    opcode-specific
//!
//! Response frame:
//!   off  size  field
//!     0    4   body_len   u32
//!     4    1   status     u8    0 = OK; nonzero = error (payload = msg)
//!     5    4   req_id     u32   echo
//!     9    N   payload    opcode-/status-specific
//! ```
//!
//! Encoders write payloads into caller-supplied buffers and return the
//! number of bytes written. Decoders take a payload slice and return a view
//! struct that borrows from the caller's buffer. Zero allocation throughout.
const std = @import("std");

/// Bumped only on incompatible wire changes. Advertised via STATS.
pub const PROTO_VERSION: u16 = 1;

/// Fixed prefix on every frame.
pub const FRAME_HEADER_SIZE: usize = 9;

/// Default cap on `body_len`. A malicious `0xFFFFFFFF` should never force
/// the server to allocate 4 GiB. Callers can lower this via ServeOptions.
pub const MAX_FRAME_BYTES_DEFAULT: u32 = 64 << 20;

/// Minimum valid `body_len` — just the opcode/status byte + req_id, no payload.
pub const MIN_BODY_LEN: u32 = 5;

/// Non-exhaustive so an unknown byte does not crash the decoder — the
/// dispatcher maps the `_` case to Status.unsupported_opcode.
pub const Opcode = enum(u8) {
    ping = 0x00,
    stats = 0x01,
    insert_vec = 0x10,
    insert_text = 0x11,
    delete = 0x12,
    replace_vec = 0x13,
    replace_text = 0x14,
    get = 0x20,
    search_vec = 0x30,
    search_text = 0x31,
    snapshot = 0x40,
    close = 0xFF,
    _,
};

pub const Status = enum(u8) {
    ok = 0,
    invalid_frame = 1,
    unsupported_opcode = 2,
    out_of_capacity = 3,
    invalid_id = 4,
    dim_mismatch = 5,
    text_too_long = 6,
    embed_failed = 7,
    io_error = 8,
    snapshot_failed = 9,
    top_k_too_large = 10,
    busy = 11,
    internal = 255,
    _,
};

pub fn statusMessage(s: Status) []const u8 {
    return switch (s) {
        .ok => "ok",
        .invalid_frame => "invalid frame",
        .unsupported_opcode => "unsupported opcode",
        .out_of_capacity => "store is at capacity",
        .invalid_id => "invalid id",
        .dim_mismatch => "vector dim does not match server dim",
        .text_too_long => "text exceeds max_text_bytes",
        .embed_failed => "embedder failed",
        .io_error => "server io error",
        .snapshot_failed => "snapshot failed",
        .top_k_too_large => "top_k exceeds ef_search",
        .busy => "server busy",
        .internal => "internal error",
        _ => "unknown status",
    };
}

pub const Header = struct {
    body_len: u32,
    /// Opcode byte on requests, status byte on responses. Caller decides.
    tag: u8,
    req_id: u32,
};

pub const HeaderError = error{ Truncated, InvalidFrame, FrameTooLarge };

pub fn encodeHeader(dst: []u8, body_len: u32, tag: u8, req_id: u32) void {
    std.debug.assert(dst.len >= FRAME_HEADER_SIZE);
    std.mem.writeInt(u32, dst[0..4], body_len, .little);
    dst[4] = tag;
    std.mem.writeInt(u32, dst[5..9], req_id, .little);
}

/// Parse the 9-byte header. Does NOT enforce the max-body-size policy;
/// callers pass `max_body_bytes` so tests and unbounded contexts can
/// bypass the check with `std.math.maxInt(u32)`.
pub fn decodeHeader(src: []const u8, max_body_bytes: u32) HeaderError!Header {
    if (src.len < FRAME_HEADER_SIZE) return error.Truncated;
    const body_len = std.mem.readInt(u32, src[0..4], .little);
    if (body_len < MIN_BODY_LEN) return error.InvalidFrame;
    if (body_len > max_body_bytes) return error.FrameTooLarge;
    return .{
        .body_len = body_len,
        .tag = src[4],
        .req_id = std.mem.readInt(u32, src[5..9], .little),
    };
}

/// Payload length given a valid header.
pub fn payloadLen(h: Header) usize {
    return @as(usize, h.body_len) - 5;
}

/// Total bytes to consume from the stream, header included.
pub fn totalFrameSize(h: Header) usize {
    return @as(usize, h.body_len) + 4;
}

inline fn readU16LE(bytes: []const u8) u16 {
    return std.mem.readInt(u16, bytes[0..2], .little);
}

inline fn readU32LE(bytes: []const u8) u32 {
    return std.mem.readInt(u32, bytes[0..4], .little);
}

inline fn readU64LE(bytes: []const u8) u64 {
    return std.mem.readInt(u64, bytes[0..8], .little);
}

inline fn readF32LE(bytes: []const u8) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
}

inline fn writeU16LE(dst: []u8, v: u16) void {
    std.mem.writeInt(u16, dst[0..2], v, .little);
}

inline fn writeU32LE(dst: []u8, v: u32) void {
    std.mem.writeInt(u32, dst[0..4], v, .little);
}

inline fn writeU64LE(dst: []u8, v: u64) void {
    std.mem.writeInt(u64, dst[0..8], v, .little);
}

inline fn writeF32LE(dst: []u8, v: f32) void {
    std.mem.writeInt(u32, dst[0..4], @bitCast(v), .little);
}

pub const DecodeError = error{ Truncated, DimMismatch, TextTooLong };

/// Write an error response payload: `u16 msg_len | utf8 message`.
/// Returns bytes written.
pub fn encodeErrorPayload(dst: []u8, msg: []const u8) usize {
    std.debug.assert(msg.len <= std.math.maxInt(u16));
    std.debug.assert(dst.len >= 2 + msg.len);
    writeU16LE(dst[0..2], @intCast(msg.len));
    @memcpy(dst[2..][0..msg.len], msg);
    return 2 + msg.len;
}

pub fn decodeErrorPayload(payload: []const u8) DecodeError![]const u8 {
    if (payload.len < 2) return error.Truncated;
    const n = readU16LE(payload[0..2]);
    if (payload.len < 2 + @as(usize, n)) return error.Truncated;
    return payload[2..][0..n];
}

pub const StatsResponse = struct {
    proto_version: u16,
    flags: u16,
    dim: u32,
    m: u16,
    live_count: u64,
    high_water: u64,
    upper_used: u64,
    max_upper_slots: u64,
    max_level: u8,
    has_entry_point: u8,
};

pub const STATS_RESPONSE_SIZE: usize = 2 + 2 + 4 + 2 + 8 + 8 + 8 + 8 + 1 + 1;

pub fn encodeStatsResponse(dst: []u8, s: StatsResponse) usize {
    std.debug.assert(dst.len >= STATS_RESPONSE_SIZE);
    writeU16LE(dst[0..2], s.proto_version);
    writeU16LE(dst[2..4], s.flags);
    writeU32LE(dst[4..8], s.dim);
    writeU16LE(dst[8..10], s.m);
    writeU64LE(dst[10..18], s.live_count);
    writeU64LE(dst[18..26], s.high_water);
    writeU64LE(dst[26..34], s.upper_used);
    writeU64LE(dst[34..42], s.max_upper_slots);
    dst[42] = s.max_level;
    dst[43] = s.has_entry_point;
    return STATS_RESPONSE_SIZE;
}

pub fn decodeStatsResponse(payload: []const u8) DecodeError!StatsResponse {
    if (payload.len < STATS_RESPONSE_SIZE) return error.Truncated;
    return .{
        .proto_version = readU16LE(payload[0..2]),
        .flags = readU16LE(payload[2..4]),
        .dim = readU32LE(payload[4..8]),
        .m = readU16LE(payload[8..10]),
        .live_count = readU64LE(payload[10..18]),
        .high_water = readU64LE(payload[18..26]),
        .upper_used = readU64LE(payload[26..34]),
        .max_upper_slots = readU64LE(payload[34..42]),
        .max_level = payload[42],
        .has_entry_point = payload[43],
    };
}

pub const InsertVecRequestView = struct {
    flags: u16,
    vec_bytes: []const u8, // length == dim * 4
};

/// Encode a `{flags, vec_bytes}` request payload. `vec_bytes` must be the
/// raw little-endian f32 representation of `dim` floats (i.e. `dim * 4`
/// bytes). Returns bytes written.
pub fn encodeInsertVecRequest(dst: []u8, flags: u16, vec_bytes: []const u8) usize {
    std.debug.assert(dst.len >= 2 + vec_bytes.len);
    writeU16LE(dst[0..2], flags);
    @memcpy(dst[2..][0..vec_bytes.len], vec_bytes);
    return 2 + vec_bytes.len;
}

/// Parse an INSERT_VEC request payload and validate that the vector size
/// matches `expected_dim`.
pub fn decodeInsertVecRequest(payload: []const u8, expected_dim: usize) DecodeError!InsertVecRequestView {
    if (payload.len < 2) return error.Truncated;
    const flags = readU16LE(payload[0..2]);
    const vec_bytes = payload[2..];
    if (vec_bytes.len != expected_dim * 4) return error.DimMismatch;
    return .{ .flags = flags, .vec_bytes = vec_bytes };
}

pub fn encodeIdResponse(dst: []u8, id: u32) usize {
    std.debug.assert(dst.len >= 4);
    writeU32LE(dst[0..4], id);
    return 4;
}

pub fn decodeIdResponse(payload: []const u8) DecodeError!u32 {
    if (payload.len < 4) return error.Truncated;
    return readU32LE(payload[0..4]);
}

pub const InsertTextRequestView = struct {
    flags: u16,
    text: []const u8,
};

pub fn encodeInsertTextRequest(dst: []u8, flags: u16, text: []const u8) usize {
    std.debug.assert(text.len <= std.math.maxInt(u16));
    std.debug.assert(dst.len >= 4 + text.len);
    writeU16LE(dst[0..2], flags);
    writeU16LE(dst[2..4], @intCast(text.len));
    @memcpy(dst[4..][0..text.len], text);
    return 4 + text.len;
}

pub fn decodeInsertTextRequest(payload: []const u8) DecodeError!InsertTextRequestView {
    if (payload.len < 4) return error.Truncated;
    const flags = readU16LE(payload[0..2]);
    const text_len = readU16LE(payload[2..4]);
    if (payload.len < 4 + @as(usize, text_len)) return error.Truncated;
    return .{ .flags = flags, .text = payload[4..][0..text_len] };
}

pub fn encodeIdRequest(dst: []u8, id: u32) usize {
    std.debug.assert(dst.len >= 4);
    writeU32LE(dst[0..4], id);
    return 4;
}

pub fn decodeIdRequest(payload: []const u8) DecodeError!u32 {
    if (payload.len < 4) return error.Truncated;
    return readU32LE(payload[0..4]);
}

pub const ReplaceVecRequestView = struct {
    id: u32,
    flags: u16,
    vec_bytes: []const u8,
};

pub fn encodeReplaceVecRequest(dst: []u8, id: u32, flags: u16, vec_bytes: []const u8) usize {
    std.debug.assert(dst.len >= 6 + vec_bytes.len);
    writeU32LE(dst[0..4], id);
    writeU16LE(dst[4..6], flags);
    @memcpy(dst[6..][0..vec_bytes.len], vec_bytes);
    return 6 + vec_bytes.len;
}

pub fn decodeReplaceVecRequest(payload: []const u8, expected_dim: usize) DecodeError!ReplaceVecRequestView {
    if (payload.len < 6) return error.Truncated;
    const id = readU32LE(payload[0..4]);
    const flags = readU16LE(payload[4..6]);
    const vec_bytes = payload[6..];
    if (vec_bytes.len != expected_dim * 4) return error.DimMismatch;
    return .{ .id = id, .flags = flags, .vec_bytes = vec_bytes };
}

pub const ReplaceTextRequestView = struct {
    id: u32,
    flags: u16,
    text: []const u8,
};

pub fn encodeReplaceTextRequest(dst: []u8, id: u32, flags: u16, text: []const u8) usize {
    std.debug.assert(text.len <= std.math.maxInt(u16));
    std.debug.assert(dst.len >= 8 + text.len);
    writeU32LE(dst[0..4], id);
    writeU16LE(dst[4..6], flags);
    writeU16LE(dst[6..8], @intCast(text.len));
    @memcpy(dst[8..][0..text.len], text);
    return 8 + text.len;
}

pub fn decodeReplaceTextRequest(payload: []const u8) DecodeError!ReplaceTextRequestView {
    if (payload.len < 8) return error.Truncated;
    const id = readU32LE(payload[0..4]);
    const flags = readU16LE(payload[4..6]);
    const text_len = readU16LE(payload[6..8]);
    if (payload.len < 8 + @as(usize, text_len)) return error.Truncated;
    return .{ .id = id, .flags = flags, .text = payload[8..][0..text_len] };
}

pub const GetResponseView = struct {
    name: []const u8,
    vec_bytes: []const u8,
};

pub fn encodeGetResponse(dst: []u8, name: []const u8, vec_bytes: []const u8) usize {
    std.debug.assert(name.len <= std.math.maxInt(u16));
    std.debug.assert(dst.len >= 2 + name.len + vec_bytes.len);
    writeU16LE(dst[0..2], @intCast(name.len));
    @memcpy(dst[2..][0..name.len], name);
    @memcpy(dst[2 + name.len ..][0..vec_bytes.len], vec_bytes);
    return 2 + name.len + vec_bytes.len;
}

pub fn decodeGetResponse(payload: []const u8, expected_dim: usize) DecodeError!GetResponseView {
    if (payload.len < 2) return error.Truncated;
    const name_len = readU16LE(payload[0..2]);
    const vec_start = 2 + @as(usize, name_len);
    if (payload.len < vec_start + expected_dim * 4) return error.Truncated;
    return .{
        .name = payload[2..vec_start],
        .vec_bytes = payload[vec_start..][0 .. expected_dim * 4],
    };
}

pub const SearchVecRequestView = struct {
    top_k: u16,
    ef: u16,
    flags: u16,
    vec_bytes: []const u8,
};

pub fn encodeSearchVecRequest(
    dst: []u8,
    top_k: u16,
    ef: u16,
    flags: u16,
    vec_bytes: []const u8,
) usize {
    std.debug.assert(dst.len >= 6 + vec_bytes.len);
    writeU16LE(dst[0..2], top_k);
    writeU16LE(dst[2..4], ef);
    writeU16LE(dst[4..6], flags);
    @memcpy(dst[6..][0..vec_bytes.len], vec_bytes);
    return 6 + vec_bytes.len;
}

pub fn decodeSearchVecRequest(payload: []const u8, expected_dim: usize) DecodeError!SearchVecRequestView {
    if (payload.len < 6) return error.Truncated;
    const top_k = readU16LE(payload[0..2]);
    const ef = readU16LE(payload[2..4]);
    const flags = readU16LE(payload[4..6]);
    const vec_bytes = payload[6..];
    if (vec_bytes.len != expected_dim * 4) return error.DimMismatch;
    return .{ .top_k = top_k, .ef = ef, .flags = flags, .vec_bytes = vec_bytes };
}

pub const SearchTextRequestView = struct {
    top_k: u16,
    ef: u16,
    flags: u16,
    text: []const u8,
};

pub fn encodeSearchTextRequest(
    dst: []u8,
    top_k: u16,
    ef: u16,
    flags: u16,
    text: []const u8,
) usize {
    std.debug.assert(text.len <= std.math.maxInt(u16));
    std.debug.assert(dst.len >= 8 + text.len);
    writeU16LE(dst[0..2], top_k);
    writeU16LE(dst[2..4], ef);
    writeU16LE(dst[4..6], flags);
    writeU16LE(dst[6..8], @intCast(text.len));
    @memcpy(dst[8..][0..text.len], text);
    return 8 + text.len;
}

pub fn decodeSearchTextRequest(payload: []const u8) DecodeError!SearchTextRequestView {
    if (payload.len < 8) return error.Truncated;
    const top_k = readU16LE(payload[0..2]);
    const ef = readU16LE(payload[2..4]);
    const flags = readU16LE(payload[4..6]);
    const text_len = readU16LE(payload[6..8]);
    if (payload.len < 8 + @as(usize, text_len)) return error.Truncated;
    return .{ .top_k = top_k, .ef = ef, .flags = flags, .text = payload[8..][0..text_len] };
}

// Search response: `u16 n | { u32 id, f32 dist, u16 name_len, name }[n]`
//
// The variable-size, interleaved layout rules out a single fixed decoder,
// so the server-side encoder writes results one at a time into the send
// buffer and the client-side decoder returns an iterator-ish view.

/// Write the leading `u16 n` count. Returns bytes written (always 2).
pub fn writeSearchResultCount(dst: []u8, n: u16) usize {
    std.debug.assert(dst.len >= 2);
    writeU16LE(dst[0..2], n);
    return 2;
}

/// Write one result entry. Returns bytes written.
pub fn writeSearchResult(dst: []u8, id: u32, dist: f32, name: []const u8) usize {
    std.debug.assert(name.len <= std.math.maxInt(u16));
    std.debug.assert(dst.len >= 4 + 4 + 2 + name.len);
    writeU32LE(dst[0..4], id);
    writeF32LE(dst[4..8], dist);
    writeU16LE(dst[8..10], @intCast(name.len));
    @memcpy(dst[10..][0..name.len], name);
    return 10 + name.len;
}

pub const SearchResult = struct {
    id: u32,
    dist: f32,
    name: []const u8,
};

/// Iterates over a search response payload, yielding `SearchResult` views
/// that borrow from the payload. Check `.err` after iteration completes to
/// catch truncation.
pub const SearchResultIter = struct {
    payload: []const u8,
    cursor: usize,
    remaining: u16,
    err: ?DecodeError = null,

    pub fn next(self: *SearchResultIter) ?SearchResult {
        if (self.remaining == 0) return null;
        if (self.payload.len < self.cursor + 10) {
            self.err = error.Truncated;
            self.remaining = 0;
            return null;
        }
        const id = readU32LE(self.payload[self.cursor .. self.cursor + 4]);
        const dist = readF32LE(self.payload[self.cursor + 4 .. self.cursor + 8]);
        const name_len = readU16LE(self.payload[self.cursor + 8 .. self.cursor + 10]);
        if (self.payload.len < self.cursor + 10 + @as(usize, name_len)) {
            self.err = error.Truncated;
            self.remaining = 0;
            return null;
        }
        const name = self.payload[self.cursor + 10 .. self.cursor + 10 + name_len];
        self.cursor += 10 + name_len;
        self.remaining -= 1;
        return .{ .id = id, .dist = dist, .name = name };
    }
};

pub fn searchResultIter(payload: []const u8) DecodeError!SearchResultIter {
    if (payload.len < 2) return error.Truncated;
    const n = readU16LE(payload[0..2]);
    return .{ .payload = payload, .cursor = 2, .remaining = n };
}

pub fn encodeSnapshotResponse(dst: []u8, elapsed_ns: u64) usize {
    std.debug.assert(dst.len >= 8);
    writeU64LE(dst[0..8], elapsed_ns);
    return 8;
}

pub fn decodeSnapshotResponse(payload: []const u8) DecodeError!u64 {
    if (payload.len < 8) return error.Truncated;
    return readU64LE(payload[0..8]);
}

const testing = std.testing;

test "encodeHeader / decodeHeader round-trip" {
    var buf: [FRAME_HEADER_SIZE]u8 = undefined;
    encodeHeader(&buf, 100, 0x10, 0xDEADBEEF);
    const h = try decodeHeader(&buf, MAX_FRAME_BYTES_DEFAULT);
    try testing.expectEqual(@as(u32, 100), h.body_len);
    try testing.expectEqual(@as(u8, 0x10), h.tag);
    try testing.expectEqual(@as(u32, 0xDEADBEEF), h.req_id);
}

test "decodeHeader rejects truncated input" {
    var buf: [FRAME_HEADER_SIZE - 1]u8 = undefined;
    try testing.expectError(error.Truncated, decodeHeader(&buf, MAX_FRAME_BYTES_DEFAULT));
}

test "decodeHeader rejects body_len below MIN_BODY_LEN" {
    var buf: [FRAME_HEADER_SIZE]u8 = undefined;
    encodeHeader(&buf, 4, 0x00, 0);
    try testing.expectError(error.InvalidFrame, decodeHeader(&buf, MAX_FRAME_BYTES_DEFAULT));
}

test "decodeHeader rejects oversized frames" {
    var buf: [FRAME_HEADER_SIZE]u8 = undefined;
    encodeHeader(&buf, 1 << 20, 0x00, 0);
    try testing.expectError(error.FrameTooLarge, decodeHeader(&buf, 1 << 10));
}

test "totalFrameSize and payloadLen" {
    const h: Header = .{ .body_len = 20, .tag = 0x10, .req_id = 7 };
    try testing.expectEqual(@as(usize, 24), totalFrameSize(h));
    try testing.expectEqual(@as(usize, 15), payloadLen(h));
}

test "encode/decodeErrorPayload" {
    var buf: [128]u8 = undefined;
    const msg = "vector dim does not match server dim";
    const n = encodeErrorPayload(&buf, msg);
    const decoded = try decodeErrorPayload(buf[0..n]);
    try testing.expectEqualStrings(msg, decoded);
}

test "encode/decodeStatsResponse" {
    var buf: [STATS_RESPONSE_SIZE]u8 = undefined;
    const s: StatsResponse = .{
        .proto_version = 1,
        .flags = 0,
        .dim = 4096,
        .m = 16,
        .live_count = 12345,
        .high_water = 12400,
        .upper_used = 800,
        .max_upper_slots = 10_000,
        .max_level = 4,
        .has_entry_point = 1,
    };
    const n = encodeStatsResponse(&buf, s);
    try testing.expectEqual(STATS_RESPONSE_SIZE, n);
    const decoded = try decodeStatsResponse(&buf);
    try testing.expectEqual(s.proto_version, decoded.proto_version);
    try testing.expectEqual(s.dim, decoded.dim);
    try testing.expectEqual(s.m, decoded.m);
    try testing.expectEqual(s.live_count, decoded.live_count);
    try testing.expectEqual(s.high_water, decoded.high_water);
    try testing.expectEqual(s.upper_used, decoded.upper_used);
    try testing.expectEqual(s.max_upper_slots, decoded.max_upper_slots);
    try testing.expectEqual(s.max_level, decoded.max_level);
    try testing.expectEqual(s.has_entry_point, decoded.has_entry_point);
}

test "encode/decodeInsertVecRequest" {
    const vec: [4]f32 = .{ 1.0, -0.5, 2.5, 0.0 };
    const vec_bytes = std.mem.sliceAsBytes(vec[0..]);

    var buf: [2 + 4 * 4]u8 = undefined;
    const n = encodeInsertVecRequest(&buf, 0, vec_bytes);
    try testing.expectEqual(@as(usize, 2 + 16), n);

    const view = try decodeInsertVecRequest(buf[0..n], 4);
    try testing.expectEqual(@as(u16, 0), view.flags);
    try testing.expectEqualSlices(u8, vec_bytes, view.vec_bytes);
}

test "decodeInsertVecRequest rejects dim mismatch" {
    const vec: [3]f32 = .{ 1.0, 2.0, 3.0 };
    const vec_bytes = std.mem.sliceAsBytes(vec[0..]);
    var buf: [2 + 3 * 4]u8 = undefined;
    const n = encodeInsertVecRequest(&buf, 0, vec_bytes);
    try testing.expectError(error.DimMismatch, decodeInsertVecRequest(buf[0..n], 4));
}

test "decodeInsertVecRequest rejects truncated flags" {
    var buf: [1]u8 = undefined;
    try testing.expectError(error.Truncated, decodeInsertVecRequest(&buf, 4));
}

test "encode/decodeIdResponse" {
    var buf: [4]u8 = undefined;
    _ = encodeIdResponse(&buf, 42);
    try testing.expectEqual(@as(u32, 42), try decodeIdResponse(&buf));
}

test "encode/decodeInsertTextRequest" {
    const text = "hello world";
    var buf: [4 + 11]u8 = undefined;
    const n = encodeInsertTextRequest(&buf, 7, text);
    try testing.expectEqual(@as(usize, 15), n);
    const view = try decodeInsertTextRequest(buf[0..n]);
    try testing.expectEqual(@as(u16, 7), view.flags);
    try testing.expectEqualStrings(text, view.text);
}

test "decodeInsertTextRequest rejects inconsistent length" {
    var buf: [4]u8 = .{ 0, 0, 10, 0 }; // claims 10 bytes of text but has 0
    try testing.expectError(error.Truncated, decodeInsertTextRequest(&buf));
}

test "encode/decodeIdRequest" {
    var buf: [4]u8 = undefined;
    _ = encodeIdRequest(&buf, 0xABCDEFAB);
    try testing.expectEqual(@as(u32, 0xABCDEFAB), try decodeIdRequest(&buf));
}

test "encode/decodeReplaceVecRequest" {
    const vec: [2]f32 = .{ 9.0, 10.0 };
    const vec_bytes = std.mem.sliceAsBytes(vec[0..]);
    var buf: [6 + 8]u8 = undefined;
    const n = encodeReplaceVecRequest(&buf, 99, 0, vec_bytes);
    const view = try decodeReplaceVecRequest(buf[0..n], 2);
    try testing.expectEqual(@as(u32, 99), view.id);
    try testing.expectEqualSlices(u8, vec_bytes, view.vec_bytes);
}

test "encode/decodeReplaceTextRequest" {
    const text = "rewrite";
    var buf: [8 + 7]u8 = undefined;
    const n = encodeReplaceTextRequest(&buf, 77, 0, text);
    const view = try decodeReplaceTextRequest(buf[0..n]);
    try testing.expectEqual(@as(u32, 77), view.id);
    try testing.expectEqualStrings(text, view.text);
}

test "encode/decodeGetResponse" {
    const vec: [2]f32 = .{ 3.14, 2.71 };
    const vec_bytes = std.mem.sliceAsBytes(vec[0..]);
    const name = "doc.txt";
    var buf: [2 + 7 + 8]u8 = undefined;
    const n = encodeGetResponse(&buf, name, vec_bytes);
    const view = try decodeGetResponse(buf[0..n], 2);
    try testing.expectEqualStrings(name, view.name);
    try testing.expectEqualSlices(u8, vec_bytes, view.vec_bytes);
}

test "encode/decodeSearchVecRequest" {
    const vec: [4]f32 = .{ 0.1, 0.2, 0.3, 0.4 };
    const vec_bytes = std.mem.sliceAsBytes(vec[0..]);
    var buf: [6 + 16]u8 = undefined;
    const n = encodeSearchVecRequest(&buf, 10, 100, 0, vec_bytes);
    const view = try decodeSearchVecRequest(buf[0..n], 4);
    try testing.expectEqual(@as(u16, 10), view.top_k);
    try testing.expectEqual(@as(u16, 100), view.ef);
    try testing.expectEqualSlices(u8, vec_bytes, view.vec_bytes);
}

test "encode/decodeSearchTextRequest" {
    const text = "find me";
    var buf: [8 + 7]u8 = undefined;
    const n = encodeSearchTextRequest(&buf, 5, 50, 0, text);
    const view = try decodeSearchTextRequest(buf[0..n]);
    try testing.expectEqual(@as(u16, 5), view.top_k);
    try testing.expectEqual(@as(u16, 50), view.ef);
    try testing.expectEqualStrings(text, view.text);
}

test "writeSearchResult / SearchResultIter round-trip" {
    var buf: [256]u8 = undefined;
    var cursor: usize = 0;
    cursor += writeSearchResultCount(buf[cursor..], 2);
    cursor += writeSearchResult(buf[cursor..], 7, 0.25, "a.txt");
    cursor += writeSearchResult(buf[cursor..], 42, 1.75, "bravo.txt");

    var iter = try searchResultIter(buf[0..cursor]);
    const r0 = iter.next().?;
    try testing.expectEqual(@as(u32, 7), r0.id);
    try testing.expectApproxEqAbs(@as(f32, 0.25), r0.dist, 1e-7);
    try testing.expectEqualStrings("a.txt", r0.name);

    const r1 = iter.next().?;
    try testing.expectEqual(@as(u32, 42), r1.id);
    try testing.expectApproxEqAbs(@as(f32, 1.75), r1.dist, 1e-7);
    try testing.expectEqualStrings("bravo.txt", r1.name);

    try testing.expect(iter.next() == null);
    try testing.expect(iter.err == null);
}

test "SearchResultIter reports Truncated on short payload" {
    var buf: [4]u8 = .{ 2, 0, 0, 0 }; // claims 2 results, has none
    var iter = try searchResultIter(&buf);
    try testing.expect(iter.next() == null);
    try testing.expectEqual(@as(?DecodeError, error.Truncated), iter.err);
}

test "encode/decodeSnapshotResponse" {
    var buf: [8]u8 = undefined;
    _ = encodeSnapshotResponse(&buf, 1_234_567);
    try testing.expectEqual(@as(u64, 1_234_567), try decodeSnapshotResponse(&buf));
}

test "Opcode enum is non-exhaustive (unknown byte decodes)" {
    // A byte not in the opcode list must not crash; dispatcher handles it.
    const op: Opcode = @enumFromInt(0x77);
    switch (op) {
        .ping, .stats, .insert_vec, .insert_text, .delete, .replace_vec,
        .replace_text, .get, .search_vec, .search_text, .snapshot, .close => unreachable,
        _ => {},
    }
}

test "Status enum is non-exhaustive" {
    const s: Status = @enumFromInt(0x88);
    switch (s) {
        .ok, .invalid_frame, .unsupported_opcode, .out_of_capacity, .invalid_id,
        .dim_mismatch, .text_too_long, .embed_failed, .io_error, .snapshot_failed,
        .top_k_too_large, .busy, .internal => unreachable,
        _ => {},
    }
}

test "decodeHeader fuzz: random bytes never panic" {
    var prng = std.Random.DefaultPrng.init(0x1234_5678);
    const random = prng.random();
    var buf: [FRAME_HEADER_SIZE]u8 = undefined;
    var i: usize = 0;
    while (i < 10_000) : (i += 1) {
        for (&buf) |*b| b.* = random.int(u8);
        _ = decodeHeader(&buf, MAX_FRAME_BYTES_DEFAULT) catch {};
    }
}

test "decode*Payload fuzz: random payloads never panic" {
    var prng = std.Random.DefaultPrng.init(0xBEEF_0001);
    const random = prng.random();
    var buf: [1024]u8 = undefined;
    var i: usize = 0;
    while (i < 2_000) : (i += 1) {
        const len = random.intRangeAtMost(usize, 0, buf.len);
        const slice = buf[0..len];
        for (slice) |*b| b.* = random.int(u8);

        _ = decodeErrorPayload(slice) catch {};
        _ = decodeStatsResponse(slice) catch {};
        _ = decodeInsertVecRequest(slice, 128) catch {};
        _ = decodeInsertTextRequest(slice) catch {};
        _ = decodeIdRequest(slice) catch {};
        _ = decodeIdResponse(slice) catch {};
        _ = decodeReplaceVecRequest(slice, 128) catch {};
        _ = decodeReplaceTextRequest(slice) catch {};
        _ = decodeGetResponse(slice, 128) catch {};
        _ = decodeSearchVecRequest(slice, 128) catch {};
        _ = decodeSearchTextRequest(slice) catch {};
        _ = decodeSnapshotResponse(slice) catch {};
        if (searchResultIter(slice)) |*iter_val| {
            var it = iter_val.*;
            while (it.next()) |_| {}
        } else |_| {}
    }
}
