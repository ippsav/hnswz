//! Blocking TCP client for `hnswz serve`.
//!
//! This is the counterpart to `server.zig`. It exists primarily so tests
//! and the benchmark harness can drive the server with the same protocol
//! implementation the server speaks. External users wanting a client in
//! another language re-implement `protocol.zig`'s framing — it's <300
//! lines and has no dependencies.
//!
//! The client pre-allocates its request and response buffers at connect
//! time; no allocation happens per request. `search*` returns an owned
//! slice of `SearchResult` copied into caller-owned memory because the
//! underlying response bytes get reused on the next call.
const std = @import("std");

const protocol = @import("protocol.zig");
const heap = @import("heap.zig");

pub const ClientError = error{
    /// Server returned a status byte this client doesn't know how to
    /// translate. Check `last_status` on the client for the raw byte.
    UnknownStatus,
    /// The server closed the connection mid-frame.
    UnexpectedEof,
    /// Response exceeded the client's receive buffer.
    ResponseTooLarge,
    /// Server returned a non-OK status. `last_status` on the client
    /// holds the specific code and `last_message` holds the diagnostic.
    ServerError,
};

pub const ClientSearchResult = struct {
    id: u32,
    dist: f32,
    name: []u8, // owned
};

pub const Stats = protocol.StatsResponse;

pub const GetResult = struct {
    name: []u8, // owned
    vec: []f32, // owned
};

pub const ClientOptions = struct {
    /// Max bytes we'll accept in a single response frame.
    recv_buf_size: usize = 1 << 20,
    /// Upper bound on request-side scratch. Large enough for INSERT_VEC at
    /// the configured `dim` plus any text payload.
    send_buf_size: usize = 1 << 20,
};

pub const Client = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    send_buf: []u8,
    recv_buf: []u8,
    next_req_id: u32,
    last_status: protocol.Status,
    last_message: []const u8, // borrows from recv_buf until next call

    pub fn connect(
        allocator: std.mem.Allocator,
        address: std.net.Address,
        opts: ClientOptions,
    ) !Self {
        const stream = try std.net.tcpConnectToAddress(address);
        errdefer stream.close();

        const send_buf = try allocator.alloc(u8, opts.send_buf_size);
        errdefer allocator.free(send_buf);
        const recv_buf = try allocator.alloc(u8, opts.recv_buf_size);

        return .{
            .allocator = allocator,
            .stream = stream,
            .send_buf = send_buf,
            .recv_buf = recv_buf,
            .next_req_id = 1,
            .last_status = .ok,
            .last_message = "",
        };
    }

    pub fn deinit(self: *Self) void {
        self.stream.close();
        self.allocator.free(self.send_buf);
        self.allocator.free(self.recv_buf);
    }

    /// Connect given a "host:port" string.
    pub fn connectByString(
        allocator: std.mem.Allocator,
        host_port: []const u8,
        opts: ClientOptions,
    ) !Self {
        const colon = std.mem.lastIndexOfScalar(u8, host_port, ':') orelse return error.InvalidAddress;
        const host = host_port[0..colon];
        const port = try std.fmt.parseInt(u16, host_port[colon + 1 ..], 10);
        const addr = try std.net.Address.parseIp(host, port);
        return try connect(allocator, addr, opts);
    }

    // ── framing plumbing ────────────────────────────────────────────

    fn nextReqId(self: *Self) u32 {
        const id = self.next_req_id;
        self.next_req_id +%= 1;
        if (self.next_req_id == 0) self.next_req_id = 1;
        return id;
    }

    /// Build a request frame in `send_buf` and ship it. `payload_len` is
    /// the number of payload bytes already written starting at offset
    /// `FRAME_HEADER_SIZE`. Returns the req_id used.
    fn sendFrame(self: *Self, opcode: protocol.Opcode, payload_len: usize) !u32 {
        const req_id = self.nextReqId();
        const body_len: u32 = @intCast(5 + payload_len);
        protocol.encodeHeader(
            self.send_buf[0..protocol.FRAME_HEADER_SIZE],
            body_len,
            @intFromEnum(opcode),
            req_id,
        );
        try self.stream.writeAll(self.send_buf[0 .. protocol.FRAME_HEADER_SIZE + payload_len]);
        return req_id;
    }

    /// Read a full response frame into `recv_buf`. Updates `last_status`
    /// and `last_message`. Returns the payload slice (empty when status
    /// is an error with no diagnostic, or for OK-with-no-body responses).
    fn recvFrame(self: *Self) ![]u8 {
        var got: usize = 0;
        while (got < protocol.FRAME_HEADER_SIZE) {
            const n = try self.stream.read(self.recv_buf[got..protocol.FRAME_HEADER_SIZE]);
            if (n == 0) return error.UnexpectedEof;
            got += n;
        }
        const h = try protocol.decodeHeader(
            self.recv_buf[0..protocol.FRAME_HEADER_SIZE],
            @intCast(self.recv_buf.len),
        );
        const total = protocol.totalFrameSize(h);
        if (total > self.recv_buf.len) return error.ResponseTooLarge;
        while (got < total) {
            const n = try self.stream.read(self.recv_buf[got..total]);
            if (n == 0) return error.UnexpectedEof;
            got += n;
        }

        self.last_status = @enumFromInt(h.tag);
        const payload = self.recv_buf[protocol.FRAME_HEADER_SIZE..total];
        if (self.last_status != .ok) {
            self.last_message = protocol.decodeErrorPayload(payload) catch "";
            return error.ServerError;
        }
        self.last_message = "";
        return payload;
    }

    // ── opcodes ─────────────────────────────────────────────────────

    pub fn ping(self: *Self) !void {
        _ = try self.sendFrame(.ping, 0);
        _ = try self.recvFrame();
    }

    pub fn stats(self: *Self) !Stats {
        _ = try self.sendFrame(.stats, 0);
        const payload = try self.recvFrame();
        return try protocol.decodeStatsResponse(payload);
    }

    pub fn insertVec(self: *Self, vec: []const f32) !u32 {
        const vec_bytes = std.mem.sliceAsBytes(vec);
        const n = protocol.encodeInsertVecRequest(
            self.send_buf[protocol.FRAME_HEADER_SIZE..],
            0,
            vec_bytes,
        );
        _ = try self.sendFrame(.insert_vec, n);
        const payload = try self.recvFrame();
        return try protocol.decodeIdResponse(payload);
    }

    pub fn insertText(self: *Self, text: []const u8) !u32 {
        const n = protocol.encodeInsertTextRequest(
            self.send_buf[protocol.FRAME_HEADER_SIZE..],
            0,
            text,
        );
        _ = try self.sendFrame(.insert_text, n);
        const payload = try self.recvFrame();
        return try protocol.decodeIdResponse(payload);
    }

    pub fn delete(self: *Self, id: u32) !void {
        const n = protocol.encodeIdRequest(self.send_buf[protocol.FRAME_HEADER_SIZE..], id);
        _ = try self.sendFrame(.delete, n);
        _ = try self.recvFrame();
    }

    pub fn replaceVec(self: *Self, id: u32, vec: []const f32) !void {
        const vec_bytes = std.mem.sliceAsBytes(vec);
        const n = protocol.encodeReplaceVecRequest(
            self.send_buf[protocol.FRAME_HEADER_SIZE..],
            id,
            0,
            vec_bytes,
        );
        _ = try self.sendFrame(.replace_vec, n);
        _ = try self.recvFrame();
    }

    pub fn replaceText(self: *Self, id: u32, text: []const u8) !void {
        const n = protocol.encodeReplaceTextRequest(
            self.send_buf[protocol.FRAME_HEADER_SIZE..],
            id,
            0,
            text,
        );
        _ = try self.sendFrame(.replace_text, n);
        _ = try self.recvFrame();
    }

    /// GET returns a heap-allocated result the caller must free via
    /// `allocator.free(result.name); allocator.free(result.vec);`.
    /// Dim is needed because the payload's vec length is implicit.
    pub fn get(self: *Self, id: u32, dim: usize) !GetResult {
        const n = protocol.encodeIdRequest(self.send_buf[protocol.FRAME_HEADER_SIZE..], id);
        _ = try self.sendFrame(.get, n);
        const payload = try self.recvFrame();
        const view = try protocol.decodeGetResponse(payload, dim);
        const name = try self.allocator.dupe(u8, view.name);
        errdefer self.allocator.free(name);
        const vec = try self.allocator.alloc(f32, dim);
        @memcpy(std.mem.sliceAsBytes(vec), view.vec_bytes);
        return .{ .name = name, .vec = vec };
    }

    /// Returns a slice the caller owns. Each `ClientSearchResult.name` is
    /// also owned and must be freed individually via `freeSearchResults`.
    pub fn searchVec(self: *Self, vec: []const f32, top_k: u16, ef: u16) ![]ClientSearchResult {
        const vec_bytes = std.mem.sliceAsBytes(vec);
        const n = protocol.encodeSearchVecRequest(
            self.send_buf[protocol.FRAME_HEADER_SIZE..],
            top_k,
            ef,
            0,
            vec_bytes,
        );
        _ = try self.sendFrame(.search_vec, n);
        const payload = try self.recvFrame();
        return try self.collectSearchResults(payload);
    }

    pub fn searchText(self: *Self, text: []const u8, top_k: u16, ef: u16) ![]ClientSearchResult {
        const n = protocol.encodeSearchTextRequest(
            self.send_buf[protocol.FRAME_HEADER_SIZE..],
            top_k,
            ef,
            0,
            text,
        );
        _ = try self.sendFrame(.search_text, n);
        const payload = try self.recvFrame();
        return try self.collectSearchResults(payload);
    }

    fn collectSearchResults(self: *Self, payload: []const u8) ![]ClientSearchResult {
        var iter = try protocol.searchResultIter(payload);
        var out: std.ArrayListUnmanaged(ClientSearchResult) = .{};
        errdefer {
            for (out.items) |r| self.allocator.free(r.name);
            out.deinit(self.allocator);
        }
        while (iter.next()) |r| {
            const name_copy = try self.allocator.dupe(u8, r.name);
            try out.append(self.allocator, .{
                .id = r.id,
                .dist = r.dist,
                .name = name_copy,
            });
        }
        if (iter.err) |e| return e;
        return out.toOwnedSlice(self.allocator);
    }

    pub fn freeSearchResults(self: *Self, results: []ClientSearchResult) void {
        for (results) |r| self.allocator.free(r.name);
        self.allocator.free(results);
    }

    /// Returns snapshot duration in nanoseconds.
    pub fn snapshot(self: *Self) !u64 {
        _ = try self.sendFrame(.snapshot, 0);
        const payload = try self.recvFrame();
        return try protocol.decodeSnapshotResponse(payload);
    }

    /// Send CLOSE; caller should follow up with `deinit`.
    pub fn sendClose(self: *Self) !void {
        _ = try self.sendFrame(.close, 0);
        _ = try self.recvFrame();
    }
};

// ── tests (round-trip against the real server) ────────────────────────

const testing = std.testing;
const server = @import("server.zig");
const Store = @import("store.zig").Store;
const HnswIndex = @import("hnsw.zig").HnswIndex;
const MutableMetadata = @import("metadata_mut.zig").MutableMetadata;
const ollama = @import("ollama.zig");
const config_mod = @import("config.zig");

const TestM: usize = 16;

const ClientTestHarness = struct {
    gpa_allocator: std.mem.Allocator,
    cfg: config_mod.Config,
    store: Store,
    index: HnswIndex(TestM),
    md: MutableMetadata,
    ws: HnswIndex(TestM).Workspace,
    embedder: ollama.FakeEmbedder,
    srv: server.Server(TestM),
    shutdown: std.atomic.Value(bool),
    thread: ?std.Thread,

    fn setUp(
        self: *ClientTestHarness,
        allocator: std.mem.Allocator,
        dim: usize,
        max_vectors: usize,
    ) !void {
        self.* = undefined;
        self.gpa_allocator = allocator;
        self.shutdown = .init(false);
        self.thread = null;
        self.cfg = .{
            .embedder = .{
                .provider = "ollama",
                .model = "fake",
                .base_url = "http://localhost:11434",
                .dim = dim,
                .max_text_bytes = 1024,
            },
            .index = .{
                .ef_construction = 40,
                .ef_search = 40,
                .max_ef = 40,
            },
            .storage = .{
                .data_dir = "./test-serve-data",
                .max_vectors = max_vectors,
            },
        };
        self.store = try Store.init(allocator, dim, max_vectors);
        self.index = try HnswIndex(TestM).init(allocator, &self.store, .{
            .max_vectors = max_vectors,
            .max_upper_slots = max_vectors,
            .ef_construction = self.cfg.index.ef_construction,
            .seed = self.cfg.index.seed,
        });
        self.md = MutableMetadata.init();
        self.ws = try HnswIndex(TestM).Workspace.init(allocator, max_vectors, self.cfg.index.max_ef);
        self.embedder = ollama.FakeEmbedder.init(dim, 0xc1_1e_17);
        const emb = self.embedder.embedder();
        self.srv = try server.Server(TestM).init(
            allocator,
            &self.store,
            &self.index,
            &self.md,
            &self.ws,
            emb,
            &self.cfg,
            .{
                .listen_port = 0,
                .idle_timeout_secs = 0,
                .auto_snapshot_secs = 0,
                .reuse_address = true,
                .skip_final_snapshot = true,
                .max_connections = 4,
            },
        );
    }

    fn start(self: *ClientTestHarness) !void {
        self.thread = try std.Thread.spawn(.{}, runner, .{self});
    }

    fn runner(self: *ClientTestHarness) !void {
        try self.srv.run(&self.shutdown);
    }

    fn tearDown(self: *ClientTestHarness) void {
        self.shutdown.store(true, .monotonic);
        if (self.thread) |t| t.join();
        self.thread = null;
        self.srv.deinit();
        self.ws.deinit(self.gpa_allocator);
        self.md.deinit(self.gpa_allocator);
        self.index.deinit();
        self.store.deinit(self.gpa_allocator);
    }
};

test "Client: ping / insertVec / searchVec" {
    var h: ClientTestHarness = undefined;
    const dim: usize = 8;
    try h.setUp(testing.allocator, dim, 16);
    defer h.tearDown();
    try h.start();

    const addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port);
    var client = try Client.connect(testing.allocator, addr, .{});
    defer client.deinit();

    try client.ping();

    var v: [dim]f32 = .{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const id = try client.insertVec(v[0..]);
    try testing.expectEqual(@as(u32, 0), id);

    const results = try client.searchVec(v[0..], 1, 10);
    defer client.freeSearchResults(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(u32, 0), results[0].id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].dist, 1e-5);
}

test "Client: DIM_MISMATCH surfaces as ServerError with message" {
    var h: ClientTestHarness = undefined;
    try h.setUp(testing.allocator, 8, 16);
    defer h.tearDown();
    try h.start();

    const addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port);
    var client = try Client.connect(testing.allocator, addr, .{});
    defer client.deinit();

    var wrong: [4]f32 = .{ 1, 2, 3, 4 };
    const result = client.insertVec(wrong[0..]);
    try testing.expectError(error.ServerError, result);
    try testing.expectEqual(protocol.Status.dim_mismatch, client.last_status);
    try testing.expect(client.last_message.len > 0);
}

test "Client: get returns heap-allocated name and vec" {
    var h: ClientTestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 8);
    defer h.tearDown();
    try h.start();

    const addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port);
    var client = try Client.connect(testing.allocator, addr, .{});
    defer client.deinit();

    var v: [dim]f32 = .{ 1, 2, 3, 4 };
    _ = try client.insertVec(v[0..]);
    const got = try client.get(0, dim);
    defer testing.allocator.free(got.name);
    defer testing.allocator.free(got.vec);

    try testing.expectEqualSlices(f32, v[0..], got.vec);
}

test "Client: sequential ops on one connection" {
    var h: ClientTestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 16);
    defer h.tearDown();
    try h.start();

    const addr = try std.net.Address.parseIp("127.0.0.1", h.srv.bound_port);
    var client = try Client.connect(testing.allocator, addr, .{});
    defer client.deinit();

    const s0 = try client.stats();
    try testing.expectEqual(@as(u64, 0), s0.live_count);

    var v: [dim]f32 = .{ 0, 1, 0, 0 };
    _ = try client.insertVec(v[0..]);
    _ = try client.insertVec(v[0..]);

    const s1 = try client.stats();
    try testing.expectEqual(@as(u64, 2), s1.live_count);

    try client.delete(0);
    const s2 = try client.stats();
    try testing.expectEqual(@as(u64, 1), s2.live_count);
}
