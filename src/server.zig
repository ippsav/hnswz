//! TCP reactor for the `hnswz serve` subcommand.
//!
//! Single-threaded `std.posix.poll` loop. Each connection runs a state
//! machine (`read_header → read_body → dispatch → write_response`) on a
//! pair of preallocated buffers. No allocation in the hot path.
//!
//! The `HnswIndex` and `Workspace` are not thread-safe, so everything
//! happens on the reactor thread — which also gives us mutex-free
//! dispatch. The trade-off is that blocking work (embedder HTTP calls on
//! the `_TEXT` opcodes) stalls all connections until it returns; clients
//! that need high throughput should pre-compute vectors and use the
//! `_VEC` variants. See README for details.
//!
//! The protocol is defined in `protocol.zig`; this module is purely
//! transport + dispatch.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

const protocol = @import("protocol.zig");
const Store = @import("store.zig").Store;
const HnswIndexFn = @import("hnsw.zig").HnswIndex;
const MutableMetadata = @import("metadata_mut.zig").MutableMetadata;
const ollama = @import("ollama.zig");
const config_mod = @import("config.zig");
const heap = @import("heap.zig");
const metadata = @import("metadata.zig");

const log = std.log.scoped(.server);

/// Cap emitted name bytes in SEARCH responses. Per-result names are
/// expected to be short filenames; we clip at this to bound the
/// worst-case response size used when sizing per-connection write
/// buffers.
pub const MAX_NAME_IN_RESPONSE: usize = 256;

pub const ServeOptions = struct {
    listen_addr: []const u8 = "127.0.0.1",
    listen_port: u16 = 9000,
    max_connections: u16 = 64,
    max_frame_bytes: u32 = protocol.MAX_FRAME_BYTES_DEFAULT,
    idle_timeout_secs: u32 = 60,
    auto_snapshot_secs: u32 = 0, // 0 = disabled
    reuse_address: bool = true,
    /// If true, the reactor will NOT attempt a snapshot on shutdown.
    /// Tests enable this to avoid touching the filesystem implicitly.
    skip_final_snapshot: bool = false,
    /// If provided, subcommand wires a SIGINT/SIGTERM handler to set this
    /// flag. Tests pass their own pointer and skip signal installation.
    shutdown_flag: ?*std.atomic.Value(bool) = null,
};

// ── signal handling ───────────────────────────────────────────────────

/// Pointer the sighandler flips. Written once by `installSignalHandlers`.
/// Handlers run at signal-delivery time with no user context, so we have
/// to route through a module-level variable.
var g_sig_flag_ptr: ?*std.atomic.Value(bool) = null;

fn sigHandler(_: i32) callconv(.c) void {
    if (g_sig_flag_ptr) |p| p.store(true, .monotonic);
}

pub fn installSignalHandlers(flag: *std.atomic.Value(bool)) void {
    g_sig_flag_ptr = flag;

    const mask = posix.sigemptyset();
    const act: posix.Sigaction = .{
        .handler = .{ .handler = sigHandler },
        .mask = mask,
        .flags = 0,
    };
    posix.sigaction(posix.SIG.INT, &act, null);
    posix.sigaction(posix.SIG.TERM, &act, null);

    // Ignore SIGPIPE — we check write errors per-call and close gracefully.
    const ign: posix.Sigaction = .{
        .handler = .{ .handler = posix.SIG.IGN },
        .mask = mask,
        .flags = 0,
    };
    posix.sigaction(posix.SIG.PIPE, &ign, null);
}

// ── handler error taxonomy ────────────────────────────────────────────

pub const HandlerError = error{
    InvalidFrame,
    UnsupportedOpcode,
    OutOfCapacity,
    InvalidId,
    DimMismatch,
    TextTooLong,
    EmbedFailed,
    IoError,
    SnapshotFailed,
    TopKTooLarge,
    Busy,
    Internal,
};

fn handlerErrToStatus(e: HandlerError) protocol.Status {
    return switch (e) {
        error.InvalidFrame => .invalid_frame,
        error.UnsupportedOpcode => .unsupported_opcode,
        error.OutOfCapacity => .out_of_capacity,
        error.InvalidId => .invalid_id,
        error.DimMismatch => .dim_mismatch,
        error.TextTooLong => .text_too_long,
        error.EmbedFailed => .embed_failed,
        error.IoError => .io_error,
        error.SnapshotFailed => .snapshot_failed,
        error.TopKTooLarge => .top_k_too_large,
        error.Busy => .busy,
        error.Internal => .internal,
    };
}

// ── connection state machine ──────────────────────────────────────────

const ConnState = enum { inactive, read_header, read_body, write_response };

const Connection = struct {
    stream: std.net.Stream,
    state: ConnState,
    read_buf: []u8,
    write_buf: []u8,
    read_pos: usize,
    header: protocol.Header,
    write_pos: usize,
    write_end: usize,
    last_activity_ns: i128,
    /// When true, close the connection cleanly once the current response
    /// has been drained. Set on CLOSE opcode.
    close_after_write: bool,
};

fn setNonBlocking(fd: posix.fd_t) !void {
    const current = try posix.fcntl(fd, posix.F.GETFL, 0);
    const nonblock_u32: u32 = @bitCast(posix.O{ .NONBLOCK = true });
    _ = try posix.fcntl(fd, posix.F.SETFL, current | @as(usize, nonblock_u32));
}

// ── Server ────────────────────────────────────────────────────────────

pub fn Server(comptime M: usize) type {
    const Index = HnswIndexFn(M);

    return struct {
        const Self = @This();
        pub const IndexType = Index;

        // deps
        allocator: std.mem.Allocator,
        store: *Store,
        index: *Index,
        metadata: *MutableMetadata,
        embedder: ?ollama.Embedder,
        ws: *Index.Workspace,
        cfg: *const config_mod.Config,
        opts: ServeOptions,

        // networking state
        listener: std.net.Server,
        bound_port: u16,

        // Parallel arrays. poll_fds[0] is the listener; poll_fds[i+1]
        // corresponds to conns[i]. Inactive slots carry `fd = -1` so
        // `poll(2)` ignores them (POSIX: negative fds are skipped).
        poll_fds: []posix.pollfd,
        conns: []Connection,

        // scratch
        vec_scratch: []align(4) f32,
        results_scratch: []heap.Entry,
        text_scratch: []u8,

        per_conn_buf_size: usize,

        // snapshot cadence
        snapshot_interval_ns: i128,
        last_snapshot_ns: i128,

        pub fn init(
            allocator: std.mem.Allocator,
            store: *Store,
            index: *Index,
            md: *MutableMetadata,
            ws: *Index.Workspace,
            embedder: ?ollama.Embedder,
            cfg: *const config_mod.Config,
            opts: ServeOptions,
        ) !Self {
            // Size per-connection read/write buffers for the largest
            // reasonable frame in either direction. SEARCH responses are
            // sized with MAX_NAME_IN_RESPONSE capped per result; names
            // exceeding that get clipped at emit time.
            const header_overhead = protocol.FRAME_HEADER_SIZE + 16;
            const search_resp_max = protocol.FRAME_HEADER_SIZE + 2 +
                cfg.index.max_ef * (10 + MAX_NAME_IN_RESPONSE);
            const text_max = protocol.FRAME_HEADER_SIZE + cfg.embedder.max_text_bytes + header_overhead;
            const vec_max = protocol.FRAME_HEADER_SIZE + cfg.embedder.dim * 4 + header_overhead;
            const per_conn_buf_size = @max(text_max, @max(vec_max, search_resp_max)) + 4096;

            // listener
            const addr = try std.net.Address.parseIp(opts.listen_addr, opts.listen_port);
            var listener = try addr.listen(.{
                .reuse_address = opts.reuse_address,
            });
            errdefer listener.deinit();

            // The listener fd is blocking by default, which is fine: we
            // only call accept() after poll() tells us there's a pending
            // connection.

            const bound = listener.listen_address;
            const bound_port: u16 = bound.getPort();

            // poll fds: [listener, conn0, conn1, ...]
            const poll_fds = try allocator.alloc(posix.pollfd, 1 + @as(usize, opts.max_connections));
            errdefer allocator.free(poll_fds);
            for (poll_fds) |*p| p.* = .{ .fd = -1, .events = 0, .revents = 0 };
            poll_fds[0] = .{
                .fd = listener.stream.handle,
                .events = posix.POLL.IN,
                .revents = 0,
            };

            const conns = try allocator.alloc(Connection, opts.max_connections);
            errdefer allocator.free(conns);
            for (conns) |*c| c.* = .{
                .stream = undefined,
                .state = .inactive,
                .read_buf = &.{},
                .write_buf = &.{},
                .read_pos = 0,
                .header = .{ .body_len = 0, .tag = 0, .req_id = 0 },
                .write_pos = 0,
                .write_end = 0,
                .last_activity_ns = 0,
                .close_after_write = false,
            };

            // Preallocate per-connection buffers up front — fits the
            // project's "preallocate everything" discipline. One flat
            // slab, one pointer per connection.
            const read_slab = try allocator.alloc(u8, opts.max_connections * per_conn_buf_size);
            errdefer allocator.free(read_slab);
            const write_slab = try allocator.alloc(u8, opts.max_connections * per_conn_buf_size);
            errdefer allocator.free(write_slab);
            for (conns, 0..) |*c, i| {
                c.read_buf = read_slab[i * per_conn_buf_size ..][0..per_conn_buf_size];
                c.write_buf = write_slab[i * per_conn_buf_size ..][0..per_conn_buf_size];
            }

            const vec_scratch = try allocator.alignedAlloc(f32, .@"4", cfg.embedder.dim);
            errdefer allocator.free(vec_scratch);
            const results_scratch = try allocator.alloc(heap.Entry, cfg.index.max_ef);
            errdefer allocator.free(results_scratch);
            const text_scratch = try allocator.alloc(u8, cfg.embedder.max_text_bytes);
            errdefer allocator.free(text_scratch);

            return .{
                .allocator = allocator,
                .store = store,
                .index = index,
                .metadata = md,
                .embedder = embedder,
                .ws = ws,
                .cfg = cfg,
                .opts = opts,
                .listener = listener,
                .bound_port = bound_port,
                .poll_fds = poll_fds,
                .conns = conns,
                .vec_scratch = vec_scratch,
                .results_scratch = results_scratch,
                .text_scratch = text_scratch,
                .per_conn_buf_size = per_conn_buf_size,
                .snapshot_interval_ns = @as(i128, opts.auto_snapshot_secs) * std.time.ns_per_s,
                .last_snapshot_ns = std.time.nanoTimestamp(),
            };
        }

        pub fn deinit(self: *Self) void {
            // close any active connections
            for (self.conns) |*c| {
                if (c.state != .inactive) c.stream.close();
            }
            self.listener.deinit();

            // one slab each; take the slab pointer from conn[0]
            if (self.conns.len > 0) {
                self.allocator.free(self.conns[0].read_buf.ptr[0 .. self.per_conn_buf_size * self.conns.len]);
                self.allocator.free(self.conns[0].write_buf.ptr[0 .. self.per_conn_buf_size * self.conns.len]);
            }
            self.allocator.free(self.conns);
            self.allocator.free(self.poll_fds);
            self.allocator.free(self.vec_scratch);
            self.allocator.free(self.results_scratch);
            self.allocator.free(self.text_scratch);
        }

        /// Main event loop. `shutdown_flag` is polled between poll()
        /// iterations; flipping it exits the loop. The flag may be
        /// driven by a signal handler (prod) or directly (tests).
        pub fn run(self: *Self, shutdown_flag: *std.atomic.Value(bool)) !void {
            log.info("listening on {s}:{d}", .{ self.opts.listen_addr, self.bound_port });

            while (!shutdown_flag.load(.monotonic)) {
                const timeout_ms = self.computePollTimeoutMs();
                const n = posix.poll(self.poll_fds, timeout_ms) catch |err| switch (err) {
                    error.SystemResources => {
                        log.warn("poll failed with system resources; retrying", .{});
                        continue;
                    },
                    else => return err,
                };

                if (n == 0) {
                    self.checkIdleTimeouts();
                    self.maybeAutoSnapshot();
                    continue;
                }

                if (self.poll_fds[0].revents & posix.POLL.IN != 0) {
                    self.handleAccept();
                }

                for (self.conns, 0..) |*c, i| {
                    if (c.state == .inactive) continue;
                    const revents = self.poll_fds[i + 1].revents;
                    const err_mask = posix.POLL.HUP | posix.POLL.ERR | posix.POLL.NVAL;
                    if (revents & err_mask != 0) {
                        self.closeConn(i);
                        continue;
                    }
                    if (revents & posix.POLL.IN != 0) self.progressRead(i);
                    if (c.state == .inactive) continue;
                    if (revents & posix.POLL.OUT != 0) self.progressWrite(i);
                }

                self.checkIdleTimeouts();
                self.maybeAutoSnapshot();
            }

            log.info("shutdown requested; flushing snapshot and closing", .{});
            if (!self.opts.skip_final_snapshot) self.finalSnapshot();

            // Drain any writing connections briefly, then hard-close all.
            for (self.conns, 0..) |*c, i| {
                if (c.state != .inactive) self.closeConn(i);
            }
        }

        // ── accept ────────────────────────────────────────────────────

        fn handleAccept(self: *Self) void {
            const c = self.listener.accept() catch |err| {
                log.warn("accept failed: {s}", .{@errorName(err)});
                return;
            };
            // Find an inactive slot.
            var slot: ?usize = null;
            for (self.conns, 0..) |*cc, i| {
                if (cc.state == .inactive) {
                    slot = i;
                    break;
                }
            }
            if (slot == null) {
                log.info("rejecting accept: max_connections reached", .{});
                c.stream.close();
                return;
            }
            const i = slot.?;

            setNonBlocking(c.stream.handle) catch |err| {
                log.warn("failed to set non-blocking: {s}", .{@errorName(err)});
                c.stream.close();
                return;
            };

            self.conns[i] = .{
                .stream = c.stream,
                .state = .read_header,
                .read_buf = self.conns[i].read_buf,
                .write_buf = self.conns[i].write_buf,
                .read_pos = 0,
                .header = .{ .body_len = 0, .tag = 0, .req_id = 0 },
                .write_pos = 0,
                .write_end = 0,
                .last_activity_ns = std.time.nanoTimestamp(),
                .close_after_write = false,
            };
            self.poll_fds[i + 1] = .{
                .fd = c.stream.handle,
                .events = posix.POLL.IN,
                .revents = 0,
            };
        }

        fn closeConn(self: *Self, i: usize) void {
            const c = &self.conns[i];
            if (c.state == .inactive) return;
            c.stream.close();
            c.state = .inactive;
            c.read_pos = 0;
            c.write_pos = 0;
            c.write_end = 0;
            c.close_after_write = false;
            self.poll_fds[i + 1] = .{ .fd = -1, .events = 0, .revents = 0 };
        }

        // ── progress read ────────────────────────────────────────────

        fn progressRead(self: *Self, i: usize) void {
            const c = &self.conns[i];

            const need: usize = switch (c.state) {
                .read_header => protocol.FRAME_HEADER_SIZE,
                .read_body => protocol.totalFrameSize(c.header),
                else => return,
            };
            if (c.read_pos >= need) return;

            const dst = c.read_buf[c.read_pos..need];
            const n = posix.read(c.stream.handle, dst) catch |err| switch (err) {
                error.WouldBlock => return,
                else => {
                    self.closeConn(i);
                    return;
                },
            };
            if (n == 0) {
                // peer half-closed
                self.closeConn(i);
                return;
            }
            c.read_pos += n;
            c.last_activity_ns = std.time.nanoTimestamp();

            // State transition. Loop so that a single `read` that
            // delivered the whole frame (header+body together) progresses
            // all the way to dispatch without waiting for another poll.
            while (true) {
                switch (c.state) {
                    .read_header => {
                        if (c.read_pos < protocol.FRAME_HEADER_SIZE) return;
                        const h = protocol.decodeHeader(
                            c.read_buf[0..protocol.FRAME_HEADER_SIZE],
                            self.opts.max_frame_bytes,
                        ) catch {
                            self.closeConn(i);
                            return;
                        };
                        const total = protocol.totalFrameSize(h);
                        if (total > c.read_buf.len) {
                            self.closeConn(i);
                            return;
                        }
                        c.header = h;
                        c.state = .read_body;
                        if (total == protocol.FRAME_HEADER_SIZE) {
                            // no payload; dispatch straight away
                            self.dispatch(i);
                            return;
                        }
                        // fall through to potentially dispatch if payload
                        // already buffered.
                    },
                    .read_body => {
                        const total = protocol.totalFrameSize(c.header);
                        if (c.read_pos < total) return;
                        self.dispatch(i);
                        return;
                    },
                    else => return,
                }
            }
        }

        // ── dispatch ─────────────────────────────────────────────────

        fn dispatch(self: *Self, i: usize) void {
            const c = &self.conns[i];
            const header = c.header;
            const payload = c.read_buf[protocol.FRAME_HEADER_SIZE..][0..protocol.payloadLen(header)];

            const w = c.write_buf;
            // Reserve room for the response header; payload writes start
            // at offset FRAME_HEADER_SIZE.
            var payload_end: usize = protocol.FRAME_HEADER_SIZE;
            var status: protocol.Status = .ok;

            const opcode: protocol.Opcode = @enumFromInt(header.tag);
            const handler_result: HandlerError!usize = switch (opcode) {
                .ping => self.handlePing(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .stats => self.handleStats(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .insert_vec => self.handleInsertVec(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .insert_text => self.handleInsertText(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .delete => self.handleDelete(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .replace_vec => self.handleReplaceVec(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .replace_text => self.handleReplaceText(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .get => self.handleGet(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .search_vec => self.handleSearchVec(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .search_text => self.handleSearchText(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .snapshot => self.handleSnapshot(payload, w[protocol.FRAME_HEADER_SIZE..]),
                .close => blk: {
                    c.close_after_write = true;
                    break :blk self.handlePing(payload, w[protocol.FRAME_HEADER_SIZE..]);
                },
                _ => HandlerError.UnsupportedOpcode,
            };

            if (handler_result) |n| {
                payload_end += n;
            } else |err| {
                status = handlerErrToStatus(err);
                const msg = protocol.statusMessage(status);
                payload_end = protocol.FRAME_HEADER_SIZE + protocol.encodeErrorPayload(
                    w[protocol.FRAME_HEADER_SIZE..],
                    msg,
                );
            }

            const body_len: u32 = @intCast(payload_end - 4);
            protocol.encodeHeader(
                w[0..protocol.FRAME_HEADER_SIZE],
                body_len,
                @intFromEnum(status),
                header.req_id,
            );

            c.write_pos = 0;
            c.write_end = payload_end;
            c.state = .write_response;
            c.read_pos = 0; // start fresh header next time
            self.poll_fds[i + 1].events = posix.POLL.OUT;

            // Best-effort immediate drain — saves a poll round-trip on
            // the common case where kernel send buffer has space.
            self.progressWrite(i);
        }

        // ── progress write ───────────────────────────────────────────

        fn progressWrite(self: *Self, i: usize) void {
            const c = &self.conns[i];
            if (c.state != .write_response) return;

            while (c.write_pos < c.write_end) {
                const slice = c.write_buf[c.write_pos..c.write_end];
                const n = posix.write(c.stream.handle, slice) catch |err| switch (err) {
                    error.WouldBlock => return, // wait for next POLL.OUT
                    else => {
                        self.closeConn(i);
                        return;
                    },
                };
                if (n == 0) {
                    self.closeConn(i);
                    return;
                }
                c.write_pos += n;
                c.last_activity_ns = std.time.nanoTimestamp();
            }

            // Response fully drained.
            if (c.close_after_write) {
                self.closeConn(i);
                return;
            }
            c.state = .read_header;
            self.poll_fds[i + 1].events = posix.POLL.IN;
        }

        // ── handlers ─────────────────────────────────────────────────

        fn handlePing(_: *Self, payload: []const u8, _: []u8) HandlerError!usize {
            if (payload.len != 0) return error.InvalidFrame;
            return 0;
        }

        fn handleStats(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            if (payload.len != 0) return error.InvalidFrame;
            const has_ep: u8 = if (self.index.entry_point != null) 1 else 0;
            const s: protocol.StatsResponse = .{
                .proto_version = protocol.PROTO_VERSION,
                .flags = 0,
                .dim = @intCast(self.store.dim),
                .m = @intCast(M),
                .live_count = @intCast(self.store.live_count),
                .high_water = @intCast(self.store.count),
                .upper_used = @intCast(self.index.upper_used),
                .max_upper_slots = @intCast(self.index.max_upper_slots),
                .max_level = self.index.max_level,
                .has_entry_point = has_ep,
            };
            return protocol.encodeStatsResponse(w, s);
        }

        fn handleInsertVec(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            const view = protocol.decodeInsertVecRequest(payload, self.store.dim) catch |err| switch (err) {
                error.DimMismatch => return error.DimMismatch,
                error.Truncated => return error.InvalidFrame,
                else => return error.Internal,
            };
            @memcpy(std.mem.sliceAsBytes(self.vec_scratch), view.vec_bytes);
            const id = self.store.add(self.vec_scratch) catch |err| switch (err) {
                error.OutOfCapacity => return error.OutOfCapacity,
            };
            self.index.insert(self.ws, id) catch {
                self.store.delete(id);
                return error.Internal;
            };
            self.metadata.setAt(self.allocator, id, "") catch {
                self.store.delete(id);
                return error.Internal;
            };
            return protocol.encodeIdResponse(w, id);
        }

        fn handleInsertText(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            const emb = self.embedder orelse return error.EmbedFailed;
            const view = protocol.decodeInsertTextRequest(payload) catch return error.InvalidFrame;
            if (view.text.len > self.cfg.embedder.max_text_bytes) return error.TextTooLong;

            emb.embed(view.text, self.vec_scratch) catch {
                return error.EmbedFailed;
            };
            const id = self.store.add(self.vec_scratch) catch |err| switch (err) {
                error.OutOfCapacity => return error.OutOfCapacity,
            };
            self.index.insert(self.ws, id) catch {
                self.store.delete(id);
                return error.Internal;
            };

            // Use the text itself (clipped) as the metadata name so that
            // SEARCH_TEXT results are actually identifiable.
            const name_len = @min(view.text.len, MAX_NAME_IN_RESPONSE);
            self.metadata.setAt(self.allocator, id, view.text[0..name_len]) catch {
                self.store.delete(id);
                return error.Internal;
            };
            return protocol.encodeIdResponse(w, id);
        }

        fn handleDelete(self: *Self, payload: []const u8, _: []u8) HandlerError!usize {
            const id = protocol.decodeIdRequest(payload) catch return error.InvalidFrame;
            if (@as(usize, id) >= self.store.count) return error.InvalidId;
            if (self.store.isDeleted(id)) return error.InvalidId;
            self.index.delete(self.ws, id) catch return error.Internal;
            self.metadata.deleteAt(self.allocator, id);
            return 0;
        }

        fn handleReplaceVec(self: *Self, payload: []const u8, _: []u8) HandlerError!usize {
            const view = protocol.decodeReplaceVecRequest(payload, self.store.dim) catch |err| switch (err) {
                error.DimMismatch => return error.DimMismatch,
                error.Truncated => return error.InvalidFrame,
                else => return error.Internal,
            };
            if (@as(usize, view.id) >= self.store.count) return error.InvalidId;
            if (self.store.isDeleted(view.id)) return error.InvalidId;
            @memcpy(std.mem.sliceAsBytes(self.vec_scratch), view.vec_bytes);
            self.index.replaceVector(self.ws, view.id, self.vec_scratch) catch return error.Internal;
            return 0;
        }

        fn handleReplaceText(self: *Self, payload: []const u8, _: []u8) HandlerError!usize {
            const emb = self.embedder orelse return error.EmbedFailed;
            const view = protocol.decodeReplaceTextRequest(payload) catch return error.InvalidFrame;
            if (view.text.len > self.cfg.embedder.max_text_bytes) return error.TextTooLong;
            if (@as(usize, view.id) >= self.store.count) return error.InvalidId;
            if (self.store.isDeleted(view.id)) return error.InvalidId;
            emb.embed(view.text, self.vec_scratch) catch return error.EmbedFailed;
            self.index.replaceVector(self.ws, view.id, self.vec_scratch) catch return error.Internal;

            const name_len = @min(view.text.len, MAX_NAME_IN_RESPONSE);
            self.metadata.setAt(self.allocator, view.id, view.text[0..name_len]) catch return error.Internal;
            return 0;
        }

        fn handleGet(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            const id = protocol.decodeIdRequest(payload) catch return error.InvalidFrame;
            if (@as(usize, id) >= self.store.count) return error.InvalidId;
            if (self.store.isDeleted(id)) return error.InvalidId;
            const vec = self.store.get(id);
            const vec_bytes = std.mem.sliceAsBytes(vec);
            const name_full = self.metadata.get(id) orelse "";
            const name = name_full[0..@min(name_full.len, MAX_NAME_IN_RESPONSE)];
            if (w.len < 2 + name.len + vec_bytes.len) return error.Internal;
            return protocol.encodeGetResponse(w, name, vec_bytes);
        }

        fn handleSearchVec(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            const view = protocol.decodeSearchVecRequest(payload, self.store.dim) catch |err| switch (err) {
                error.DimMismatch => return error.DimMismatch,
                error.Truncated => return error.InvalidFrame,
                else => return error.Internal,
            };
            if (view.top_k == 0) return error.InvalidFrame;
            if (view.top_k > view.ef) return error.TopKTooLarge;
            if (view.ef > self.cfg.index.max_ef) return error.TopKTooLarge;

            @memcpy(std.mem.sliceAsBytes(self.vec_scratch), view.vec_bytes);
            const out_slots = self.results_scratch[0..view.top_k];
            const results = self.index.search(self.ws, self.vec_scratch, view.top_k, view.ef, out_slots) catch {
                return error.Internal;
            };
            return self.encodeSearchResponse(w, results);
        }

        fn handleSearchText(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            const emb = self.embedder orelse return error.EmbedFailed;
            const view = protocol.decodeSearchTextRequest(payload) catch return error.InvalidFrame;
            if (view.text.len > self.cfg.embedder.max_text_bytes) return error.TextTooLong;
            if (view.top_k == 0) return error.InvalidFrame;
            if (view.top_k > view.ef) return error.TopKTooLarge;
            if (view.ef > self.cfg.index.max_ef) return error.TopKTooLarge;

            emb.embed(view.text, self.vec_scratch) catch return error.EmbedFailed;
            const out_slots = self.results_scratch[0..view.top_k];
            const results = self.index.search(self.ws, self.vec_scratch, view.top_k, view.ef, out_slots) catch {
                return error.Internal;
            };
            return self.encodeSearchResponse(w, results);
        }

        fn encodeSearchResponse(self: *Self, w: []u8, results: []const heap.Entry) HandlerError!usize {
            var cursor: usize = 0;
            if (w.len < 2) return error.Internal;
            cursor += protocol.writeSearchResultCount(w, @intCast(results.len));
            for (results) |r| {
                const name_full = self.metadata.get(r.id) orelse "";
                const name = name_full[0..@min(name_full.len, MAX_NAME_IN_RESPONSE)];
                const need = 10 + name.len;
                if (cursor + need > w.len) return error.Internal;
                cursor += protocol.writeSearchResult(w[cursor..], r.id, r.dist, name);
            }
            return cursor;
        }

        fn handleSnapshot(self: *Self, payload: []const u8, w: []u8) HandlerError!usize {
            if (payload.len != 0) return error.InvalidFrame;
            const started = std.time.nanoTimestamp();
            self.snapshotNow() catch return error.SnapshotFailed;
            const elapsed: u64 = @intCast(std.time.nanoTimestamp() - started);
            return protocol.encodeSnapshotResponse(w, elapsed);
        }

        // ── snapshot ─────────────────────────────────────────────────

        fn snapshotNow(self: *Self) !void {
            std.fs.cwd().makePath(self.cfg.storage.data_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
            var dir = try std.fs.cwd().openDir(self.cfg.storage.data_dir, .{});
            defer dir.close();

            const remap = try self.store.buildRemap(self.allocator);
            defer self.allocator.free(remap);

            try self.store.save(dir, self.cfg.storage.vectors_file);
            try self.index.save(dir, self.cfg.storage.graph_file, remap);
            try self.metadata.save(self.allocator, dir, self.cfg.storage.metadata_file, remap);
            self.last_snapshot_ns = std.time.nanoTimestamp();
        }

        fn finalSnapshot(self: *Self) void {
            self.snapshotNow() catch |err| {
                log.err("final snapshot failed: {s}", .{@errorName(err)});
            };
        }

        // ── periodic maintenance ─────────────────────────────────────

        fn computePollTimeoutMs(_: *const Self) i32 {
            // 500 ms heartbeat. Bounds shutdown-flag latency and drives
            // idle-timeout / auto-snapshot checks without a self-pipe.
            // Cost is one idle poll wake every 500 ms — negligible.
            return 500;
        }

        fn checkIdleTimeouts(self: *Self) void {
            if (self.opts.idle_timeout_secs == 0) return;
            const cutoff_ns: i128 = std.time.nanoTimestamp() -
                @as(i128, self.opts.idle_timeout_secs) * std.time.ns_per_s;
            for (self.conns, 0..) |*c, i| {
                if (c.state == .inactive) continue;
                if (c.last_activity_ns < cutoff_ns) {
                    log.info("closing idle connection {d}", .{i});
                    self.closeConn(i);
                }
            }
        }

        fn maybeAutoSnapshot(self: *Self) void {
            if (self.snapshot_interval_ns == 0) return;
            const now = std.time.nanoTimestamp();
            if (now - self.last_snapshot_ns < self.snapshot_interval_ns) return;
            const started = now;
            self.snapshotNow() catch |err| {
                log.err("auto-snapshot failed: {s}", .{@errorName(err)});
                return;
            };
            const elapsed = std.time.nanoTimestamp() - started;
            log.info("auto-snapshot: {d} ns", .{elapsed});
        }
    };
}

// ── integration tests ────────────────────────────────────────────────
//
// Tests spin up the server on 127.0.0.1:0 with a FakeEmbedder and use a
// blocking client socket for round-trips. The server runs on a worker
// thread; tests flip a per-test shutdown flag to exit cleanly. Signal
// handlers are NOT installed in tests — they'd pollute the binary.

const testing = std.testing;

const TestM: usize = 16;

const TestHarness = struct {
    gpa_allocator: std.mem.Allocator,
    cfg: config_mod.Config,
    store: Store,
    index: HnswIndexFn(TestM),
    md: MutableMetadata,
    ws: HnswIndexFn(TestM).Workspace,
    embedder: ollama.FakeEmbedder,
    srv: Server(TestM),
    shutdown: std.atomic.Value(bool),
    thread: ?std.Thread,

    fn setUp(
        self: *TestHarness,
        allocator: std.mem.Allocator,
        dim: usize,
        max_vectors: usize,
        max_connections: u16,
    ) !void {
        self.* = undefined;
        self.gpa_allocator = allocator;
        self.shutdown = .init(false);
        self.thread = null;
        self.cfg = .{
            .embedder = .{
                .provider = "ollama",
                .base_url = "http://localhost:11434",
                .model = "fake",
                .dim = dim,
                .max_text_bytes = 4096,
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
        self.index = try HnswIndexFn(TestM).init(allocator, &self.store, .{
            .max_vectors = max_vectors,
            .max_upper_slots = max_vectors,
            .ef_construction = self.cfg.index.ef_construction,
            .seed = self.cfg.index.seed,
        });
        self.md = MutableMetadata.init();
        self.ws = try HnswIndexFn(TestM).Workspace.init(allocator, max_vectors, self.cfg.index.max_ef);
        self.embedder = ollama.FakeEmbedder.init(dim, 0x5eed);
        const emb_opt = self.embedder.embedder();
        self.srv = try Server(TestM).init(
            allocator,
            &self.store,
            &self.index,
            &self.md,
            &self.ws,
            emb_opt,
            &self.cfg,
            .{
                .listen_addr = "127.0.0.1",
                .listen_port = 0, // ephemeral
                .max_connections = max_connections,
                .max_frame_bytes = 1 << 20,
                .idle_timeout_secs = 0,
                .auto_snapshot_secs = 0,
                .reuse_address = true,
                .skip_final_snapshot = true,
            },
        );
    }

    fn start(self: *TestHarness) !void {
        self.thread = try std.Thread.spawn(.{}, runServer, .{self});
    }

    /// Tears everything down in the correct order: signal reactor to
    /// exit, join the worker thread, then free server + deps. Tests
    /// MUST `defer h.tearDown()` AFTER `setUp` to keep this one-shot.
    fn tearDown(self: *TestHarness) void {
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

fn runServer(h: *TestHarness) !void {
    try h.srv.run(&h.shutdown);
}

fn connect(port: u16) !std.net.Stream {
    const addr = try std.net.Address.parseIp("127.0.0.1", port);
    return try std.net.tcpConnectToAddress(addr);
}

/// Write a full request frame (9-byte header + payload) and block until
/// a full response frame has been read. Uses a fixed 1 MB buffer; tests
/// never exchange larger frames than that.
fn roundTrip(
    stream: std.net.Stream,
    opcode: protocol.Opcode,
    req_id: u32,
    payload: []const u8,
    resp_buf: []u8,
) !struct { header: protocol.Header, payload: []u8 } {
    var hdr_buf: [protocol.FRAME_HEADER_SIZE]u8 = undefined;
    const body_len: u32 = @intCast(5 + payload.len);
    protocol.encodeHeader(&hdr_buf, body_len, @intFromEnum(opcode), req_id);
    try stream.writeAll(&hdr_buf);
    if (payload.len > 0) try stream.writeAll(payload);

    var got: usize = 0;
    while (got < protocol.FRAME_HEADER_SIZE) {
        const n = try stream.read(resp_buf[got..protocol.FRAME_HEADER_SIZE]);
        if (n == 0) return error.UnexpectedEof;
        got += n;
    }
    const h = try protocol.decodeHeader(resp_buf[0..protocol.FRAME_HEADER_SIZE], 1 << 20);
    const total = protocol.totalFrameSize(h);
    if (total > resp_buf.len) return error.ResponseBufferTooSmall;
    while (got < total) {
        const n = try stream.read(resp_buf[got..total]);
        if (n == 0) return error.UnexpectedEof;
        got += n;
    }
    const payload_start = protocol.FRAME_HEADER_SIZE;
    return .{ .header = h, .payload = resp_buf[payload_start..total] };
}

test "PING round-trip" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 8, 16, 4);
    defer h.tearDown();

    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    var buf: [64]u8 = undefined;
    const r = try roundTrip(stream, .ping, 0xABCD, "", &buf);
    try testing.expectEqual(@as(u8, 0), r.header.tag); // status = OK
    try testing.expectEqual(@as(u32, 0xABCD), r.header.req_id);
    try testing.expectEqual(@as(usize, 0), r.payload.len);
}

test "STATS returns configured dim and M" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 8, 16, 4);
    defer h.tearDown();

    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    var buf: [128]u8 = undefined;
    const r = try roundTrip(stream, .stats, 1, "", &buf);
    try testing.expectEqual(@as(u8, 0), r.header.tag);
    const s = try protocol.decodeStatsResponse(r.payload);
    try testing.expectEqual(@as(u32, 8), s.dim);
    try testing.expectEqual(@as(u16, TestM), s.m);
    try testing.expectEqual(@as(u64, 0), s.live_count);
    try testing.expectEqual(@as(u8, 0), s.has_entry_point);
}

test "INSERT_VEC then SEARCH_VEC finds the inserted vector" {
    var h: TestHarness = undefined;
    const dim: usize = 8;
    try h.setUp(testing.allocator, dim, 16, 4);
    defer h.tearDown();

    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    var vec: [dim]f32 = .{ 1.0, 0, 0, 0, 0, 0, 0, 0 };
    const vec_bytes = std.mem.sliceAsBytes(vec[0..]);

    var req_buf: [2 + dim * 4]u8 = undefined;
    const req_n = protocol.encodeInsertVecRequest(&req_buf, 0, vec_bytes);
    var resp_buf: [128]u8 = undefined;
    const r = try roundTrip(stream, .insert_vec, 1, req_buf[0..req_n], &resp_buf);
    try testing.expectEqual(@as(u8, 0), r.header.tag);
    const id = try protocol.decodeIdResponse(r.payload);
    try testing.expectEqual(@as(u32, 0), id);

    // search for the same vector
    var sreq: [6 + dim * 4]u8 = undefined;
    const sn = protocol.encodeSearchVecRequest(&sreq, 1, 40, 0, vec_bytes);
    var sresp: [512]u8 = undefined;
    const sr = try roundTrip(stream, .search_vec, 2, sreq[0..sn], &sresp);
    try testing.expectEqual(@as(u8, 0), sr.header.tag);
    var it = try protocol.searchResultIter(sr.payload);
    const first = it.next().?;
    try testing.expectEqual(@as(u32, 0), first.id);
    try testing.expectApproxEqAbs(@as(f32, 0.0), first.dist, 1e-5);
}

test "INSERT_VEC with wrong-size body returns DIM_MISMATCH" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 8, 16, 4);
    defer h.tearDown();

    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    // dim=8 expected; send 4 floats worth
    var bad_vec: [4]f32 = .{ 1, 2, 3, 4 };
    const bad_bytes = std.mem.sliceAsBytes(bad_vec[0..]);
    var req_buf: [2 + 16]u8 = undefined;
    const req_n = protocol.encodeInsertVecRequest(&req_buf, 0, bad_bytes);
    var resp_buf: [256]u8 = undefined;
    const r = try roundTrip(stream, .insert_vec, 1, req_buf[0..req_n], &resp_buf);
    try testing.expectEqual(@intFromEnum(protocol.Status.dim_mismatch), r.header.tag);
}

test "DELETE on never-inserted id returns INVALID_ID" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 4, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();
    const stream = try connect(port);
    defer stream.close();

    var req: [4]u8 = undefined;
    _ = protocol.encodeIdRequest(&req, 999);
    var resp: [256]u8 = undefined;
    const r = try roundTrip(stream, .delete, 1, &req, &resp);
    try testing.expectEqual(@intFromEnum(protocol.Status.invalid_id), r.header.tag);
}

test "INSERT_VEC past capacity returns OUT_OF_CAPACITY" {
    var h: TestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 2, 4); // cap = 2
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();
    const stream = try connect(port);
    defer stream.close();

    var vec: [dim]f32 = .{ 1, 0, 0, 0 };
    const vb = std.mem.sliceAsBytes(vec[0..]);
    var req: [2 + dim * 4]u8 = undefined;
    const rn = protocol.encodeInsertVecRequest(&req, 0, vb);
    var resp: [128]u8 = undefined;

    _ = try roundTrip(stream, .insert_vec, 1, req[0..rn], &resp);
    _ = try roundTrip(stream, .insert_vec, 2, req[0..rn], &resp);
    const r3 = try roundTrip(stream, .insert_vec, 3, req[0..rn], &resp);
    try testing.expectEqual(@intFromEnum(protocol.Status.out_of_capacity), r3.header.tag);
}

test "SEARCH_VEC with top_k > ef returns TOP_K_TOO_LARGE" {
    var h: TestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();
    const stream = try connect(port);
    defer stream.close();

    var vec: [dim]f32 = .{ 1, 0, 0, 0 };
    const vb = std.mem.sliceAsBytes(vec[0..]);
    var req: [6 + dim * 4]u8 = undefined;
    const rn = protocol.encodeSearchVecRequest(&req, 10, 5, 0, vb);
    var resp: [128]u8 = undefined;
    const r = try roundTrip(stream, .search_vec, 1, req[0..rn], &resp);
    try testing.expectEqual(@intFromEnum(protocol.Status.top_k_too_large), r.header.tag);
}

test "partial-read: byte-by-byte frame still dispatches" {
    var h: TestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    // Build a PING request and write it 1 byte at a time.
    var hdr: [protocol.FRAME_HEADER_SIZE]u8 = undefined;
    protocol.encodeHeader(&hdr, 5, @intFromEnum(protocol.Opcode.ping), 42);
    for (hdr) |b| try stream.writeAll(&[_]u8{b});

    // Read the response header, also 1 byte at a time via blocking read.
    var got: [protocol.FRAME_HEADER_SIZE]u8 = undefined;
    var fill: usize = 0;
    while (fill < protocol.FRAME_HEADER_SIZE) {
        const n = try stream.read(got[fill..]);
        if (n == 0) return error.UnexpectedEof;
        fill += n;
    }
    const r = try protocol.decodeHeader(&got, 1 << 20);
    try testing.expectEqual(@as(u32, 42), r.req_id);
    try testing.expectEqual(@as(u8, 0), r.tag);
}

test "REPLACE_VEC updates stored vector" {
    var h: TestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();
    const stream = try connect(port);
    defer stream.close();

    var v0: [dim]f32 = .{ 1, 0, 0, 0 };
    var v1: [dim]f32 = .{ 0, 1, 0, 0 };
    const v0b = std.mem.sliceAsBytes(v0[0..]);
    const v1b = std.mem.sliceAsBytes(v1[0..]);

    var ireq: [2 + dim * 4]u8 = undefined;
    const in = protocol.encodeInsertVecRequest(&ireq, 0, v0b);
    var resp: [256]u8 = undefined;
    const ir = try roundTrip(stream, .insert_vec, 1, ireq[0..in], &resp);
    try testing.expectEqual(@as(u8, 0), ir.header.tag);

    var rreq: [6 + dim * 4]u8 = undefined;
    const rn = protocol.encodeReplaceVecRequest(&rreq, 0, 0, v1b);
    const rr = try roundTrip(stream, .replace_vec, 2, rreq[0..rn], &resp);
    try testing.expectEqual(@as(u8, 0), rr.header.tag);

    // GET id 0 and confirm the vec is v1
    var greq: [4]u8 = undefined;
    _ = protocol.encodeIdRequest(&greq, 0);
    const gr = try roundTrip(stream, .get, 3, &greq, &resp);
    try testing.expectEqual(@as(u8, 0), gr.header.tag);
    const view = try protocol.decodeGetResponse(gr.payload, dim);
    try testing.expectEqualSlices(u8, v1b, view.vec_bytes);
}

test "CLOSE opcode closes the connection after ack" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 4, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();
    var resp: [64]u8 = undefined;
    _ = try roundTrip(stream, .close, 1, "", &resp);
    // Server should have closed; subsequent read returns 0.
    var trailing: [16]u8 = undefined;
    const n = try stream.read(&trailing);
    try testing.expectEqual(@as(usize, 0), n);
}

test "unknown opcode returns UNSUPPORTED_OPCODE" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 4, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();
    const op: protocol.Opcode = @enumFromInt(0x77);
    var resp: [128]u8 = undefined;
    const r = try roundTrip(stream, op, 1, "", &resp);
    try testing.expectEqual(@intFromEnum(protocol.Status.unsupported_opcode), r.header.tag);
}

test "SNAPSHOT writes files that round-trip with existing load" {
    var h: TestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 8, 4);
    defer h.tearDown();

    // Use a unique per-run directory so tests don't collide.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const real = try tmp.dir.realpathAlloc(testing.allocator, ".");
    defer testing.allocator.free(real);
    h.cfg.storage.data_dir = real;

    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    var vec: [dim]f32 = .{ 1, 0, 0, 0 };
    const vb = std.mem.sliceAsBytes(vec[0..]);
    var ireq: [2 + dim * 4]u8 = undefined;
    const in = protocol.encodeInsertVecRequest(&ireq, 0, vb);
    var resp: [128]u8 = undefined;
    _ = try roundTrip(stream, .insert_vec, 1, ireq[0..in], &resp);

    const sr = try roundTrip(stream, .snapshot, 2, "", &resp);
    try testing.expectEqual(@as(u8, 0), sr.header.tag);
    const elapsed = try protocol.decodeSnapshotResponse(sr.payload);
    try testing.expect(elapsed > 0);

    // Verify we can load via the existing non-server APIs.
    var loaded = try Store.load(testing.allocator, tmp.dir, h.cfg.storage.vectors_file, null);
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), loaded.live_count);
    try testing.expectEqualSlices(u8, vb, std.mem.sliceAsBytes(loaded.get(0)));
}

test "multiple sequential requests on one connection" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 4, 16, 4);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    var resp: [128]u8 = undefined;
    const r1 = try roundTrip(stream, .ping, 1, "", &resp);
    try testing.expectEqual(@as(u32, 1), r1.header.req_id);
    const r2 = try roundTrip(stream, .ping, 2, "", &resp);
    try testing.expectEqual(@as(u32, 2), r2.header.req_id);
    const r3 = try roundTrip(stream, .ping, 3, "", &resp);
    try testing.expectEqual(@as(u32, 3), r3.header.req_id);
}

test "max_connections: extras get closed immediately" {
    var h: TestHarness = undefined;
    try h.setUp(testing.allocator, 4, 8, 1); // only 1 slot
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();

    const s1 = try connect(port);
    defer s1.close();
    // hold s1 open by sending a ping
    var resp: [64]u8 = undefined;
    _ = try roundTrip(s1, .ping, 1, "", &resp);

    // s2 should be accepted and dropped
    const s2 = try connect(port);
    defer s2.close();
    var buf: [16]u8 = undefined;
    const n = s2.read(&buf) catch 0;
    try testing.expectEqual(@as(usize, 0), n);
}
