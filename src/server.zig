//! TCP server for the `hnswz serve` subcommand.
//!
//!   * Main thread: kqueue-driven event loop (`src/io/darwin.zig`).
//!     Accepts new connections, runs the per-connection state machine
//!     (`read_header → read_body → dispatch → write_response`), and
//!     dispatches to the worker pool.
//!
//!   * Worker pool (`src/dispatcher.zig`): N threads, each with its own
//!     `HnswIndex.Workspace`, vector scratch, and results scratch. A
//!     `std.Thread.RwLock` protects the shared store/index/metadata —
//!     searches take it shared, writes take it exclusive.
//!
//!   * Cross-thread wakeup: workers post results to a done queue and
//!     write one byte to a pipe. The main thread registers a
//!     `readReady` completion on the pipe's read end, drains the done
//!     queue, and submits `writeReady` for each response.
//!
//! This departs from TigerBeetle's pure single-threaded model: our
//! HNSW ops (~19 ms insert, ~4 ms search at dim=4096) are not bounded
//! enough to run inline on the loop.
//!
//! The protocol lives in `protocol.zig`; this module is pure transport +
//! dispatch.

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
const io_mod = @import("io.zig");
const IO = io_mod.IO;
const dispatcher_mod = @import("dispatcher.zig");

const log = std.log.scoped(.server);

/// Cap emitted name bytes in SEARCH responses. Duplicated across
/// server.zig and dispatcher.zig to keep the dispatcher self-contained.
pub const MAX_NAME_IN_RESPONSE: usize = 256;

pub const ServeOptions = struct {
    listen_addr: []const u8 = "127.0.0.1",
    listen_port: u16 = 9000,
    max_connections: u16 = 64,
    max_frame_bytes: u32 = protocol.MAX_FRAME_BYTES_DEFAULT,
    idle_timeout_secs: u32 = 60,
    auto_snapshot_secs: u32 = 0,
    reuse_address: bool = true,
    skip_final_snapshot: bool = false,
    shutdown_flag: ?*std.atomic.Value(bool) = null,

    /// Worker-pool size. 0 = auto (cpu_count - 2, clamped to at least 1).
    n_workers: usize = 0,
};

// The signal handler sets an atomic flag AND writes one byte to
// `g_sig_wake_w` so the main loop's kevent wakes immediately instead of
// waiting for the next tick. `write(2)` is async-signal-safe per POSIX.

var g_sig_flag_ptr: ?*std.atomic.Value(bool) = null;
var g_sig_wake_w: posix.fd_t = -1;

fn sigHandler(_: i32) callconv(.c) void {
    if (g_sig_flag_ptr) |p| p.store(true, .monotonic);
    if (g_sig_wake_w >= 0) _ = posix.write(g_sig_wake_w, "s") catch {};
}

pub fn installSignalHandlers(flag: *std.atomic.Value(bool), wake_w: posix.fd_t) void {
    g_sig_flag_ptr = flag;
    g_sig_wake_w = wake_w;

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

const ConnState = enum {
    inactive,
    read_header,
    read_body,
    awaiting_worker,
    write_response,
};

const Connection = struct {
    fd: posix.fd_t = -1,
    state: ConnState = .inactive,
    read_buf: []u8 = &.{},
    write_buf: []u8 = &.{},
    read_pos: usize = 0,
    header: protocol.Header = .{ .body_len = 0, .tag = 0, .req_id = 0 },
    write_pos: usize = 0,
    write_end: usize = 0,
    last_activity_ns: i128 = 0,
    /// When set, close after draining the current write.
    close_after_write: bool = false,
    /// Completion for the IO wait on this conn. Only one at a time.
    io_completion: IO.Completion = .{},
};

fn setNonBlocking(fd: posix.fd_t) !void {
    const current = try posix.fcntl(fd, posix.F.GETFL, 0);
    const nonblock_u32: u32 = @bitCast(posix.O{ .NONBLOCK = true });
    _ = try posix.fcntl(fd, posix.F.SETFL, current | @as(usize, nonblock_u32));
}

/// Rough ceiling on how often we recompute idle-timeouts / auto-snapshot.
/// 100 ms is tight enough for tests and lax enough for production.
const tick_interval_ns: i64 = 100 * std.time.ns_per_ms;

pub fn Server(comptime M: usize) type {
    const Index = HnswIndexFn(M);
    const Dispatcher = dispatcher_mod.Dispatcher(M);

    return struct {
        const Self = @This();
        pub const IndexType = Index;

        allocator: std.mem.Allocator,
        store: *Store,
        index: *Index,
        metadata: *MutableMetadata,
        embedder: ?ollama.Embedder,
        cfg: *const config_mod.Config,
        opts: ServeOptions,

        listener: std.net.Server,
        bound_port: u16,

        io: IO,
        accept_completion: IO.Completion = .{},
        wake_completion: IO.Completion = .{},
        shutdown_completion: IO.Completion = .{},
        tick_completion: IO.Completion = .{},

        /// Signal handler writes one byte here; main thread's readReady
        /// callback flips the shutdown flag.
        shutdown_r: posix.fd_t,
        shutdown_w: posix.fd_t,

        conns: []Connection,
        per_conn_buf_size: usize,

        dispatcher: Dispatcher,

        /// Points at the caller-provided flag (or our own fallback when
        /// none was provided). Checked between runForNs iterations.
        shutdown_flag: *std.atomic.Value(bool),
        /// Backing storage when the caller didn't pass `opts.shutdown_flag`.
        own_shutdown_flag: std.atomic.Value(bool) = .init(false),

        snapshot_interval_ns: i128,
        last_snapshot_ns: i128,

        /// Small scratch for the auto-snapshot response payload. The
        /// internal snapshot has nowhere to send this, but handlers
        /// write here unconditionally.
        snap_scratch: [64]u8 = undefined,

        pub fn init(
            allocator: std.mem.Allocator,
            store: *Store,
            index: *Index,
            md: *MutableMetadata,
            embedder: ?ollama.Embedder,
            cfg: *const config_mod.Config,
            opts: ServeOptions,
        ) !Self {
            // Per-connection buffer sizing — same policy as the old
            // reactor. Worst-case across INSERT_TEXT payload, INSERT_VEC
            // payload, and SEARCH response.
            const header_overhead = protocol.FRAME_HEADER_SIZE + 16;
            const search_resp_max = protocol.FRAME_HEADER_SIZE + 2 +
                cfg.index.max_ef * (10 + MAX_NAME_IN_RESPONSE);
            const text_max = protocol.FRAME_HEADER_SIZE + cfg.embedder.max_text_bytes + header_overhead;
            const vec_max = protocol.FRAME_HEADER_SIZE + cfg.embedder.dim * 4 + header_overhead;
            const per_conn_buf_size = @max(text_max, @max(vec_max, search_resp_max)) + 4096;

            const addr = try std.net.Address.parseIp(opts.listen_addr, opts.listen_port);
            var listener = try addr.listen(.{
                .reuse_address = opts.reuse_address,
            });
            errdefer listener.deinit();
            // Listener stays blocking; accept(2) is gated by kqueue so
            // we only call it when there's a pending connection.
            const bound = listener.listen_address;
            const bound_port: u16 = bound.getPort();

            var io = try IO.init(allocator);
            errdefer io.deinit();

            const fds = try posix.pipe2(.{ .NONBLOCK = true });
            errdefer posix.close(fds[0]);
            errdefer posix.close(fds[1]);

            const conns = try allocator.alloc(Connection, opts.max_connections);
            errdefer allocator.free(conns);
            for (conns) |*c| c.* = .{};

            const read_slab = try allocator.alloc(u8, @as(usize, opts.max_connections) * per_conn_buf_size);
            errdefer allocator.free(read_slab);
            const write_slab = try allocator.alloc(u8, @as(usize, opts.max_connections) * per_conn_buf_size);
            errdefer allocator.free(write_slab);
            for (conns, 0..) |*c, i| {
                c.read_buf = read_slab[i * per_conn_buf_size ..][0..per_conn_buf_size];
                c.write_buf = write_slab[i * per_conn_buf_size ..][0..per_conn_buf_size];
            }

            const dispatcher = try Dispatcher.init(
                allocator,
                store,
                index,
                md,
                embedder,
                cfg,
                .{
                    .n_workers = opts.n_workers,
                    .request_pool_size = opts.max_connections,
                },
            );

            return .{
                .allocator = allocator,
                .store = store,
                .index = index,
                .metadata = md,
                .embedder = embedder,
                .cfg = cfg,
                .opts = opts,
                .listener = listener,
                .bound_port = bound_port,
                .io = io,
                .shutdown_r = fds[0],
                .shutdown_w = fds[1],
                .conns = conns,
                .per_conn_buf_size = per_conn_buf_size,
                .dispatcher = dispatcher,
                .shutdown_flag = opts.shutdown_flag orelse @as(*std.atomic.Value(bool), undefined), // fixed below
                .snapshot_interval_ns = @as(i128, opts.auto_snapshot_secs) * std.time.ns_per_s,
                .last_snapshot_ns = std.time.nanoTimestamp(),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.conns) |*c| {
                if (c.state != .inactive) {
                    posix.close(c.fd);
                    c.fd = -1;
                }
            }
            self.listener.deinit();
            // One slab each; take the slab pointer from conn[0].
            if (self.conns.len > 0) {
                self.allocator.free(self.conns[0].read_buf.ptr[0 .. self.per_conn_buf_size * self.conns.len]);
                self.allocator.free(self.conns[0].write_buf.ptr[0 .. self.per_conn_buf_size * self.conns.len]);
            }
            self.allocator.free(self.conns);

            posix.close(self.shutdown_r);
            posix.close(self.shutdown_w);

            self.dispatcher.deinit();
            self.io.deinit();
        }

        /// Main event loop. Called from the main thread.
        ///
        /// `shutdown_flag` is the same atomic the caller would have
        /// passed in `opts.shutdown_flag` (kept as a parameter for API
        /// compatibility with the pre-rewrite server).
        pub fn run(self: *Self, shutdown_flag: *std.atomic.Value(bool)) !void {
            self.shutdown_flag = shutdown_flag;
            log.info("listening on {s}:{d}", .{ self.opts.listen_addr, self.bound_port });

            // Spawn worker threads.
            try self.dispatcher.start();
            errdefer self.dispatcher.shutdown();

            // Submit long-lived completions.
            self.io.readReady(*Self, self, onAccept, &self.accept_completion, self.listener.stream.handle);
            self.io.readReady(*Self, self, onDispatcherDone, &self.wake_completion, self.dispatcher.wake_r);
            self.io.readReady(*Self, self, onShutdownByte, &self.shutdown_completion, self.shutdown_r);
            self.io.timeout(*Self, self, onTick, &self.tick_completion, tick_interval_ns);

            while (!shutdown_flag.load(.monotonic)) {
                try self.io.runForNs(tick_interval_ns);
            }

            log.info("shutdown requested; flushing snapshot and closing", .{});

            self.dispatcher.shutdown();

            if (!self.opts.skip_final_snapshot) self.finalSnapshot();

            for (self.conns, 0..) |*c, i| {
                if (c.state != .inactive) self.closeConn(i);
            }
        }

        /// Convenience: write to our own shutdown pipe, then flip the
        /// flag. Tests and external callers may use this instead of
        /// racing on the atomic alone.
        pub fn requestShutdown(self: *Self) void {
            _ = posix.write(self.shutdown_w, "s") catch {};
            self.shutdown_flag.store(true, .monotonic);
        }

        fn onAccept(self: *Self, _: *IO.Completion, result: IO.Result) void {
            // Unconditionally re-arm for the next connection. Any error
            // is fatal-ish but we keep going so the loop doesn't wedge.
            defer self.io.readReady(*Self, self, onAccept, &self.accept_completion, self.listener.stream.handle);

            switch (result) {
                .ready => {},
                else => return,
            }

            const conn = self.listener.accept() catch |err| {
                if (err != error.WouldBlock) {
                    log.warn("accept failed: {s}", .{@errorName(err)});
                }
                return;
            };

            // Find an inactive slot.
            var slot_opt: ?usize = null;
            for (self.conns, 0..) |*cc, i| {
                if (cc.state == .inactive) {
                    slot_opt = i;
                    break;
                }
            }
            const slot = slot_opt orelse {
                log.info("rejecting accept: max_connections reached", .{});
                conn.stream.close();
                return;
            };

            setNonBlocking(conn.stream.handle) catch |err| {
                log.warn("failed to set non-blocking: {s}", .{@errorName(err)});
                conn.stream.close();
                return;
            };

            const c = &self.conns[slot];
            c.fd = conn.stream.handle;
            c.state = .read_header;
            c.read_pos = 0;
            c.header = .{ .body_len = 0, .tag = 0, .req_id = 0 };
            c.write_pos = 0;
            c.write_end = 0;
            c.last_activity_ns = std.time.nanoTimestamp();
            c.close_after_write = false;

            // Arm the first readReady for this conn.
            self.armRead(slot);
        }

        fn closeConn(self: *Self, i: usize) void {
            const c = &self.conns[i];
            if (c.state == .inactive) return;
            posix.close(c.fd);
            c.fd = -1;
            c.state = .inactive;
            c.read_pos = 0;
            c.write_pos = 0;
            c.write_end = 0;
            c.close_after_write = false;
        }

        // Convenience: look up the conn slot for a completion.
        fn connIndexFor(self: *Self, c: *IO.Completion) usize {
            const base = @intFromPtr(&self.conns[0].io_completion);
            const step = @sizeOf(Connection);
            const idx = (@intFromPtr(c) - base) / step;
            std.debug.assert(idx < self.conns.len);
            return idx;
        }

        fn armRead(self: *Self, i: usize) void {
            const c = &self.conns[i];
            self.io.readReady(*Self, self, onConnReadReady, &c.io_completion, c.fd);
        }

        fn onConnReadReady(self: *Self, comp: *IO.Completion, result: IO.Result) void {
            const i = self.connIndexFor(comp);
            const c = &self.conns[i];
            if (c.state != .read_header and c.state != .read_body) return;

            switch (result) {
                .ready => {},
                else => {
                    self.closeConn(i);
                    return;
                },
            }

            self.progressRead(i);
        }

        fn progressRead(self: *Self, i: usize) void {
            const c = &self.conns[i];

            const need: usize = switch (c.state) {
                .read_header => protocol.FRAME_HEADER_SIZE,
                .read_body => protocol.totalFrameSize(c.header),
                else => return,
            };
            if (c.read_pos >= need) {
                self.tryAdvance(i);
                return;
            }

            const dst = c.read_buf[c.read_pos..need];
            const n = posix.read(c.fd, dst) catch |err| switch (err) {
                error.WouldBlock => {
                    // Nothing ready yet — re-arm.
                    self.armRead(i);
                    return;
                },
                else => {
                    self.closeConn(i);
                    return;
                },
            };
            if (n == 0) {
                // Peer closed.
                self.closeConn(i);
                return;
            }
            c.read_pos += n;
            c.last_activity_ns = std.time.nanoTimestamp();

            self.tryAdvance(i);
        }

        fn tryAdvance(self: *Self, i: usize) void {
            const c = &self.conns[i];
            while (true) {
                switch (c.state) {
                    .read_header => {
                        if (c.read_pos < protocol.FRAME_HEADER_SIZE) {
                            self.armRead(i);
                            return;
                        }
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
                            self.dispatch(i);
                            return;
                        }
                        // Fall through: we may have body bytes queued in
                        // the same recv.
                    },
                    .read_body => {
                        const total = protocol.totalFrameSize(c.header);
                        if (c.read_pos < total) {
                            self.armRead(i);
                            return;
                        }
                        self.dispatch(i);
                        return;
                    },
                    else => return,
                }
            }
        }

        fn dispatch(self: *Self, i: usize) void {
            const c = &self.conns[i];
            const header = c.header;
            const payload = c.read_buf[protocol.FRAME_HEADER_SIZE..][0..protocol.payloadLen(header)];
            const opcode: protocol.Opcode = @enumFromInt(header.tag);

            // PING and CLOSE stay on the main thread — they do no real
            // work and going through the worker pool would skew the
            // protocol-floor benchmarks and add latency.
            switch (opcode) {
                .ping => {
                    if (payload.len != 0) {
                        self.writeImmediateErr(i, .invalid_frame);
                    } else {
                        self.writeImmediateOk(i, 0);
                    }
                    return;
                },
                .close => {
                    c.close_after_write = true;
                    self.writeImmediateOk(i, 0);
                    return;
                },
                else => {},
            }

            // Anything else: grab a pooled request, fill it in, submit.
            const req = self.dispatcher.acquireRequest() orelse {
                // Pool exhausted → server is saturated. Respond BUSY so
                // the client can retry. Keeps the connection alive.
                self.writeImmediateErr(i, .busy);
                return;
            };
            req.conn_idx = i;
            req.req_id = header.req_id;
            req.opcode = opcode;
            req.payload = payload;
            req.response_buf = c.write_buf[protocol.FRAME_HEADER_SIZE..];
            req.status = .ok;
            req.response_len = 0;

            c.state = .awaiting_worker;
            c.read_pos = 0; // fresh header when worker is done
            self.dispatcher.submit(req);
        }

        /// Encode (status, 0-length payload) into conn.write_buf and
        /// start the write phase. For ops handled synchronously on the
        /// main thread (PING, CLOSE).
        fn writeImmediateOk(self: *Self, i: usize, payload_len: usize) void {
            const c = &self.conns[i];
            const body_len: u32 = @intCast(5 + payload_len);
            protocol.encodeHeader(
                c.write_buf[0..protocol.FRAME_HEADER_SIZE],
                body_len,
                @intFromEnum(protocol.Status.ok),
                c.header.req_id,
            );
            c.write_pos = 0;
            c.write_end = protocol.FRAME_HEADER_SIZE + payload_len;
            c.state = .write_response;
            c.read_pos = 0;
            self.progressWrite(i);
        }

        fn writeImmediateErr(self: *Self, i: usize, status: protocol.Status) void {
            const c = &self.conns[i];
            const msg = protocol.statusMessage(status);
            const payload_len = protocol.encodeErrorPayload(
                c.write_buf[protocol.FRAME_HEADER_SIZE..],
                msg,
            );
            const body_len: u32 = @intCast(5 + payload_len);
            protocol.encodeHeader(
                c.write_buf[0..protocol.FRAME_HEADER_SIZE],
                body_len,
                @intFromEnum(status),
                c.header.req_id,
            );
            c.write_pos = 0;
            c.write_end = protocol.FRAME_HEADER_SIZE + payload_len;
            c.state = .write_response;
            c.read_pos = 0;
            self.progressWrite(i);
        }

        fn onDispatcherDone(self: *Self, _: *IO.Completion, result: IO.Result) void {
            // Always re-arm the read on the wake pipe. Worker bytes may
            // arrive in the interval between drain and re-arm; those
            // will wake the next readiness edge.
            defer self.io.readReady(
                *Self,
                self,
                onDispatcherDone,
                &self.wake_completion,
                self.dispatcher.wake_r,
            );

            switch (result) {
                .ready => {},
                else => return,
            }

            var out: [32]*dispatcher_mod.InflightRequest = undefined;
            while (true) {
                const n = self.dispatcher.drainDone(&out);
                if (n == 0) break;
                for (out[0..n]) |req| {
                    self.onRequestDone(req);
                }
                if (n < out.len) break;
            }
        }

        fn onRequestDone(self: *Self, req: *dispatcher_mod.InflightRequest) void {
            // Detached auto-snapshot has no conn to reply on.
            if (req.conn_idx == std.math.maxInt(usize)) {
                if (req.status != .ok) {
                    log.warn("auto-snapshot failed: {s}", .{protocol.statusMessage(req.status)});
                }
                self.dispatcher.releaseRequest(req);
                return;
            }

            const i = req.conn_idx;
            const c = &self.conns[i];
            // Encode the frame header. The worker has already placed
            // the response payload at write_buf[FRAME_HEADER_SIZE..].
            const payload_end = protocol.FRAME_HEADER_SIZE + req.response_len;
            const body_len: u32 = @intCast(5 + req.response_len);
            protocol.encodeHeader(
                c.write_buf[0..protocol.FRAME_HEADER_SIZE],
                body_len,
                @intFromEnum(req.status),
                req.req_id,
            );
            c.write_pos = 0;
            c.write_end = payload_end;
            c.state = .write_response;
            c.read_pos = 0;

            self.dispatcher.releaseRequest(req);

            // Attempt an immediate drain — saves a kevent round-trip on
            // the common case where the kernel's send buffer has room.
            self.progressWrite(i);
        }

        fn armWrite(self: *Self, i: usize) void {
            const c = &self.conns[i];
            self.io.writeReady(*Self, self, onConnWriteReady, &c.io_completion, c.fd);
        }

        fn onConnWriteReady(self: *Self, comp: *IO.Completion, result: IO.Result) void {
            const i = self.connIndexFor(comp);
            const c = &self.conns[i];
            if (c.state != .write_response) return;

            switch (result) {
                .ready => {},
                else => {
                    self.closeConn(i);
                    return;
                },
            }

            self.progressWrite(i);
        }

        fn progressWrite(self: *Self, i: usize) void {
            const c = &self.conns[i];
            while (c.write_pos < c.write_end) {
                const slice = c.write_buf[c.write_pos..c.write_end];
                const n = posix.write(c.fd, slice) catch |err| switch (err) {
                    error.WouldBlock => {
                        self.armWrite(i);
                        return;
                    },
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

            if (c.close_after_write) {
                self.closeConn(i);
                return;
            }
            c.state = .read_header;
            c.read_pos = 0;
            self.armRead(i);
        }

        fn onTick(self: *Self, _: *IO.Completion, _: IO.Result) void {
            // Always re-arm.
            defer self.io.timeout(*Self, self, onTick, &self.tick_completion, tick_interval_ns);
            self.checkIdleTimeouts();
            self.maybeAutoSnapshot();
        }

        fn checkIdleTimeouts(self: *Self) void {
            if (self.opts.idle_timeout_secs == 0) return;
            const cutoff_ns: i128 = std.time.nanoTimestamp() -
                @as(i128, self.opts.idle_timeout_secs) * std.time.ns_per_s;
            for (self.conns, 0..) |*c, i| {
                if (c.state == .inactive) continue;
                // Don't idle-close a conn with a worker result pending.
                if (c.state == .awaiting_worker) continue;
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
            // Route the snapshot through the dispatcher so it takes the
            // exclusive lock; we don't block the main loop here.
            // Simplest way: acquire a request, target SNAPSHOT, submit.
            const req = self.dispatcher.acquireRequest() orelse return;
            // We don't have a conn to send the response back to — the
            // auto-snapshot is internal. We synthesize a detached
            // request: response_buf = an inline scratch, and on done we
            // just release it without writing to a conn. A sentinel
            // conn_idx == max tells onDispatcherDone to skip.
            req.conn_idx = std.math.maxInt(usize);
            req.opcode = .snapshot;
            req.payload = &.{};
            req.response_buf = self.snap_scratch[0..];
            req.status = .ok;
            req.response_len = 0;
            self.dispatcher.submit(req);
            self.last_snapshot_ns = started;
        }

        fn onShutdownByte(self: *Self, _: *IO.Completion, result: IO.Result) void {
            // Drain the pipe; any byte means "shut down".
            var buf: [16]u8 = undefined;
            _ = posix.read(self.shutdown_r, &buf) catch {};
            self.shutdown_flag.store(true, .monotonic);
            // Don't re-arm — the loop's main `while` will exit.
            _ = result;
        }

        fn finalSnapshot(self: *Self) void {
            // Workers have already stopped by the time we call this.
            // No concurrency; just do the snapshot directly.
            self.snapshotInline() catch |err| {
                log.err("final snapshot failed: {s}", .{@errorName(err)});
            };
        }

        fn snapshotInline(self: *Self) !void {
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
        }
    };
}

// Tests spin up the server on 127.0.0.1:0 with a FakeEmbedder and use a
// blocking client socket for round-trips. The server runs on a worker
// thread; tests call `srv.requestShutdown()` to exit cleanly. Signal
// handlers are NOT installed in tests — they'd pollute the binary.

const testing = std.testing;

const TestM: usize = 16;

const TestHarness = struct {
    gpa_allocator: std.mem.Allocator,
    cfg: config_mod.Config,
    store: Store,
    index: HnswIndexFn(TestM),
    md: MutableMetadata,
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
        self.embedder = ollama.FakeEmbedder.init(dim, 0x5eed);
        const emb_opt = self.embedder.embedder();
        self.srv = try Server(TestM).init(
            allocator,
            &self.store,
            &self.index,
            &self.md,
            emb_opt,
            &self.cfg,
            .{
                .listen_addr = "127.0.0.1",
                .listen_port = 0,
                .max_connections = max_connections,
                .max_frame_bytes = 1 << 20,
                .idle_timeout_secs = 0,
                .auto_snapshot_secs = 0,
                .reuse_address = true,
                .skip_final_snapshot = true,
                .n_workers = 2, // small but still exercises the pool
            },
        );
    }

    fn start(self: *TestHarness) !void {
        self.thread = try std.Thread.spawn(.{}, runServer, .{self});
    }

    fn tearDown(self: *TestHarness) void {
        self.srv.requestShutdown();
        if (self.thread) |t| t.join();
        self.thread = null;
        self.srv.deinit();
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
    try testing.expectEqual(@as(u8, 0), r.header.tag);
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
    try h.setUp(testing.allocator, dim, 2, 4);
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

    var hdr: [protocol.FRAME_HEADER_SIZE]u8 = undefined;
    protocol.encodeHeader(&hdr, 5, @intFromEnum(protocol.Opcode.ping), 42);
    for (hdr) |b| try stream.writeAll(&[_]u8{b});

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

    var loaded = try Store.load(testing.allocator, tmp.dir, h.cfg.storage.vectors_file, null);
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), loaded.live_count);
    try testing.expectEqualSlices(u8, vb, std.mem.sliceAsBytes(loaded.get(0)));
}

test "auto-snapshot fires without crashing (detached request sentinel)" {
    var h: TestHarness = undefined;
    const dim: usize = 4;
    try h.setUp(testing.allocator, dim, 8, 4);
    defer h.tearDown();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const real = try tmp.dir.realpathAlloc(testing.allocator, ".");
    defer testing.allocator.free(real);
    h.cfg.storage.data_dir = real;

    // Force auto-snapshot to fire on the very next tick.
    h.srv.snapshot_interval_ns = 1;
    h.srv.last_snapshot_ns = 0;

    const port = h.srv.bound_port;
    try h.start();

    const stream = try connect(port);
    defer stream.close();

    // Insert something so the snapshot has payload.
    var vec: [dim]f32 = .{ 1, 0, 0, 0 };
    const vb = std.mem.sliceAsBytes(vec[0..]);
    var ireq: [2 + dim * 4]u8 = undefined;
    const in = protocol.encodeInsertVecRequest(&ireq, 0, vb);
    var resp: [128]u8 = undefined;
    _ = try roundTrip(stream, .insert_vec, 1, ireq[0..in], &resp);

    // Give the tick timer (100 ms) a few iterations to fire and the
    // auto-snapshot to complete via the worker pool.
    std.Thread.sleep(400 * std.time.ns_per_ms);

    // Server must still be responsive — this is the regression guard
    // for the OOB read on the detached conn_idx sentinel.
    const r = try roundTrip(stream, .ping, 2, "", &resp);
    try testing.expectEqual(@as(u8, 0), r.header.tag);

    // Snapshot files should exist on disk.
    var loaded = try Store.load(testing.allocator, tmp.dir, h.cfg.storage.vectors_file, null);
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), loaded.live_count);
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
    try h.setUp(testing.allocator, 4, 8, 1);
    defer h.tearDown();
    const port = h.srv.bound_port;
    try h.start();

    const s1 = try connect(port);
    defer s1.close();
    var resp: [64]u8 = undefined;
    _ = try roundTrip(s1, .ping, 1, "", &resp);

    const s2 = try connect(port);
    defer s2.close();
    var buf: [16]u8 = undefined;
    const n = s2.read(&buf) catch 0;
    try testing.expectEqual(@as(usize, 0), n);
}
