//! Worker-pool dispatcher for HNSW compute.
//!
//! The server's `poll()` reactor used to call `HnswIndex.insert / .search`
//! synchronously on the reactor thread. At dim=4096 a single insert runs
//! ~19 ms and a single search ~3.6 ms, so every request stalled every
//! other client. This module moves that compute to a pool of worker
//! threads and coordinates via:
//!
//!   * a `std.Thread.RwLock` around the index/store/metadata triple
//!     (search holds it shared; insert/delete/replace/snapshot hold it
//!     exclusive),
//!   * a single pending queue the workers block on,
//!   * a single done queue the main thread drains,
//!   * a plain pipe(2) the workers write one byte to when a request is
//!     complete, registered on the main thread's `IO` instance as a
//!     `readReady` completion so the loop wakes without polling.
//!
//! This departs deliberately from TigerBeetle's pure-single-threaded
//! model
//! why: our ops are not bounded enough to run on the loop thread.
//!
//! The module is generic over `M` (HNSW fan-out) just like `HnswIndex`
//! so the compile-time M propagates through.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const assert = std.debug.assert;

const protocol = @import("protocol.zig");
const Store = @import("store.zig").Store;
const HnswIndexFn = @import("hnsw.zig").HnswIndex;
const MutableMetadata = @import("metadata_mut.zig").MutableMetadata;
const ollama = @import("ollama.zig");
const config_mod = @import("config.zig");
const heap = @import("heap.zig");
const wal_mod = @import("wal.zig");

const log = std.log.scoped(.dispatcher);

/// Must match server's clipping cap (see server.zig). Duplicated to keep
/// the dispatcher self-contained.
pub const MAX_NAME_IN_RESPONSE: usize = 256;

/// In-flight work item. The server allocates these from a fixed pool at
/// startup and recycles them once a request completes. No allocation on
/// the hot path.
///
/// Lifecycle:
///   * main thread: `acquire`, fill payload/response slices + opcode +
///     req_id + conn_idx, call `submit`.
///   * worker: pops from pending, runs handler, writes status + response
///     bytes, pushes to done, writes one byte to the wake pipe.
///   * main thread: reads pipe → `drainDone` → sends response on conn →
///     `release`.
///
/// Invariant: once a request is in flight, its connection slot is NOT
/// reused by a new `accept`. That's enforced by the server's per-conn
/// state (the slot stays in `.awaiting_worker` until the main thread
/// receives the done notification and writes the response).
pub const InflightRequest = struct {
    /// Slot index in the server's connection array.
    conn_idx: usize = 0,
    /// Echoed back in the response header.
    req_id: u32 = 0,
    /// What the worker must run.
    opcode: protocol.Opcode = .ping,
    /// Payload bytes borrowed from the conn's read buffer. Valid until
    /// the worker posts `.response_len`.
    payload: []const u8 = &.{},
    /// Writable slice borrowed from the conn's write buffer, after the
    /// 9-byte frame-header reservation. Worker writes response payload
    /// here. Main thread writes the frame header before sending.
    response_buf: []u8 = &.{},

    /// Worker fills these.
    status: protocol.Status = .ok,
    response_len: usize = 0,

    /// Intrusive pointer for pending/done queues.
    next: ?*InflightRequest = null,

    /// Set by main thread to recycle the slot.
    pub fn reset(self: *InflightRequest) void {
        self.* = .{};
    }
};

const RequestPool = struct {
    slots: []InflightRequest,
    /// Indices of free slots in `slots`. LIFO for cache locality.
    free_stack: []usize,
    free_len: usize,
    mutex: std.Thread.Mutex = .{},

    fn init(allocator: std.mem.Allocator, capacity: usize) !RequestPool {
        assert(capacity > 0);
        const slots = try allocator.alloc(InflightRequest, capacity);
        errdefer allocator.free(slots);
        const free_stack = try allocator.alloc(usize, capacity);
        for (slots) |*s| s.* = .{};
        for (free_stack, 0..) |*idx, i| idx.* = capacity - 1 - i; // 0..N-1 LIFO
        return .{
            .slots = slots,
            .free_stack = free_stack,
            .free_len = capacity,
        };
    }

    fn deinit(self: *RequestPool, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
        allocator.free(self.free_stack);
    }

    fn acquire(self: *RequestPool) ?*InflightRequest {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.free_len == 0) return null;
        self.free_len -= 1;
        const idx = self.free_stack[self.free_len];
        self.slots[idx].reset();
        return &self.slots[idx];
    }

    fn release(self: *RequestPool, req: *InflightRequest) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const idx_ptr = @intFromPtr(req) - @intFromPtr(self.slots.ptr);
        const idx: usize = idx_ptr / @sizeOf(InflightRequest);
        assert(idx < self.slots.len);
        self.free_stack[self.free_len] = idx;
        self.free_len += 1;
    }
};

pub fn Dispatcher(comptime M: usize) type {
    const Index = HnswIndexFn(M);

    return struct {
        const Self = @This();
        pub const IndexType = Index;

        allocator: std.mem.Allocator,

        // Shared state — protected by `rwlock`.
        rwlock: std.Thread.RwLock = .{},
        store: *Store,
        index: *Index,
        metadata: *MutableMetadata,
        /// Optional WAL handle. When non-null, every mutation is
        /// appended + fsync'd before the in-memory apply. When null,
        /// writes proceed without durability — used for benchmarks and
        /// tests where we care about throughput over crash safety.
        wal: ?*wal_mod.Wal,

        // Effectively read-only after init.
        embedder: ?ollama.Embedder,
        cfg: *const config_mod.Config,

        // Worker pool.
        workers: []Worker,
        threads: []std.Thread,
        // Set true before broadcasting pending_cond to unblock workers on
        // shutdown. Observed under `pending_mutex`.
        shutting_down: bool = false,

        // Pending work queue.
        pending_mutex: std.Thread.Mutex = .{},
        pending_cond: std.Thread.Condition = .{},
        pending_head: ?*InflightRequest = null,
        pending_tail: ?*InflightRequest = null,
        pending_len: usize = 0,

        // Completed-but-not-yet-drained queue.
        done_mutex: std.Thread.Mutex = .{},
        done_head: ?*InflightRequest = null,
        done_tail: ?*InflightRequest = null,

        // Pipe the workers write to to wake the main loop. Main owns the
        // read end and registers `readReady` on it via IO.
        wake_r: posix.fd_t,
        wake_w: posix.fd_t,

        pool: RequestPool,

        pub const Worker = struct {
            dispatcher: *Self,
            idx: usize,
            ws: Index.Workspace,
            vec_scratch: []align(4) f32,
            results_scratch: []heap.Entry,
            /// Scratch for outgoing SEARCH response cursoring. Sized to the
            /// worst-case response.
            _pad: [7]u8 = undefined,
        };

        pub const Options = struct {
            /// Number of worker threads. Default 0 = use cpu_count.
            n_workers: usize = 0,
            /// Max concurrent in-flight requests. Usually = max_connections.
            request_pool_size: usize = 64,
        };

        pub fn init(
            allocator: std.mem.Allocator,
            store: *Store,
            index: *Index,
            metadata: *MutableMetadata,
            wal: ?*wal_mod.Wal,
            embedder: ?ollama.Embedder,
            cfg: *const config_mod.Config,
            opts: Options,
        ) !Self {
            const n = blk: {
                if (opts.n_workers > 0) break :blk opts.n_workers;
                const cpu_count = std.Thread.getCpuCount() catch 4;
                // Leave one core for the event loop + a little headroom.
                break :blk @max(1, cpu_count -| 2);
            };

            const workers = try allocator.alloc(Worker, n);
            errdefer allocator.free(workers);
            const threads = try allocator.alloc(std.Thread, n);
            errdefer allocator.free(threads);

            var initialized: usize = 0;
            errdefer for (workers[0..initialized]) |*w| {
                w.ws.deinit(allocator);
                allocator.free(w.vec_scratch);
                allocator.free(w.results_scratch);
            };

            for (workers, 0..) |*w, i| {
                w.* = .{
                    .dispatcher = undefined,
                    .idx = i,
                    .ws = try Index.Workspace.init(
                        allocator,
                        cfg.storage.max_vectors,
                        cfg.index.max_ef,
                    ),
                    .vec_scratch = try allocator.alignedAlloc(f32, .@"4", cfg.embedder.dim),
                    .results_scratch = try allocator.alloc(heap.Entry, cfg.index.max_ef),
                };
                initialized += 1;
            }

            const fds = try posix.pipe2(.{ .NONBLOCK = true });
            errdefer posix.close(fds[0]);
            errdefer posix.close(fds[1]);

            const pool = try RequestPool.init(allocator, opts.request_pool_size);

            return .{
                .allocator = allocator,
                .store = store,
                .index = index,
                .metadata = metadata,
                .wal = wal,
                .embedder = embedder,
                .cfg = cfg,
                .workers = workers,
                .threads = threads,
                .wake_r = fds[0],
                .wake_w = fds[1],
                .pool = pool,
            };
        }

        /// Spawn the worker threads. Separate from `init` so the caller
        /// can place `Self` at a stable address before workers hold
        /// pointers to it.
        pub fn start(self: *Self) !void {
            for (self.workers, 0..) |*w, i| {
                w.dispatcher = self;
                self.threads[i] = try std.Thread.spawn(.{}, workerLoop, .{w});
            }
        }

        /// Signal shutdown and join all workers. Safe to call once.
        pub fn shutdown(self: *Self) void {
            {
                self.pending_mutex.lock();
                defer self.pending_mutex.unlock();
                self.shutting_down = true;
                self.pending_cond.broadcast();
            }
            for (self.threads) |t| t.join();
        }

        pub fn deinit(self: *Self) void {
            posix.close(self.wake_r);
            posix.close(self.wake_w);
            for (self.workers) |*w| {
                w.ws.deinit(self.allocator);
                self.allocator.free(w.vec_scratch);
                self.allocator.free(w.results_scratch);
            }
            self.allocator.free(self.workers);
            self.allocator.free(self.threads);
            self.pool.deinit(self.allocator);
        }

        pub fn acquireRequest(self: *Self) ?*InflightRequest {
            return self.pool.acquire();
        }

        pub fn releaseRequest(self: *Self, req: *InflightRequest) void {
            self.pool.release(req);
        }

        pub fn submit(self: *Self, req: *InflightRequest) void {
            self.pending_mutex.lock();
            defer self.pending_mutex.unlock();
            req.next = null;
            if (self.pending_tail) |t| t.next = req else self.pending_head = req;
            self.pending_tail = req;
            self.pending_len += 1;
            self.pending_cond.signal();
        }

        /// Drain up to `out.len` completed requests. Returns the number
        /// of items written to `out`. Does NOT consume bytes from the
        /// wake pipe; caller is responsible for draining the pipe (one
        /// byte per request pushed; we read them all into a throwaway
        /// buffer).
        pub fn drainDone(self: *Self, out: []*InflightRequest) usize {
            // Drain the wake pipe first. Workers may have written more
            // bytes than requests if several completed between our
            // readReady wakeups; we read up to 64 at once, enough to
            // unblock the pipe.
            var trash: [64]u8 = undefined;
            _ = posix.read(self.wake_r, &trash) catch {};

            self.done_mutex.lock();
            defer self.done_mutex.unlock();

            var n: usize = 0;
            while (n < out.len) {
                const h = self.done_head orelse break;
                self.done_head = h.next;
                if (self.done_head == null) self.done_tail = null;
                h.next = null;
                out[n] = h;
                n += 1;
            }
            return n;
        }

        fn workerLoop(worker: *Worker) void {
            const self = worker.dispatcher;
            while (true) {
                const req = self.popPending() orelse return;
                handle(worker, req);
                self.pushDone(req);
                // Wake the main loop. Best-effort: if the pipe is full
                // we drop the byte, but the main loop will still discover
                // pending items on its next wake (from another request
                // or the tick timeout). Pipe is 64 KiB on Darwin, so
                // we'd need 65 536 pending wakeups for this to fire.
                _ = posix.write(self.wake_w, "x") catch {};
            }
        }

        fn popPending(self: *Self) ?*InflightRequest {
            self.pending_mutex.lock();
            defer self.pending_mutex.unlock();
            while (self.pending_head == null) {
                if (self.shutting_down) return null;
                self.pending_cond.wait(&self.pending_mutex);
            }
            const h = self.pending_head.?;
            self.pending_head = h.next;
            if (self.pending_head == null) self.pending_tail = null;
            h.next = null;
            self.pending_len -= 1;
            return h;
        }

        fn pushDone(self: *Self, req: *InflightRequest) void {
            self.done_mutex.lock();
            defer self.done_mutex.unlock();
            req.next = null;
            if (self.done_tail) |t| t.next = req else self.done_head = req;
            self.done_tail = req;
        }

        fn handle(worker: *Worker, req: *InflightRequest) void {
            const op = req.opcode;
            // Each handler writes `req.status` and `req.response_len`.
            switch (op) {
                .stats => handleStats(worker, req),
                .insert_vec => handleInsertVec(worker, req),
                .insert_text => handleInsertText(worker, req),
                .delete => handleDelete(worker, req),
                .replace_vec => handleReplaceVec(worker, req),
                .replace_text => handleReplaceText(worker, req),
                .get => handleGet(worker, req),
                .search_vec => handleSearchVec(worker, req),
                .search_text => handleSearchText(worker, req),
                .snapshot => handleSnapshot(worker, req),
                else => {
                    // ping and close shouldn't reach the worker; main
                    // thread handles those synchronously. If one does,
                    // report it as unsupported rather than crashing.
                    writeErr(req, .unsupported_opcode);
                },
            }
        }

        // Each handler:
        //   * reads `req.payload`
        //   * holds `rwlock.lockShared()` for read-only ops or
        //     `rwlock.lock()` for mutating ops (snapshot also uses
        //     exclusive to avoid racing a writer's mid-insert state),
        //   * writes response bytes into `req.response_buf`,
        //   * sets `req.response_len` and `req.status`.

        fn handleStats(worker: *Worker, req: *InflightRequest) void {
            if (req.payload.len != 0) return writeErr(req, .invalid_frame);
            const self = worker.dispatcher;

            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();

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
            req.response_len = protocol.encodeStatsResponse(req.response_buf, s);
            req.status = .ok;
        }

        /// Pre-check whether `level` upper-layer slots are available. Mirrors
        /// the logic inside `HnswIndex.insertWithLevel` so we can surface
        /// capacity errors BEFORE writing to the WAL — an acknowledged WAL
        /// record whose apply fails is an unrecoverable inconsistency.
        fn hasUpperSlotRoom(self: *const Self, level: u8) bool {
            if (level == 0) return true;
            const idx = self.index;
            if (idx.free_upper_runs[level].items.len > 0) return true;
            return idx.upper_used + @as(usize, level) <= idx.max_upper_slots;
        }

        /// Single entry point for "apply raced ahead of WAL" panic paths.
        /// By the time we call this, a WAL record has been fsync'd but
        /// the in-memory apply failed. The two states are now divergent
        /// and there is no safe way to continue; refuse to serve any
        /// more requests and let the process exit.
        fn panicPostWalApply(self: *Self, comptime op: []const u8, err: anyerror) noreturn {
            _ = self;
            std.debug.panic(
                "hnswz: " ++ op ++
                    " failed after WAL fsync (err={s}). In-memory state diverges from WAL; aborting.",
                .{@errorName(err)},
            );
        }

        fn handleInsertVec(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const view = protocol.decodeInsertVecRequest(req.payload, self.store.dim) catch |err| switch (err) {
                error.DimMismatch => return writeErr(req, .dim_mismatch),
                error.Truncated => return writeErr(req, .invalid_frame),
                else => return writeErr(req, .internal),
            };
            @memcpy(std.mem.sliceAsBytes(worker.vec_scratch), view.vec_bytes);

            self.rwlock.lock();
            defer self.rwlock.unlock();

            const reserved_id = self.store.peekNextId() orelse return writeErr(req, .out_of_capacity);
            const level = self.index.drawLevel();
            if (!self.hasUpperSlotRoom(level)) return writeErr(req, .out_of_capacity);

            if (self.wal) |wal| {
                wal.appendInsert(.{
                    .id = reserved_id,
                    .level = level,
                    .vec = worker.vec_scratch,
                    .name = "",
                }) catch return writeErr(req, .internal);
            }

            const id = self.store.add(worker.vec_scratch) catch |err| self.panicPostWalApply("insert_vec/store.add", err);
            std.debug.assert(id == reserved_id);
            self.index.insertWithLevel(&worker.ws, id, level) catch |err| self.panicPostWalApply("insert_vec/index.insert", err);
            self.metadata.setAt(self.allocator, id, "") catch |err| {
                // Metadata is advisory (the vector + graph are the
                // authoritative state). Log and continue; the WAL has
                // the record so a restart will re-attempt the setAt.
                log.warn("insert_vec: metadata setAt failed: {s}", .{@errorName(err)});
            };
            req.response_len = protocol.encodeIdResponse(req.response_buf, id);
            req.status = .ok;
        }

        fn handleInsertText(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const emb = self.embedder orelse return writeErr(req, .embed_failed);
            const view = protocol.decodeInsertTextRequest(req.payload) catch return writeErr(req, .invalid_frame);
            if (view.text.len > self.cfg.embedder.max_text_bytes) return writeErr(req, .text_too_long);

            // Embed WITHOUT the lock — this is an HTTP call and can
            // take ~seconds; holding the write lock would serialize
            // every other client.
            emb.embed(view.text, worker.vec_scratch) catch return writeErr(req, .embed_failed);

            self.rwlock.lock();
            defer self.rwlock.unlock();

            const reserved_id = self.store.peekNextId() orelse return writeErr(req, .out_of_capacity);
            const level = self.index.drawLevel();
            if (!self.hasUpperSlotRoom(level)) return writeErr(req, .out_of_capacity);

            const name_len = @min(view.text.len, MAX_NAME_IN_RESPONSE);
            const name = view.text[0..name_len];

            if (self.wal) |wal| {
                wal.appendInsert(.{
                    .id = reserved_id,
                    .level = level,
                    .vec = worker.vec_scratch,
                    .name = name,
                }) catch return writeErr(req, .internal);
            }

            const id = self.store.add(worker.vec_scratch) catch |err| self.panicPostWalApply("insert_text/store.add", err);
            std.debug.assert(id == reserved_id);
            self.index.insertWithLevel(&worker.ws, id, level) catch |err| self.panicPostWalApply("insert_text/index.insert", err);
            self.metadata.setAt(self.allocator, id, name) catch |err| {
                log.warn("insert_text: metadata setAt failed: {s}", .{@errorName(err)});
            };
            req.response_len = protocol.encodeIdResponse(req.response_buf, id);
            req.status = .ok;
        }

        fn handleDelete(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const id = protocol.decodeIdRequest(req.payload) catch return writeErr(req, .invalid_frame);

            self.rwlock.lock();
            defer self.rwlock.unlock();
            if (@as(usize, id) >= self.store.count) return writeErr(req, .invalid_id);
            if (self.store.isDeleted(id)) return writeErr(req, .invalid_id);

            if (self.wal) |wal| {
                wal.appendDelete(.{ .id = id }) catch return writeErr(req, .internal);
            }

            self.index.delete(&worker.ws, id) catch |err| self.panicPostWalApply("delete/index.delete", err);
            self.metadata.deleteAt(self.allocator, id);
            req.response_len = 0;
            req.status = .ok;
        }

        fn handleReplaceVec(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const view = protocol.decodeReplaceVecRequest(req.payload, self.store.dim) catch |err| switch (err) {
                error.DimMismatch => return writeErr(req, .dim_mismatch),
                error.Truncated => return writeErr(req, .invalid_frame),
                else => return writeErr(req, .internal),
            };
            @memcpy(std.mem.sliceAsBytes(worker.vec_scratch), view.vec_bytes);

            self.rwlock.lock();
            defer self.rwlock.unlock();
            if (@as(usize, view.id) >= self.store.count) return writeErr(req, .invalid_id);
            if (self.store.isDeleted(view.id)) return writeErr(req, .invalid_id);

            const level = self.index.drawLevel();
            if (!self.hasUpperSlotRoom(level)) return writeErr(req, .out_of_capacity);

            if (self.wal) |wal| {
                wal.appendReplace(.{
                    .id = view.id,
                    .level = level,
                    .vec = worker.vec_scratch,
                    .name = null,
                }) catch return writeErr(req, .internal);
            }

            self.index.replaceVectorWithLevel(&worker.ws, view.id, worker.vec_scratch, level) catch |err|
                self.panicPostWalApply("replace_vec/index.replace", err);
            req.response_len = 0;
            req.status = .ok;
        }

        fn handleReplaceText(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const emb = self.embedder orelse return writeErr(req, .embed_failed);
            const view = protocol.decodeReplaceTextRequest(req.payload) catch return writeErr(req, .invalid_frame);
            if (view.text.len > self.cfg.embedder.max_text_bytes) return writeErr(req, .text_too_long);

            emb.embed(view.text, worker.vec_scratch) catch return writeErr(req, .embed_failed);

            self.rwlock.lock();
            defer self.rwlock.unlock();
            if (@as(usize, view.id) >= self.store.count) return writeErr(req, .invalid_id);
            if (self.store.isDeleted(view.id)) return writeErr(req, .invalid_id);

            const level = self.index.drawLevel();
            if (!self.hasUpperSlotRoom(level)) return writeErr(req, .out_of_capacity);

            const name_len = @min(view.text.len, MAX_NAME_IN_RESPONSE);
            const name = view.text[0..name_len];

            if (self.wal) |wal| {
                wal.appendReplace(.{
                    .id = view.id,
                    .level = level,
                    .vec = worker.vec_scratch,
                    .name = name,
                }) catch return writeErr(req, .internal);
            }

            self.index.replaceVectorWithLevel(&worker.ws, view.id, worker.vec_scratch, level) catch |err|
                self.panicPostWalApply("replace_text/index.replace", err);

            self.metadata.setAt(self.allocator, view.id, name) catch |err| {
                log.warn("replace_text: metadata setAt failed: {s}", .{@errorName(err)});
            };
            req.response_len = 0;
            req.status = .ok;
        }

        fn handleGet(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const id = protocol.decodeIdRequest(req.payload) catch return writeErr(req, .invalid_frame);

            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            if (@as(usize, id) >= self.store.count) return writeErr(req, .invalid_id);
            if (self.store.isDeleted(id)) return writeErr(req, .invalid_id);
            const vec = self.store.get(id);
            const vec_bytes = std.mem.sliceAsBytes(vec);
            const name_full = self.metadata.get(id) orelse "";
            const name = name_full[0..@min(name_full.len, MAX_NAME_IN_RESPONSE)];
            if (req.response_buf.len < 2 + name.len + vec_bytes.len) return writeErr(req, .internal);
            req.response_len = protocol.encodeGetResponse(req.response_buf, name, vec_bytes);
            req.status = .ok;
        }

        fn handleSearchVec(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const view = protocol.decodeSearchVecRequest(req.payload, self.store.dim) catch |err| switch (err) {
                error.DimMismatch => return writeErr(req, .dim_mismatch),
                error.Truncated => return writeErr(req, .invalid_frame),
                else => return writeErr(req, .internal),
            };
            if (view.top_k == 0) return writeErr(req, .invalid_frame);
            if (view.top_k > view.ef) return writeErr(req, .top_k_too_large);
            if (view.ef > self.cfg.index.max_ef) return writeErr(req, .top_k_too_large);

            @memcpy(std.mem.sliceAsBytes(worker.vec_scratch), view.vec_bytes);

            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();

            const out_slots = worker.results_scratch[0..view.top_k];
            const results = self.index.search(
                &worker.ws,
                worker.vec_scratch,
                view.top_k,
                view.ef,
                out_slots,
            ) catch return writeErr(req, .internal);

            const n = encodeSearchResponse(worker, req.response_buf, results) catch return writeErr(req, .internal);
            req.response_len = n;
            req.status = .ok;
        }

        fn handleSearchText(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            const emb = self.embedder orelse return writeErr(req, .embed_failed);
            const view = protocol.decodeSearchTextRequest(req.payload) catch return writeErr(req, .invalid_frame);
            if (view.text.len > self.cfg.embedder.max_text_bytes) return writeErr(req, .text_too_long);
            if (view.top_k == 0) return writeErr(req, .invalid_frame);
            if (view.top_k > view.ef) return writeErr(req, .top_k_too_large);
            if (view.ef > self.cfg.index.max_ef) return writeErr(req, .top_k_too_large);

            emb.embed(view.text, worker.vec_scratch) catch return writeErr(req, .embed_failed);

            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();

            const out_slots = worker.results_scratch[0..view.top_k];
            const results = self.index.search(
                &worker.ws,
                worker.vec_scratch,
                view.top_k,
                view.ef,
                out_slots,
            ) catch return writeErr(req, .internal);

            const n = encodeSearchResponse(worker, req.response_buf, results) catch return writeErr(req, .internal);
            req.response_len = n;
            req.status = .ok;
        }

        fn handleSnapshot(worker: *Worker, req: *InflightRequest) void {
            const self = worker.dispatcher;
            if (req.payload.len != 0) return writeErr(req, .invalid_frame);

            // Snapshot doesn't mutate the graph but we take the exclusive
            // lock so concurrent inserts don't produce a torn on-disk
            // state. Under high write load this briefly stalls
            // everything; acceptable because snapshot is infrequent.
            self.rwlock.lock();
            defer self.rwlock.unlock();

            const started = std.time.nanoTimestamp();
            snapshotNow(self) catch return writeErr(req, .snapshot_failed);
            const elapsed: u64 = @intCast(std.time.nanoTimestamp() - started);
            req.response_len = protocol.encodeSnapshotResponse(req.response_buf, elapsed);
            req.status = .ok;
        }

        fn encodeSearchResponse(worker: *Worker, w: []u8, results: []const heap.Entry) !usize {
            const self = worker.dispatcher;
            var cursor: usize = 0;
            if (w.len < 2) return error.Short;
            cursor += protocol.writeSearchResultCount(w, @intCast(results.len));
            for (results) |r| {
                const name_full = self.metadata.get(r.id) orelse "";
                const name = name_full[0..@min(name_full.len, MAX_NAME_IN_RESPONSE)];
                const need = 10 + name.len;
                if (cursor + need > w.len) return error.Short;
                cursor += protocol.writeSearchResult(w[cursor..], r.id, r.dist, name);
            }
            return cursor;
        }

        /// Write a snapshot to disk and, if the WAL is enabled, truncate
        /// it. The caller must hold exclusive access to the shared state
        /// — either via `self.rwlock.lock()` (live server) or by having
        /// joined the worker pool (shutdown path).
        ///
        /// Ordering guarantee: each writer fsyncs its own file, then we
        /// fsync the data directory so the dentries are durable, and
        /// only THEN do we truncate the WAL. A crash between snapshot
        /// durability and WAL truncate is safe: the snapshot is intact
        /// and the WAL still holds every record; replay on restart is
        /// idempotent for inserts whose ids are already present.
        pub fn snapshotNow(self: *Self) !void {
            std.fs.cwd().makePath(self.cfg.storage.data_dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
            var dir = try std.fs.cwd().openDir(self.cfg.storage.data_dir, .{});
            defer dir.close();

            try self.store.save(dir, self.cfg.storage.vectors_file);
            try self.index.save(dir, self.cfg.storage.graph_file);
            try self.metadata.save(
                self.allocator,
                dir,
                self.cfg.storage.metadata_file,
                self.store.count,
            );

            // Fsync the directory so the three files' rename-over
            // (createFile + write + close) is durable before we drop
            // WAL records that vouch for them.
            posix.fsync(dir.fd) catch |err| return err;

            if (self.wal) |wal| {
                wal.truncateAfterSnapshot() catch |err| {
                    log.warn("snapshot ok but WAL truncate failed: {s}", .{@errorName(err)});
                };
            }
        }
    };
}

fn writeErr(req: *InflightRequest, status: protocol.Status) void {
    const msg = protocol.statusMessage(status);
    req.status = status;
    req.response_len = protocol.encodeErrorPayload(req.response_buf, msg);
}

const testing = std.testing;

test "RequestPool acquires and releases in LIFO order" {
    var pool = try RequestPool.init(testing.allocator, 4);
    defer pool.deinit(testing.allocator);

    const a = pool.acquire().?;
    const b = pool.acquire().?;
    pool.release(a);
    const c = pool.acquire().?;
    try testing.expectEqual(a, c); // LIFO
    pool.release(b);
    pool.release(c);
    try testing.expectEqual(@as(usize, 4), pool.free_len);
}

test "RequestPool returns null when exhausted" {
    var pool = try RequestPool.init(testing.allocator, 2);
    defer pool.deinit(testing.allocator);
    _ = pool.acquire().?;
    _ = pool.acquire().?;
    try testing.expect(pool.acquire() == null);
}
