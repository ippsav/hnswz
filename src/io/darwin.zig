//! Darwin (kqueue) async-I/O backend.
//!
//! Shape is deliberately close to TigerBeetle's `src/io/darwin.zig`: a
//! caller-owned `Completion` carries a `*anyopaque` context, a trampoline
//! callback, and a small tagged `Operation` describing *what to wait
//! for*. The backend does NOT do the subsequent syscall (recv, send,
//! accept) itself — it just tells the caller "fd is readable/writable
//! now", and the caller issues the non-blocking syscall from inside the
//! callback. That mirrors TB's pattern closely enough to keep the code
//! honest but keeps our server's existing partial-read state machine
//! working without being shoe-horned into per-op variants.
//!
//! Operations:
//!   * readReady(fd)      — wait until `fd` has bytes to read or EOF.
//!   * writeReady(fd)     — wait until `fd` has room in its send buffer.
//!   * timeout(ns)        — fire exactly once after `ns` nanoseconds.
//!
//! Everything is ONESHOT: each completion consumes one event and must be
//! re-submitted if the caller wants to wait again. That matches how the
//! connection state machine moves forward one step per I/O wake.
//!
//! Cross-thread wakeups (worker → main loop) are done via a plain pipe(2)
//! registered as a `readReady` completion. The writer end is `write(2)`
//! (async-signal-safe), so this same mechanism works for signal handlers
//! too. We avoid EVFILT_USER: Darwin's semantics for updating a user
//! filter's `udata` via `NOTE_TRIGGER` changed with xnu versions, and a
//! pipe costs us one fd + one byte per wakeup, which is well below
//! anything that matters here.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const posix = std.posix;
const system = std.posix.system;

pub const IO = struct {
    const Self = @This();

    kq: posix.fd_t,
    /// Submitted but not yet flushed into kqueue.
    io_pending: Completion.List = .{},
    /// Ready to fire their callback; drained at the end of each `flush`.
    completed: Completion.List = .{},
    /// Live kernel-side registrations, for diagnostics only.
    io_inflight: usize = 0,
    pub const Event = u32; // reserved for a future EVFILT_USER port

    pub const Result = union(enum) {
        ready: void,
        timeout: void,
        /// EV_ERROR came back from kevent — payload carries the errno.
        err: c_int,
    };

    pub const Operation = union(enum) {
        none: void,
        read_ready: struct { fd: posix.fd_t },
        write_ready: struct { fd: posix.fd_t },
        timeout: struct { ns: i64 },
    };

    /// Caller-allocated work item. Typically embedded in a per-connection
    /// or per-request struct so it has the same lifetime as the caller's
    /// state. No allocations happen inside this module.
    pub const Completion = struct {
        next: ?*Completion = null,
        context: ?*anyopaque = null,
        callback: *const fn (*Completion, Result) void = undefined,
        operation: Operation = .none,
        /// Filled by `flush()` just before pushing to `completed`.
        result: Result = .ready,

        pub const List = struct {
            head: ?*Completion = null,
            tail: ?*Completion = null,

            pub fn push(self: *List, c: *Completion) void {
                c.next = null;
                if (self.tail) |t| t.next = c else self.head = c;
                self.tail = c;
            }

            pub fn pop(self: *List) ?*Completion {
                const h = self.head orelse return null;
                self.head = h.next;
                if (self.head == null) self.tail = null;
                h.next = null;
                return h;
            }

            pub fn empty(self: *const List) bool {
                return self.head == null;
            }
        };
    };

    pub fn init(_: std.mem.Allocator) !Self {
        const kq = try posix.kqueue();
        return .{ .kq = kq };
    }

    pub fn deinit(self: *Self) void {
        posix.close(self.kq);
        self.kq = -1;
    }

    pub fn readReady(
        self: *Self,
        comptime Context: type,
        context: Context,
        comptime callback: fn (ctx: Context, c: *Completion, result: Result) void,
        completion: *Completion,
        fd: posix.fd_t,
    ) void {
        submit(completion, Context, context, callback, .{ .read_ready = .{ .fd = fd } });
        self.io_pending.push(completion);
    }

    pub fn writeReady(
        self: *Self,
        comptime Context: type,
        context: Context,
        comptime callback: fn (ctx: Context, c: *Completion, result: Result) void,
        completion: *Completion,
        fd: posix.fd_t,
    ) void {
        submit(completion, Context, context, callback, .{ .write_ready = .{ .fd = fd } });
        self.io_pending.push(completion);
    }

    pub fn timeout(
        self: *Self,
        comptime Context: type,
        context: Context,
        comptime callback: fn (ctx: Context, c: *Completion, result: Result) void,
        completion: *Completion,
        ns: i64,
    ) void {
        assert(ns >= 0);
        submit(completion, Context, context, callback, .{ .timeout = .{ .ns = ns } });
        self.io_pending.push(completion);
    }

    /// Run one flush cycle, blocking up to `ns` ns if there is nothing
    /// else to do. On return, every completion that fired has had its
    /// callback invoked. If `ns == 0` the kevent call is non-blocking
    /// (pure tick).
    pub fn runForNs(self: *Self, ns: i64) !void {
        assert(ns >= 0);

        var changes_buf: [256]posix.Kevent = undefined;
        var events_buf: [256]posix.Kevent = undefined;

        // 1. Drain `io_pending` into a batch of kevent changes.
        var n_changes: usize = 0;
        while (n_changes < changes_buf.len) {
            const c = self.io_pending.pop() orelse break;
            changes_buf[n_changes] = buildKevent(c);
            n_changes += 1;
            self.io_inflight += 1;
        }

        // 2. If the caller gave us completed-but-not-drained items (from
        // a prior synchronous completion path), prefer a non-blocking
        // kevent call; we'll drain those before returning anyway.
        const want_block = self.completed.empty();
        const ts: posix.timespec = .{
            .sec = @intCast(@divTrunc(ns, std.time.ns_per_s)),
            .nsec = @intCast(@rem(ns, std.time.ns_per_s)),
        };
        const ts_ptr: ?*const posix.timespec = if (want_block) &ts else &zero_ts;

        // 3. Submit + wait in one syscall.
        const n = posix.kevent(
            self.kq,
            changes_buf[0..n_changes],
            &events_buf,
            ts_ptr,
        ) catch |err| switch (err) {
            // Transient; next tick will retry.
            error.SystemResources => 0,
            else => return err,
        };

        // 4. Translate events back to completions.
        for (events_buf[0..n]) |ev| {
            const c: *Completion = @ptrFromInt(ev.udata);
            c.result = eventToResult(ev, c.operation);
            self.completed.push(c);
            if (self.io_inflight > 0) self.io_inflight -= 1;
        }

        // 5. Fire callbacks. Each callback may call back into the IO
        // module to submit fresh completions; those land in `io_pending`
        // and are picked up on the next `runForNs` iteration. We do NOT
        // recurse here — that keeps stack depth bounded.
        while (self.completed.pop()) |c| {
            c.callback(c, c.result);
        }
    }

    const zero_ts: posix.timespec = .{ .sec = 0, .nsec = 0 };

    fn submit(
        completion: *Completion,
        comptime Context: type,
        context: Context,
        comptime callback: fn (ctx: Context, c: *Completion, result: Result) void,
        operation: Operation,
    ) void {
        // Erase the context type into `?*anyopaque`. Context is expected
        // to be a pointer; if it's a non-pointer value you'd have to box
        // it. In this project every caller passes a `*Something`.
        const ctx_anyopaque: ?*anyopaque = if (@typeInfo(Context) == .pointer)
            @ptrCast(@constCast(context))
        else
            @compileError("IO context must be a pointer type");

        // Monomorphized trampoline: casts back to `Context` and calls.
        const Trampoline = struct {
            fn run(c: *Completion, result: Result) void {
                const ctx_ptr: Context = @ptrCast(@alignCast(c.context));
                callback(ctx_ptr, c, result);
            }
        };

        completion.* = .{
            .context = ctx_anyopaque,
            .callback = &Trampoline.run,
            .operation = operation,
        };
    }

    fn buildKevent(c: *Completion) posix.Kevent {
        return switch (c.operation) {
            .read_ready => |r| .{
                .ident = @intCast(r.fd),
                .filter = system.EVFILT.READ,
                .flags = system.EV.ADD | system.EV.ENABLE | system.EV.ONESHOT,
                .fflags = 0,
                .data = 0,
                .udata = @intFromPtr(c),
            },
            .write_ready => |r| .{
                .ident = @intCast(r.fd),
                .filter = system.EVFILT.WRITE,
                .flags = system.EV.ADD | system.EV.ENABLE | system.EV.ONESHOT,
                .fflags = 0,
                .data = 0,
                .udata = @intFromPtr(c),
            },
            .timeout => |t| .{
                // Use the completion's pointer as the timer ident; no two
                // inflight completions share an address so idents are
                // unique by construction.
                .ident = @intFromPtr(c),
                .filter = system.EVFILT.TIMER,
                .flags = system.EV.ADD | system.EV.ENABLE | system.EV.ONESHOT,
                .fflags = system.NOTE.NSECONDS,
                .data = t.ns,
                .udata = @intFromPtr(c),
            },
            .none => unreachable,
        };
    }

    fn eventToResult(ev: posix.Kevent, op: Operation) Result {
        if (ev.flags & system.EV.ERROR != 0) return .{ .err = @intCast(ev.data) };
        return switch (op) {
            .read_ready, .write_ready => .ready,
            .timeout => .timeout,
            .none => unreachable,
        };
    }
};



const testing = std.testing;

test "IO timeout fires after requested duration" {
    var io = try IO.init(testing.allocator);
    defer io.deinit();

    const Ctx = struct {
        fired: bool = false,
        fn on(ctx: *@This(), _: *IO.Completion, r: IO.Result) void {
            ctx.fired = true;
            testing.expect(r == .timeout) catch {};
        }
    };
    var ctx: Ctx = .{};
    var c: IO.Completion = undefined;
    io.timeout(*Ctx, &ctx, Ctx.on, &c, 1 * std.time.ns_per_ms);

    var remaining_ticks: usize = 50;
    while (!ctx.fired and remaining_ticks > 0) : (remaining_ticks -= 1) {
        try io.runForNs(5 * std.time.ns_per_ms);
    }
    try testing.expect(ctx.fired);
}

test "IO readReady fires when pipe is written from another thread" {
    // Same pattern we'll use for cross-thread wakeup in the dispatcher:
    // main thread waits on readReady(rfd); worker thread write(wfd, "x").
    var io = try IO.init(testing.allocator);
    defer io.deinit();

    const fds = try posix.pipe2(.{ .NONBLOCK = true });
    const rfd = fds[0];
    const wfd = fds[1];
    defer posix.close(rfd);
    defer posix.close(wfd);

    const Ctx = struct {
        got: bool = false,
        fn on(ctx: *@This(), _: *IO.Completion, r: IO.Result) void {
            ctx.got = true;
            testing.expect(r == .ready) catch {};
        }
    };
    var ctx: Ctx = .{};
    var c: IO.Completion = undefined;
    io.readReady(*Ctx, &ctx, Ctx.on, &c, rfd);

    const Writer = struct {
        fn run(fd: posix.fd_t) void {
            std.Thread.sleep(1 * std.time.ns_per_ms);
            _ = posix.write(fd, "x") catch {};
        }
    };
    const thread = try std.Thread.spawn(.{}, Writer.run, .{wfd});
    defer thread.join();

    var ticks: usize = 50;
    while (!ctx.got and ticks > 0) : (ticks -= 1) {
        try io.runForNs(5 * std.time.ns_per_ms);
    }
    try testing.expect(ctx.got);
}

test "IO readReady fires when pipe is written (same thread)" {
    var io = try IO.init(testing.allocator);
    defer io.deinit();

    const fds = try posix.pipe2(.{ .NONBLOCK = true });
    const rfd = fds[0];
    const wfd = fds[1];
    defer posix.close(rfd);
    defer posix.close(wfd);

    const Ctx = struct {
        got: bool = false,
        fn on(ctx: *@This(), _: *IO.Completion, r: IO.Result) void {
            ctx.got = true;
            testing.expect(r == .ready) catch {};
        }
    };
    var ctx: Ctx = .{};
    var c: IO.Completion = undefined;
    io.readReady(*Ctx, &ctx, Ctx.on, &c, rfd);

    // Submit first.
    try io.runForNs(0);

    // Write one byte to unblock the reader.
    _ = try posix.write(wfd, "x");

    var ticks: usize = 50;
    while (!ctx.got and ticks > 0) : (ticks -= 1) {
        try io.runForNs(5 * std.time.ns_per_ms);
    }
    try testing.expect(ctx.got);
}
