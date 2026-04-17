//! Async I/O dispatcher. Picks the per-OS backend at compile time.
//!
//! The shape is deliberately close to TigerBeetle's `src/io.zig` so that
//! callers work in terms of caller-owned `Completion` objects and typed
//! callbacks, not `poll(2)` state tables. The goal is a single primitive
//! (`run_for_ns`) that advances everything: accepted connections, readable
//! / writable sockets, timeouts, and cross-thread wakeups from the worker
//! pool.
//!
//! Only Darwin (kqueue) is implemented today. Linux (io_uring) is a TODO;
//! `hnswz` is developed on macOS and the server is the same one the
//! benchmarks exercise.

const builtin = @import("builtin");

pub const IO = switch (builtin.target.os.tag) {
    .macos, .ios, .tvos, .watchos => @import("io/darwin.zig").IO,
    else => @compileError("io: unsupported OS (only Darwin is implemented)"),
};

pub const Completion = IO.Completion;
pub const Result = IO.Result;
pub const Event = IO.Event;

test {
    _ = @import("io/darwin.zig");
}
