//! POSIX advisory file lock for preventing concurrent access to a data
//! directory. Uses `flock(2)` which:
//!
//!   * Works on Darwin and Linux (and other BSDs) with one API.
//!   * Is advisory: processes must opt in. A root/other-language process
//!     that doesn't call `flock` will not be stopped. For this project
//!     that is acceptable because all writers go through this module.
//!   * Is released automatically when the fd is closed or the process
//!     dies. Which means a stale lock file can never strand a data dir:
//!     even after a kill -9, the kernel drops the lock.
//!
//! A `LockFile` instance owns its underlying file handle. Call `release`
//! when done (or on server shutdown). `release` is idempotent.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

pub const LockError = error{
    /// Another process already holds the lock on this path.
    AlreadyLocked,
    /// The underlying filesystem does not support advisory locking
    /// (NFS without -o lock, some FUSE drivers, etc.).
    LockNotSupported,
    /// The lock file itself could not be opened/created.
    FileOpenFailed,
    /// Kernel-side resource exhaustion (too many locks system-wide).
    SystemResources,
    /// Anything else the syscall may have returned.
    LockFailed,
};

pub const LockFile = struct {
    file: std.fs.File,

    /// Acquire an exclusive advisory lock on `sub_path` within `dir`,
    /// creating the file if it does not exist. Non-blocking: if the lock
    /// is already held by another process, returns `error.AlreadyLocked`
    /// immediately.
    ///
    /// On success, the caller must call `release` to drop the lock.
    /// Dropping the LockFile via normal scope exit WITHOUT calling
    /// `release` leaks the fd (and thus the lock).
    pub fn acquire(dir: std.fs.Dir, sub_path: []const u8) LockError!LockFile {
        // .truncate = false so re-acquiring across restarts doesn't wipe
        // the PID line before we re-write it. .read = true in case a
        // future caller wants to read it for diagnostics.
        const file = dir.createFile(sub_path, .{
            .read = true,
            .truncate = false,
        }) catch return error.FileOpenFailed;
        errdefer file.close();

        posix.flock(file.handle, posix.LOCK.EX | posix.LOCK.NB) catch |err| switch (err) {
            error.WouldBlock => return error.AlreadyLocked,
            error.FileLocksNotSupported => return error.LockNotSupported,
            error.SystemResources => return error.SystemResources,
            error.Unexpected => return error.LockFailed,
        };

        // Best-effort: stamp the current pid for human diagnostics. If
        // this fails the lock is still valid — no correctness impact.
        writePid(file) catch {};

        return .{ .file = file };
    }

    /// Close the underlying fd, which implicitly releases the flock. Safe
    /// to call multiple times; subsequent calls are no-ops.
    pub fn release(self: *LockFile) void {
        if (self.file.handle < 0) return;
        self.file.close();
        self.file = .{ .handle = -1 };
    }

    fn writePid(file: std.fs.File) !void {
        const pid: i32 = switch (builtin.os.tag) {
            .linux => @intCast(std.os.linux.getpid()),
            .macos, .freebsd, .openbsd, .netbsd, .dragonfly => @intCast(std.c.getpid()),
            else => 0,
        };
        try file.setEndPos(0);
        var buf: [16]u8 = undefined;
        const slice = try std.fmt.bufPrint(&buf, "{d}\n", .{pid});
        try file.writeAll(slice);
    }
};

const testing = std.testing;

test "LockFile.acquire succeeds on a fresh file" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var lock = try LockFile.acquire(tmp.dir, "test.lock");
    defer lock.release();

    // File exists and contains a pid line.
    var buf: [32]u8 = undefined;
    const f = try tmp.dir.openFile("test.lock", .{});
    defer f.close();
    const n = try f.readAll(&buf);
    try testing.expect(n > 0);
}

test "LockFile.acquire fails when another holder is active" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var lock1 = try LockFile.acquire(tmp.dir, "contend.lock");
    defer lock1.release();

    const res = LockFile.acquire(tmp.dir, "contend.lock");
    try testing.expectError(error.AlreadyLocked, res);
}

test "LockFile.release allows re-acquire" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var lock1 = try LockFile.acquire(tmp.dir, "reuse.lock");
    lock1.release();

    var lock2 = try LockFile.acquire(tmp.dir, "reuse.lock");
    defer lock2.release();
}

test "LockFile.release is idempotent" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var lock = try LockFile.acquire(tmp.dir, "idem.lock");
    lock.release();
    lock.release(); // must not crash
}
