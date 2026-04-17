//! Write-Ahead Log for the hnswz vector store.
//!
//! ## What it guarantees
//!
//! Every mutation the server acknowledges is durable on disk before the
//! ack goes out. If the process dies for any reason (SIGKILL, OOM,
//! sudden power loss that the OS survives), on restart we replay the WAL
//! and recover all acknowledged writes. Writes that crashed mid-ack are
//! not re-applied — the client will see that as a failed request and can
//! retry.
//!
//! ## Why we WAL the resolved vector, not the text
//!
//! `insert_text` and `replace_text` both invoke an external embedder
//! that can change model, dimension, or simply be unreachable on
//! restart. Persisting the post-embed `[]f32` means replay is
//! deterministic and never needs to hit a network service. The original
//! text is stored in `name_bytes` purely for metadata (so the name -> id
//! mapping survives a restart).
//!
//! ## Record format
//!
//!    header (20 bytes, written once at file create):
//!      magic      [4]u8   "HWAL"
//!      version    u32     1
//!      dim        u32     (must match runtime config.embedder.dim)
//!      reserved   u32     0  (future flags)
//!      hdr_crc    u32     CRC32 over the first 16 bytes
//!
//!    record (repeated, append-only):
//!      length     u32     byte length of body (seq..name_bytes)
//!      body       <length bytes>:
//!        seq       u64    monotonic per-file sequence number
//!        opcode    u8     1=insert, 2=delete, 3=replace
//!        flags     u16    reserved 0
//!        id        u32    store id the record refers to
//!        level     u8     HNSW level for insert/replace; 0 for delete
//!        vec_len   u16    f32 element count (0 for delete; == dim otherwise)
//!        name_len  u16    byte length of name (may be 0)
//!        vec_bytes <vec_len * 4 bytes>
//!        name_bytes <name_len bytes>
//!      crc        u32     CRC32 over [length bytes || body bytes]
//!
//! The CRC covers the length prefix so a torn write at any byte boundary
//! produces a mismatch and stops replay; we then truncate the WAL at
//! the last fully-valid record.
//!
//! ## Snapshot interaction
//!
//! When a snapshot succeeds, the caller invokes `truncateAfterSnapshot`.
//! That atomically replaces the WAL with a header-only copy: we write a
//! fresh header to `wal.hwal.tmp`, fsync it, and rename over the live
//! file. Replay skips nothing — the file simply contains no records.
//!
//! If the truncate itself crashes partway through, the old WAL is still
//! on disk intact (rename(2) is atomic), so replay replays every record
//! — each insert is already in the snapshot and is detected as a
//! duplicate by the replay logic (see `applyInsert`).

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

const Store = @import("store.zig").Store;
const HnswIndexFn = @import("hnsw.zig").HnswIndex;
const MutableMetadata = @import("metadata_mut.zig").MutableMetadata;

pub const MAGIC: [4]u8 = .{ 'H', 'W', 'A', 'L' };
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: u32 = 20;

/// Hard cap on a single record's body length. Anything bigger than this
/// is interpreted as a torn/garbled write and terminates replay. At
/// dim=4096 (the project's documented ceiling) an insert record is
/// ~16.4 KiB; 1 MiB leaves orders of magnitude of slack while still
/// being small enough that a corrupt 4 GiB length is rejected instantly.
pub const MAX_RECORD_BYTES: u32 = 1 << 20;

pub const Opcode = enum(u8) {
    insert = 1,
    delete = 2,
    replace = 3,
};

pub const InsertEntry = struct {
    id: u32,
    level: u8,
    flags: u16 = 0,
    vec: []const f32,
    name: []const u8 = "",
};

pub const DeleteEntry = struct {
    id: u32,
    flags: u16 = 0,
};

pub const ReplaceEntry = struct {
    id: u32,
    level: u8,
    flags: u16 = 0,
    vec: []const f32,
    /// Empty string ⇒ name unchanged. Callers that want to explicitly
    /// clear a name should pass a 0-length slice with `flags` bit 0 set;
    /// for now we carry the same semantics as the server has today
    /// (replace_vec never touches metadata; replace_text overwrites it).
    name: ?[]const u8 = null,
};

pub const OpenResult = struct {
    wal: Wal,
    /// Number of records successfully replayed. Useful for logging.
    replayed: usize,
    /// True when replay stopped early because a record was torn or
    /// corrupt. The WAL file has been truncated to the last valid
    /// record. Callers that care (tests) can assert on this.
    truncated: bool,
};

pub const OpenError = error{
    /// The data_dir could not be opened or the WAL file could not be
    /// created/opened.
    FileOpenFailed,
    /// Header read short, bad magic, unknown version, or header CRC
    /// mismatch — anything that says "this isn't our WAL or we can't
    /// safely replay it".
    CorruptHeader,
    /// The dim recorded in the WAL header disagrees with the dim the
    /// server expects. Misconfigured restart; caller should surface a
    /// clear error and refuse to start.
    DimMismatch,
    /// Replay successfully read records but one of them failed to apply
    /// to the in-memory state (e.g. referenced an id past capacity).
    /// Usually indicates config changes (e.g. max_vectors shrunk) and
    /// needs human intervention.
    ReplayFailed,
    /// I/O error while reading/writing the WAL.
    IoFailed,
    /// Out of memory during replay (we allocate a scratch vec buffer).
    OutOfMemory,
};

pub const AppendError = error{
    IoFailed,
    /// vec.len or name.len doesn't fit the on-disk encoding.
    EntryTooLarge,
};

/// A Write-Ahead Log handle. Not thread-safe — callers must serialize
/// append/truncate calls (the server does so implicitly via its
/// write-path RwLock, same lock that covers store/index/metadata).
pub const Wal = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    /// Parent data_dir. Owned by the caller (server/main); we do not
    /// close it here. Held for `truncateAfterSnapshot`'s atomic rename.
    dir: std.fs.Dir,
    /// The live WAL file. Appends write here. Ownership passes to the
    /// WAL on open.
    file: std.fs.File,
    /// Configured dim, validated against header on open and against
    /// every append payload.
    dim: u32,
    /// Next sequence number to emit. After replay this is set to
    /// `max_seq_in_file + 1` (or 1 if the file is empty).
    next_seq: u64,
    /// Copy of the file path, used for atomic truncate via a sibling
    /// `.tmp` file + rename.
    sub_path: []const u8,

    /// Open or create the WAL file at `sub_path` inside `dir`. Replays
    /// any records present, applying them to `store`, `index`, and
    /// `metadata`. Returns the open handle plus replay stats.
    ///
    /// `index` must be the generic `HnswIndex(M)` instance the server
    /// is about to run. Its type is inferred from the argument so this
    /// function itself doesn't have to be generic.
    ///
    /// On corruption or partial-record-at-EOF, replay stops early and
    /// the file is truncated to the last fully-valid record. This is
    /// the expected case after a crash; no error is returned, only
    /// `truncated = true`.
    pub fn open(
        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        sub_path: []const u8,
        dim: u32,
        store: *Store,
        index: anytype,
        metadata: *MutableMetadata,
    ) OpenError!OpenResult {
        var file = openOrCreateWithHeader(dir, sub_path, dim) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.CorruptHeader => return error.CorruptHeader,
            error.DimMismatch => return error.DimMismatch,
            else => return error.FileOpenFailed,
        };
        errdefer file.close();

        const replay = replayAll(allocator, file, dim, store, index, metadata) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ReplayFailed => return error.ReplayFailed,
            else => return error.IoFailed,
        };

        // Seek to the end of valid records for future appends.
        file.seekTo(replay.valid_end_offset) catch return error.IoFailed;
        if (replay.truncated) {
            file.setEndPos(replay.valid_end_offset) catch return error.IoFailed;
            file.sync() catch return error.IoFailed;
        }

        return .{
            .wal = .{
                .allocator = allocator,
                .dir = dir,
                .file = file,
                .dim = dim,
                .next_seq = replay.last_seq + 1,
                .sub_path = sub_path,
            },
            .replayed = replay.applied,
            .truncated = replay.truncated,
        };
    }

    pub fn close(self: *Self) void {
        self.file.close();
    }

    /// Append an insert record, fsync, and update `next_seq`. Callers
    /// must have already validated that `vec.len == dim`.
    pub fn appendInsert(self: *Self, e: InsertEntry) AppendError!void {
        std.debug.assert(e.vec.len == self.dim);
        if (e.name.len > std.math.maxInt(u16)) return error.EntryTooLarge;
        try self.appendRecord(.insert, e.flags, e.id, e.level, e.vec, e.name);
    }

    pub fn appendDelete(self: *Self, e: DeleteEntry) AppendError!void {
        try self.appendRecord(.delete, e.flags, e.id, 0, &.{}, "");
    }

    pub fn appendReplace(self: *Self, e: ReplaceEntry) AppendError!void {
        std.debug.assert(e.vec.len == self.dim);
        const name = e.name orelse "";
        if (name.len > std.math.maxInt(u16)) return error.EntryTooLarge;
        try self.appendRecord(.replace, e.flags, e.id, e.level, e.vec, name);
    }

    /// Atomically rewrite the WAL as a header-only file, discarding all
    /// existing records. Must be called ONLY after the paired snapshot
    /// files have been written AND fsync'd — the WAL's contents are the
    /// recovery material for anything not yet in the snapshot.
    ///
    /// Procedure: write `{sub_path}.tmp` containing just a fresh header,
    /// fsync it, rename it over the live WAL, fsync the directory so the
    /// rename itself is durable, then swap our file handle to the new
    /// file.
    pub fn truncateAfterSnapshot(self: *Self) !void {
        const tmp_name_buf_len = 64;
        var tmp_buf: [tmp_name_buf_len]u8 = undefined;
        const tmp_name = try std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{self.sub_path});

        // Write fresh header to tmp.
        {
            const tmp = self.dir.createFile(tmp_name, .{ .read = true, .truncate = true }) catch return error.IoFailed;
            defer tmp.close();
            var hdr: [HEADER_SIZE]u8 = undefined;
            writeHeader(&hdr, self.dim);
            tmp.writeAll(&hdr) catch return error.IoFailed;
            tmp.sync() catch return error.IoFailed;
        }

        // Atomic rename. On Darwin/Linux this is guaranteed to either
        // succeed in full or leave the original file untouched.
        self.dir.rename(tmp_name, self.sub_path) catch return error.IoFailed;

        // Fsync the directory so the rename survives a crash. The Dir
        // struct exposes `fd` directly on POSIX; call posix.fsync on it.
        // std.fs.Dir has no .sync method in 0.15.2.
        posix.fsync(self.dir.fd) catch return error.IoFailed;

        // Swap the file handle: close the stale one, reopen the new.
        self.file.close();
        self.file = self.dir.openFile(self.sub_path, .{ .mode = .read_write }) catch return error.IoFailed;
        self.file.seekFromEnd(0) catch return error.IoFailed;

        // Sequence numbers continue to grow across truncates — any record
        // already applied to the snapshot is gone from disk, so there's
        // no risk of collision. Keeping them monotonic helps debugging.
    }

    // ---- internals ----

    fn appendRecord(
        self: *Self,
        opcode: Opcode,
        flags: u16,
        id: u32,
        level: u8,
        vec: []const f32,
        name: []const u8,
    ) AppendError!void {
        const vec_bytes = std.mem.sliceAsBytes(vec);
        // seq u64 | op u8 | flags u16 | id u32 | level u8 | vec_len u16 | name_len u16 | vec | name
        const body_len: u64 = 8 + 1 + 2 + 4 + 1 + 2 + 2 + vec_bytes.len + name.len;
        if (body_len > MAX_RECORD_BYTES) return error.EntryTooLarge;

        var head: [4 + 8 + 1 + 2 + 4 + 1 + 2 + 2]u8 = undefined;
        std.mem.writeInt(u32, head[0..4], @intCast(body_len), .little);
        std.mem.writeInt(u64, head[4..12], self.next_seq, .little);
        head[12] = @intFromEnum(opcode);
        std.mem.writeInt(u16, head[13..15], flags, .little);
        std.mem.writeInt(u32, head[15..19], id, .little);
        head[19] = level;
        std.mem.writeInt(u16, head[20..22], @intCast(vec.len), .little);
        std.mem.writeInt(u16, head[22..24], @intCast(name.len), .little);

        var crc = std.hash.Crc32.init();
        crc.update(head[0..]);
        if (vec_bytes.len > 0) crc.update(vec_bytes);
        if (name.len > 0) crc.update(name);
        const crc_final = crc.final();

        var tail: [4]u8 = undefined;
        std.mem.writeInt(u32, tail[0..4], crc_final, .little);

        // Write in three chunks (head/vec/name/tail). For typical insert
        // sizes (dim=4096, name~256) the entire record is ≤17 KiB so
        // this is two or three syscalls — acceptable given we fsync
        // anyway.
        self.file.writeAll(head[0..]) catch return error.IoFailed;
        if (vec_bytes.len > 0) self.file.writeAll(vec_bytes) catch return error.IoFailed;
        if (name.len > 0) self.file.writeAll(name) catch return error.IoFailed;
        self.file.writeAll(tail[0..]) catch return error.IoFailed;
        self.file.sync() catch return error.IoFailed;

        self.next_seq += 1;
    }

    const OpenWithHeaderError = error{
        OutOfMemory,
        CorruptHeader,
        DimMismatch,
        IoFailed,
    };

    fn openOrCreateWithHeader(dir: std.fs.Dir, sub_path: []const u8, dim: u32) OpenWithHeaderError!std.fs.File {
        // First try to open for read+write — existing WAL path.
        if (dir.openFile(sub_path, .{ .mode = .read_write })) |file| {
            errdefer file.close();
            const size = file.getEndPos() catch return error.IoFailed;
            if (size == 0) {
                // Zero-length file: treat like a fresh create.
                try writeFreshHeader(file, dim);
                return file;
            }
            try validateExistingHeader(file, dim);
            return file;
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return error.IoFailed,
        }

        // File doesn't exist — create it with a fresh header.
        const file = dir.createFile(sub_path, .{ .read = true, .truncate = true }) catch return error.IoFailed;
        errdefer file.close();
        try writeFreshHeader(file, dim);
        return file;
    }

    fn writeFreshHeader(file: std.fs.File, dim: u32) OpenWithHeaderError!void {
        var hdr: [HEADER_SIZE]u8 = undefined;
        writeHeader(&hdr, dim);
        file.writeAll(&hdr) catch return error.IoFailed;
        file.sync() catch return error.IoFailed;
    }

    fn validateExistingHeader(file: std.fs.File, dim: u32) OpenWithHeaderError!void {
        file.seekTo(0) catch return error.IoFailed;
        var hdr: [HEADER_SIZE]u8 = undefined;
        const n = file.readAll(&hdr) catch return error.IoFailed;
        if (n != HEADER_SIZE) return error.CorruptHeader;
        if (!std.mem.eql(u8, hdr[0..4], &MAGIC)) return error.CorruptHeader;
        const ver = std.mem.readInt(u32, hdr[4..8], .little);
        if (ver != VERSION) return error.CorruptHeader;
        const file_dim = std.mem.readInt(u32, hdr[8..12], .little);
        if (file_dim != dim) return error.DimMismatch;
        const expected = std.hash.Crc32.hash(hdr[0..16]);
        const actual = std.mem.readInt(u32, hdr[16..20], .little);
        if (expected != actual) return error.CorruptHeader;
    }

    const ReplaySummary = struct {
        applied: usize,
        last_seq: u64,
        valid_end_offset: u64,
        truncated: bool,
    };

    const ReplayError = error{
        OutOfMemory,
        ReplayFailed,
        IoFailed,
    };

    fn replayAll(
        allocator: std.mem.Allocator,
        file: std.fs.File,
        dim: u32,
        store: *Store,
        index: anytype,
        metadata: *MutableMetadata,
    ) ReplayError!ReplaySummary {
        // Allocate one vec buffer and reuse for every insert/replace.
        const vec_buf = allocator.alloc(f32, dim) catch return error.OutOfMemory;
        defer allocator.free(vec_buf);

        // One workspace for the duration of replay — HNSW insert/delete
        // reuse it under the single-threaded replay path. Sizing mirrors
        // the live server's dispatcher: max_vectors for `visited` width,
        // ef_construction for the results heap.
        const IndexType = @TypeOf(index.*);
        var ws = IndexType.Workspace.init(
            allocator,
            index.max_vectors,
            index.ef_construction,
        ) catch return error.OutOfMemory;
        defer ws.deinit(allocator);

        // Iterate records from just past the header.
        file.seekTo(HEADER_SIZE) catch return error.IoFailed;
        var applied: usize = 0;
        var last_seq: u64 = 0;
        var valid_end: u64 = HEADER_SIZE;
        var truncated = false;

        while (true) {
            const rec = readOneRecord(allocator, file, vec_buf, dim) catch |err| switch (err) {
                error.EndOfStream, error.CorruptRecord => {
                    truncated = switch (err) {
                        error.CorruptRecord => true,
                        else => false,
                    };
                    break;
                },
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.IoFailed,
            };

            applyRecord(allocator, rec, store, index, metadata, &ws) catch return error.ReplayFailed;

            applied += 1;
            last_seq = rec.seq;
            valid_end = file.getPos() catch return error.IoFailed;
        }

        return .{
            .applied = applied,
            .last_seq = last_seq,
            .valid_end_offset = valid_end,
            .truncated = truncated,
        };
    }

    /// In-memory record shape used by `readOneRecord` → `applyRecord`.
    /// `vec` points into the caller-supplied scratch buffer; `name`
    /// points into a per-record allocation released after apply.
    const Record = struct {
        seq: u64,
        opcode: Opcode,
        flags: u16,
        id: u32,
        level: u8,
        vec: []const f32, // len == 0 for delete
        name: []const u8,
        name_owned: ?[]u8, // name allocation to free after apply
    };

    const ReadError = error{
        EndOfStream,
        CorruptRecord,
        OutOfMemory,
        IoFailed,
    };

    fn readOneRecord(allocator: std.mem.Allocator, file: std.fs.File, vec_buf: []f32, dim: u32) ReadError!Record {
        var len_buf: [4]u8 = undefined;
        const ln = file.readAll(&len_buf) catch return error.IoFailed;
        if (ln == 0) return error.EndOfStream;
        if (ln < 4) return error.CorruptRecord;
        const length = std.mem.readInt(u32, &len_buf, .little);
        if (length < 8 + 1 + 2 + 4 + 1 + 2 + 2) return error.CorruptRecord;
        if (length > MAX_RECORD_BYTES) return error.CorruptRecord;

        // Peek at the fixed prefix of the body so we know vec_len/name_len
        // before allocating.
        var prefix: [8 + 1 + 2 + 4 + 1 + 2 + 2]u8 = undefined;
        const pn = file.readAll(&prefix) catch return error.IoFailed;
        if (pn != prefix.len) return error.CorruptRecord;

        const seq = std.mem.readInt(u64, prefix[0..8], .little);
        const opcode_raw = prefix[8];
        const flags = std.mem.readInt(u16, prefix[9..11], .little);
        const id = std.mem.readInt(u32, prefix[11..15], .little);
        const level = prefix[15];
        const vec_len = std.mem.readInt(u16, prefix[16..18], .little);
        const name_len = std.mem.readInt(u16, prefix[18..20], .little);

        const body_rest: u64 = @as(u64, length) - @as(u64, prefix.len);
        const vec_bytes: u64 = @as(u64, vec_len) * 4;
        if (vec_bytes + name_len != body_rest) return error.CorruptRecord;

        // Validate shape against opcode semantics.
        const opcode = std.meta.intToEnum(Opcode, opcode_raw) catch return error.CorruptRecord;
        switch (opcode) {
            .insert, .replace => {
                if (vec_len != dim) return error.CorruptRecord;
            },
            .delete => {
                if (vec_len != 0) return error.CorruptRecord;
            },
        }

        // Read vec into caller's buffer.
        if (vec_bytes > 0) {
            const vb = std.mem.sliceAsBytes(vec_buf[0..vec_len]);
            const vn = file.readAll(vb) catch return error.IoFailed;
            if (vn != vb.len) return error.CorruptRecord;
        }

        // Allocate+read name using the WAL's allocator — same allocator
        // that the apply path passes to metadata.setAt / deleteAt, which
        // keeps allocator ownership consistent.
        var name_slice: []u8 = &.{};
        var name_owned: ?[]u8 = null;
        if (name_len > 0) {
            const name_alloc = allocator.alloc(u8, name_len) catch return error.OutOfMemory;
            errdefer allocator.free(name_alloc);
            const nn = file.readAll(name_alloc) catch return error.IoFailed;
            if (nn != name_alloc.len) return error.CorruptRecord;
            name_slice = name_alloc;
            name_owned = name_alloc;
        }

        // Verify CRC.
        var crc_buf: [4]u8 = undefined;
        const cn = file.readAll(&crc_buf) catch return error.IoFailed;
        if (cn != 4) return error.CorruptRecord;
        const file_crc = std.mem.readInt(u32, &crc_buf, .little);

        var hasher = std.hash.Crc32.init();
        hasher.update(len_buf[0..]);
        hasher.update(prefix[0..]);
        if (vec_bytes > 0) hasher.update(std.mem.sliceAsBytes(vec_buf[0..vec_len]));
        if (name_len > 0) hasher.update(name_slice);
        if (hasher.final() != file_crc) {
            if (name_owned) |n| allocator.free(n);
            return error.CorruptRecord;
        }

        return .{
            .seq = seq,
            .opcode = opcode,
            .flags = flags,
            .id = id,
            .level = level,
            .vec = if (vec_len > 0) vec_buf[0..vec_len] else &.{},
            .name = name_slice,
            .name_owned = name_owned,
        };
    }

    fn applyRecord(
        allocator: std.mem.Allocator,
        rec: Record,
        store: *Store,
        index: anytype,
        metadata: *MutableMetadata,
        ws: anytype,
    ) !void {
        defer if (rec.name_owned) |n| allocator.free(n);

        switch (rec.opcode) {
            .insert => try applyInsert(allocator, rec, store, index, metadata, ws),
            .delete => try applyDelete(allocator, rec, index, metadata, ws),
            .replace => try applyReplace(allocator, rec, index, metadata, ws),
        }
    }

    fn applyInsert(
        allocator: std.mem.Allocator,
        rec: Record,
        store: *Store,
        index: anytype,
        metadata: *MutableMetadata,
        ws: anytype,
    ) !void {
        // Idempotent apply: if this id is already live in the store, the
        // record predates a snapshot that already captured it. This
        // happens when a snapshot succeeded but the subsequent WAL
        // truncate crashed (the rename(2) is atomic but the process can
        // die between "snapshot files written" and "rename wal.tmp ->
        // wal"). Skip.
        if (@as(usize, rec.id) < store.count and !store.isDeleted(rec.id)) return;

        if (@as(usize, rec.id) < store.count and store.isDeleted(rec.id)) {
            // This record is re-inserting into a tombstoned slot. Use
            // the addAt path which preserves the id.
            try store.addAt(rec.id, rec.vec);
        } else {
            // Fresh slot. Store assigns ids in the same deterministic
            // order as the original run, so `store.add` will return
            // exactly `rec.id`. Mismatch here means the snapshot and
            // WAL disagree about history — unrecoverable.
            const assigned = try store.add(rec.vec);
            if (assigned != rec.id) return error.ReplayDivergent;
        }
        try index.insertWithLevel(ws, rec.id, rec.level);
        try metadata.setAt(allocator, rec.id, rec.name);
    }

    fn applyDelete(
        allocator: std.mem.Allocator,
        rec: Record,
        index: anytype,
        metadata: *MutableMetadata,
        ws: anytype,
    ) !void {
        // Idempotent — index.delete is a no-op if already deleted.
        try index.delete(ws, rec.id);
        metadata.deleteAt(allocator, rec.id);
    }

    fn applyReplace(
        allocator: std.mem.Allocator,
        rec: Record,
        index: anytype,
        metadata: *MutableMetadata,
        ws: anytype,
    ) !void {
        try index.replaceVectorWithLevel(ws, rec.id, rec.vec, rec.level);
        if (rec.name.len > 0) try metadata.setAt(allocator, rec.id, rec.name);
    }

    fn writeHeader(hdr: []u8, dim: u32) void {
        std.debug.assert(hdr.len == HEADER_SIZE);
        @memcpy(hdr[0..4], &MAGIC);
        std.mem.writeInt(u32, hdr[4..8], VERSION, .little);
        std.mem.writeInt(u32, hdr[8..12], dim, .little);
        std.mem.writeInt(u32, hdr[12..16], 0, .little);
        const crc = std.hash.Crc32.hash(hdr[0..16]);
        std.mem.writeInt(u32, hdr[16..20], crc, .little);
    }
};

// --- tests ------------------------------------------------------------

const testing = std.testing;

test "writeHeader + validateExistingHeader round-trip" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("w.wal", .{ .read = true, .truncate = true });
    defer file.close();

    var hdr: [HEADER_SIZE]u8 = undefined;
    Wal.writeHeader(&hdr, 128);
    try file.writeAll(&hdr);
    try file.sync();

    try Wal.validateExistingHeader(file, 128);
    try testing.expectError(error.DimMismatch, Wal.validateExistingHeader(file, 129));
}

test "openOrCreateWithHeader creates a fresh file" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try Wal.openOrCreateWithHeader(tmp.dir, "fresh.wal", 32);
    defer file.close();

    const sz = try file.getEndPos();
    try testing.expectEqual(@as(u64, HEADER_SIZE), sz);
}

test "Wal.open then append + replay round-trip" {
    const Index = HnswIndexFn(8);
    const dim: u32 = 4;
    const cap: usize = 32;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // --- phase 1: write a WAL with a couple records, close ---
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer index.deinit();
        var md = MutableMetadata.init();
        defer md.deinit(testing.allocator);

        var res = try Wal.open(testing.allocator, tmp.dir, "a.wal", dim, &store, &index, &md);
        defer res.wal.close();
        try testing.expectEqual(@as(usize, 0), res.replayed);

        const v0 = [_]f32{ 1, 0, 0, 0 };
        const v1 = [_]f32{ 0, 1, 0, 0 };
        try res.wal.appendInsert(.{ .id = 0, .level = 0, .vec = &v0, .name = "first" });
        try res.wal.appendInsert(.{ .id = 1, .level = 0, .vec = &v1, .name = "second" });
        try res.wal.appendDelete(.{ .id = 1 });
    }

    // --- phase 2: open a fresh set of in-memory structures, replay ---
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer index.deinit();
        var md = MutableMetadata.init();
        defer md.deinit(testing.allocator);

        var res = try Wal.open(testing.allocator, tmp.dir, "a.wal", dim, &store, &index, &md);
        defer res.wal.close();

        try testing.expectEqual(@as(usize, 3), res.replayed);
        try testing.expectEqual(false, res.truncated);
        // id 0 alive, id 1 deleted.
        try testing.expectEqual(@as(usize, 1), store.live_count);
        try testing.expect(!store.isDeleted(0));
        try testing.expect(store.isDeleted(1));
        try testing.expectEqualStrings("first", md.get(0).?);
    }
}

test "replay stops at corrupt record and truncates file" {
    const Index = HnswIndexFn(8);
    const dim: u32 = 2;
    const cap: usize = 8;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // --- phase 1: write a valid record, then append garbage ---
    var valid_end: u64 = 0;
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer index.deinit();
        var md = MutableMetadata.init();
        defer md.deinit(testing.allocator);

        var res = try Wal.open(testing.allocator, tmp.dir, "bad.wal", dim, &store, &index, &md);
        const v0 = [_]f32{ 1, 0 };
        try res.wal.appendInsert(.{ .id = 0, .level = 0, .vec = &v0 });
        valid_end = try res.wal.file.getPos();
        res.wal.close();

        // Append raw garbage to simulate a torn write.
        const f = try tmp.dir.openFile("bad.wal", .{ .mode = .read_write });
        defer f.close();
        try f.seekTo(valid_end);
        try f.writeAll(&[_]u8{ 0xff, 0xff, 0xff, 0x7f, 0xaa, 0xbb }); // absurd length + some body
    }

    // --- phase 2: open, replay, verify truncated ---
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer index.deinit();
        var md = MutableMetadata.init();
        defer md.deinit(testing.allocator);

        var res = try Wal.open(testing.allocator, tmp.dir, "bad.wal", dim, &store, &index, &md);
        defer res.wal.close();

        try testing.expectEqual(@as(usize, 1), res.replayed);
        try testing.expectEqual(true, res.truncated);
        const sz = try res.wal.file.getEndPos();
        try testing.expectEqual(valid_end, sz);
    }
}

test "truncateAfterSnapshot wipes records and preserves header" {
    const Index = HnswIndexFn(8);
    const dim: u32 = 2;
    const cap: usize = 8;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var store = try Store.init(testing.allocator, dim, cap);
    defer store.deinit(testing.allocator);
    var index = try Index.init(testing.allocator, &store, .{
        .max_vectors = cap,
        .max_upper_slots = cap,
        .ef_construction = 16,
        .seed = 1,
    });
    defer index.deinit();
    var md = MutableMetadata.init();
    defer md.deinit(testing.allocator);

    var res = try Wal.open(testing.allocator, tmp.dir, "t.wal", dim, &store, &index, &md);
    defer res.wal.close();

    const v0 = [_]f32{ 1, 0 };
    try res.wal.appendInsert(.{ .id = 0, .level = 0, .vec = &v0 });
    try res.wal.appendInsert(.{ .id = 1, .level = 0, .vec = &v0 });
    try testing.expect((try res.wal.file.getEndPos()) > HEADER_SIZE);

    try res.wal.truncateAfterSnapshot();
    try testing.expectEqual(@as(u64, HEADER_SIZE), try res.wal.file.getEndPos());

    // New appends should still work after truncate.
    try res.wal.appendInsert(.{ .id = 2, .level = 0, .vec = &v0 });
    try testing.expect((try res.wal.file.getEndPos()) > HEADER_SIZE);
}

// Helper: emulate dispatcher.snapshotNow outside a full server — writes all
// three files, fsyncs the dir, then truncates the WAL. Kept adjacent to the
// tests so they read like what the server actually does.
fn snapshotAll(
    allocator: std.mem.Allocator,
    dir: std.fs.Dir,
    store: anytype,
    index: anytype,
    md: *MutableMetadata,
    wal: ?*Wal,
) !void {
    try store.save(dir, "vectors.hvsf");
    try index.save(dir, "graph.hgrf");
    try md.save(allocator, dir, "metadata.hmtf", store.count);
    try posix.fsync(dir.fd);
    if (wal) |w| try w.truncateAfterSnapshot();
}

// Regression for the id-space divergence bug: a post-snapshot insert that
// reuses a freed id must survive a kill-9 + replay. Before the fix the
// snapshot compacted ids (old id 1 → new id 0) while the live server kept
// using the pre-snapshot free list, so the WAL record emitted after the
// snapshot referred to id 0 in the OLD id space — and replay silently
// skipped it as "already in snapshot" because id 0 in the NEW (loaded)
// id space also happened to be live.
test "post-snapshot insert into reused slot survives crash + replay" {
    const Index = HnswIndexFn(8);
    const dim: u32 = 2;
    const cap: usize = 8;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const v_a = [_]f32{ 1, 0 };
    const v_b = [_]f32{ 0, 1 };
    const v_c = [_]f32{ 0.5, 0.5 };

    // --- phase 1: insert A, insert B, delete 0, snapshot, insert C, crash ---
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer index.deinit();
        var md = MutableMetadata.init();
        defer md.deinit(testing.allocator);
        var ws = try Index.Workspace.init(testing.allocator, cap, 16);
        defer ws.deinit(testing.allocator);

        var res = try Wal.open(testing.allocator, tmp.dir, "w.wal", dim, &store, &index, &md);
        defer res.wal.close();

        // insert A
        try res.wal.appendInsert(.{ .id = 0, .level = 0, .vec = &v_a, .name = "A" });
        const id_a = try store.add(&v_a);
        try testing.expectEqual(@as(u32, 0), id_a);
        try index.insertWithLevel(&ws, id_a, 0);
        try md.setAt(testing.allocator, id_a, "A");

        // insert B
        try res.wal.appendInsert(.{ .id = 1, .level = 0, .vec = &v_b, .name = "B" });
        const id_b = try store.add(&v_b);
        try testing.expectEqual(@as(u32, 1), id_b);
        try index.insertWithLevel(&ws, id_b, 0);
        try md.setAt(testing.allocator, id_b, "B");

        // delete 0
        try res.wal.appendDelete(.{ .id = 0 });
        try index.delete(&ws, 0);
        md.deleteAt(testing.allocator, 0);

        // snapshot (store has count=2, live_count=1, free_list=[0])
        try snapshotAll(testing.allocator, tmp.dir, &store, &index, &md, &res.wal);

        // insert C — store.add reuses id 0 from the free list, since in-memory
        // state is not re-compacted by snapshot.
        const reserved = store.peekNextId().?;
        try testing.expectEqual(@as(u32, 0), reserved);
        try res.wal.appendInsert(.{ .id = reserved, .level = 0, .vec = &v_c, .name = "C" });
        const id_c = try store.add(&v_c);
        try testing.expectEqual(@as(u32, 0), id_c);
        try index.insertWithLevel(&ws, id_c, 0);
        try md.setAt(testing.allocator, id_c, "C");

        try testing.expectEqual(@as(usize, 2), store.live_count);
        // ... abrupt crash (no clean shutdown) ...
    }

    // --- phase 2: reopen, expect both live vectors recovered ---
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index_zero = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        index_zero.deinit();

        var loaded_store = try Store.load(testing.allocator, tmp.dir, "vectors.hvsf", cap);
        defer loaded_store.deinit(testing.allocator);
        var loaded_index = try Index.load(testing.allocator, &loaded_store, tmp.dir, "graph.hgrf", .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer loaded_index.deinit();

        var loaded_md = MutableMetadata.init();
        defer loaded_md.deinit(testing.allocator);
        var static_md = try @import("metadata.zig").load(testing.allocator, tmp.dir, "metadata.hmtf");
        defer static_md.deinit(testing.allocator);
        for (0..static_md.count) |i| {
            if (static_md.isTombstone(@intCast(i))) continue;
            try loaded_md.setAt(testing.allocator, @intCast(i), static_md.get(@intCast(i)));
        }

        var res = try Wal.open(testing.allocator, tmp.dir, "w.wal", dim, &loaded_store, &loaded_index, &loaded_md);
        defer res.wal.close();

        // Exactly one WAL record (the post-snapshot insert C) should have
        // replayed — and not been silently skipped.
        try testing.expectEqual(@as(usize, 1), res.replayed);

        try testing.expectEqual(@as(usize, 2), loaded_store.live_count);
        try testing.expect(!loaded_store.isDeleted(0));
        try testing.expect(!loaded_store.isDeleted(1));
        try testing.expectEqualSlices(f32, &v_c, loaded_store.get(0));
        try testing.expectEqualSlices(f32, &v_b, loaded_store.get(1));
        try testing.expectEqualStrings("C", loaded_md.get(0).?);
        try testing.expectEqualStrings("B", loaded_md.get(1).?);
    }
}

// Regression for the SIGTERM-doesn't-truncate-WAL bug. Before the fix the
// final snapshot on clean shutdown wrote a compacted snapshot but left the
// WAL full. On restart, replay layered the pre-snapshot records over the
// compacted state and produced a nonsensical store (e.g. live at id 1 with
// a graph that only had a node at id 0). After the fix we go through the
// single snapshot path that fsyncs + truncates, so restart sees a clean
// WAL and the on-disk snapshot is the authoritative state.
test "clean-shutdown snapshot truncates WAL; restart matches in-memory" {
    const Index = HnswIndexFn(8);
    const dim: u32 = 2;
    const cap: usize = 8;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const v_a = [_]f32{ 1, 0 };
    const v_b = [_]f32{ 0, 1 };

    // --- phase 1: insert A, insert B, delete 0, clean-shutdown snapshot ---
    {
        var store = try Store.init(testing.allocator, dim, cap);
        defer store.deinit(testing.allocator);
        var index = try Index.init(testing.allocator, &store, .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer index.deinit();
        var md = MutableMetadata.init();
        defer md.deinit(testing.allocator);
        var ws = try Index.Workspace.init(testing.allocator, cap, 16);
        defer ws.deinit(testing.allocator);

        var res = try Wal.open(testing.allocator, tmp.dir, "w.wal", dim, &store, &index, &md);

        try res.wal.appendInsert(.{ .id = 0, .level = 0, .vec = &v_a, .name = "A" });
        _ = try store.add(&v_a);
        try index.insertWithLevel(&ws, 0, 0);
        try md.setAt(testing.allocator, 0, "A");

        try res.wal.appendInsert(.{ .id = 1, .level = 0, .vec = &v_b, .name = "B" });
        _ = try store.add(&v_b);
        try index.insertWithLevel(&ws, 1, 0);
        try md.setAt(testing.allocator, 1, "B");

        try res.wal.appendDelete(.{ .id = 0 });
        try index.delete(&ws, 0);
        md.deleteAt(testing.allocator, 0);

        // Clean shutdown: exactly what dispatcher.snapshotNow does.
        try snapshotAll(testing.allocator, tmp.dir, &store, &index, &md, &res.wal);

        // WAL is now header-only; no dangling records to replay on restart.
        try testing.expectEqual(@as(u64, HEADER_SIZE), try res.wal.file.getEndPos());
        res.wal.close();
    }

    // --- phase 2: reopen, expect exactly the pre-shutdown state ---
    {
        var loaded_store = try Store.load(testing.allocator, tmp.dir, "vectors.hvsf", cap);
        defer loaded_store.deinit(testing.allocator);
        var loaded_index = try Index.load(testing.allocator, &loaded_store, tmp.dir, "graph.hgrf", .{
            .max_vectors = cap,
            .max_upper_slots = cap,
            .ef_construction = 16,
            .seed = 1,
        });
        defer loaded_index.deinit();

        var loaded_md = MutableMetadata.init();
        defer loaded_md.deinit(testing.allocator);
        var static_md = try @import("metadata.zig").load(testing.allocator, tmp.dir, "metadata.hmtf");
        defer static_md.deinit(testing.allocator);
        for (0..static_md.count) |i| {
            if (static_md.isTombstone(@intCast(i))) continue;
            try loaded_md.setAt(testing.allocator, @intCast(i), static_md.get(@intCast(i)));
        }

        var res = try Wal.open(testing.allocator, tmp.dir, "w.wal", dim, &loaded_store, &loaded_index, &loaded_md);
        defer res.wal.close();
        try testing.expectEqual(@as(usize, 0), res.replayed);

        // id 0 was deleted, id 1 survives — same as the pre-shutdown
        // in-memory state.
        try testing.expectEqual(@as(usize, 2), loaded_store.count);
        try testing.expectEqual(@as(usize, 1), loaded_store.live_count);
        try testing.expect(loaded_store.isDeleted(0));
        try testing.expect(!loaded_store.isDeleted(1));
        try testing.expectEqualSlices(f32, &v_b, loaded_store.get(1));
        try testing.expectEqualStrings("B", loaded_md.get(1).?);
    }
}
