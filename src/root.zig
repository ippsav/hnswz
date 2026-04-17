//! hnswz — Hierarchical Navigable Small World index library.
const std = @import("std");

pub const Store = @import("store.zig").Store;
pub const HnswIndex = @import("hnsw.zig").HnswIndex;

const ollama = @import("ollama.zig");
pub const OllamaClient = ollama.OllamaClient;
pub const Embedder = ollama.Embedder;
pub const FakeEmbedder = ollama.FakeEmbedder;

pub const bruteforce = @import("bruteforce.zig");
pub const distance = @import("distance.zig");
pub const heap = @import("heap.zig");
pub const config = @import("config.zig");
pub const metadata = @import("metadata.zig");
pub const benchmark = @import("benchmark.zig");
pub const protocol = @import("protocol.zig");
pub const metadata_mut = @import("metadata_mut.zig");
pub const server = @import("server.zig");
pub const client = @import("client.zig");
pub const io = @import("io.zig");
pub const dispatcher = @import("dispatcher.zig");
pub const lockfile = @import("lockfile.zig");
pub const wal = @import("wal.zig");

test {
    _ = @import("distance.zig");
    _ = @import("store.zig");
    _ = @import("bruteforce.zig");
    _ = @import("heap.zig");
    _ = @import("hnsw.zig");
    _ = @import("ollama.zig");
    _ = @import("config.zig");
    _ = @import("metadata.zig");
    _ = @import("benchmark.zig");
    _ = @import("protocol.zig");
    _ = @import("metadata_mut.zig");
    _ = @import("server.zig");
    _ = @import("client.zig");
    _ = @import("io.zig");
    _ = @import("dispatcher.zig");
    _ = @import("lockfile.zig");
    _ = @import("wal.zig");
}
