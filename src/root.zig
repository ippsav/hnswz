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

test {
    _ = @import("distance.zig");
    _ = @import("store.zig");
    _ = @import("bruteforce.zig");
    _ = @import("heap.zig");
    _ = @import("hnsw.zig");
    _ = @import("ollama.zig");
}
