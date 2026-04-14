//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

test {
    _ = @import("distance.zig");
    _ = @import("store.zig");
    _ = @import("bruteforce.zig");
    _ = @import("heap.zig");
    _ = @import("hnsw.zig");
    _ = @import("ollama.zig");
}
