const std = @import("std");
const distance = @import("distance.zig");

/// Runtime-polymorphic embedder interface. Implementations write the embedding
/// into the caller-provided `out` buffer (length must equal the model's dim).
pub const Embedder = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        embed: *const fn (ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void,
    };

    pub fn embed(self: Embedder, text: []const u8, out: []f32) anyerror!void {
        return self.vtable.embed(self.ptr, text, out);
    }
};

/// Ollama embedding client. All scratch buffers are preallocated at init.
/// Single-threaded — not safe for concurrent use.
pub const OllamaClient = struct {
    http_client: std.http.Client,
    model: []const u8,
    base_url: []const u8,
    allocator: std.mem.Allocator,

    // Preallocated scratch.
    request_body: []u8,
    response_body: []u8,
    url_buf: []u8,
    parse_arena_buf: []u8,

    pub const InitOptions = struct {
        /// Upper bound on the text we ever embed. Used to size request buffer.
        max_text_bytes: usize,
        /// Expected embedding dimension. Used to size response buffer.
        /// Budget: dim * ~24 bytes (float string + JSON overhead) + envelope.
        dim: usize,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        model: []const u8,
        base_url: []const u8,
        opts: InitOptions,
    ) !OllamaClient {
        // Request envelope: {"model":"...","input":["..."]} → ~60 bytes overhead.
        const request_body = try allocator.alloc(u8, opts.max_text_bytes + 256 + model.len);
        errdefer allocator.free(request_body);

        // Response envelope: {"model":"...","embeddings":[[...]]} + dim floats.
        const response_body = try allocator.alloc(u8, opts.dim * 24 + 1024);
        errdefer allocator.free(response_body);

        const url_buf = try allocator.alloc(u8, base_url.len + 64);
        errdefer allocator.free(url_buf);

        // JSON parse arena: needs room for the outer struct + embeddings[][] shape.
        const parse_arena_buf = try allocator.alloc(u8, opts.dim * 16 + 4096);

        return .{
            .http_client = .{ .allocator = allocator },
            .model = model,
            .base_url = base_url,
            .allocator = allocator,
            .request_body = request_body,
            .response_body = response_body,
            .url_buf = url_buf,
            .parse_arena_buf = parse_arena_buf,
        };
    }

    pub fn deinit(self: *OllamaClient) void {
        self.http_client.deinit();
        self.allocator.free(self.request_body);
        self.allocator.free(self.response_body);
        self.allocator.free(self.url_buf);
        self.allocator.free(self.parse_arena_buf);
    }

    pub fn embedder(self: *OllamaClient) Embedder {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable_impl,
        };
    }

    const vtable_impl = Embedder.VTable{
        .embed = typeErasedEmbed,
    };

    fn typeErasedEmbed(ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void {
        const self: *OllamaClient = @ptrCast(@alignCast(ptr));
        return self.embed(text, out);
    }

    pub fn embed(self: *OllamaClient, text: []const u8, out: []f32) !void {
        // 1. Build JSON request body into self.request_body.
        const Request = struct {
            model: []const u8,
            input: []const []const u8,
        };
        const texts_arr = [_][]const u8{text};
        var req_writer = std.io.Writer.fixed(self.request_body);
        try std.json.Stringify.value(
            Request{ .model = self.model, .input = &texts_arr },
            .{},
            &req_writer,
        );
        const body = self.request_body[0..req_writer.end];

        // 2. Build URL into self.url_buf.
        const url = try std.fmt.bufPrint(self.url_buf, "{s}/api/embed", .{self.base_url});

        // 3. Fetch into self.response_body via a fixed writer.
        var resp_writer = std.io.Writer.fixed(self.response_body);
        const result = try self.http_client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
            },
            .response_writer = &resp_writer,
        });

        if (result.status != .ok) return error.HttpRequestFailed;

        const response_body = self.response_body[0..resp_writer.end];

        // 4. Parse into out via a reset-per-call FixedBufferAllocator arena.
        var fba = std.heap.FixedBufferAllocator.init(self.parse_arena_buf);
        try parseSingleEmbedding(response_body, out, fba.allocator());
    }
};

fn parseSingleEmbedding(body: []const u8, out: []f32, arena: std.mem.Allocator) !void {
    const Response = struct {
        embeddings: [][]f32,
    };
    const parsed = try std.json.parseFromSliceLeaky(Response, arena, body, .{
        .ignore_unknown_fields = true,
    });

    if (parsed.embeddings.len != 1) return error.UnexpectedEmbeddingCount;
    const vec = parsed.embeddings[0];
    if (vec.len != out.len) return error.DimensionMismatch;
    @memcpy(out, vec);
}

/// Check that a vector is unit-normalized (L2 norm ≈ 1.0).
pub fn verifyNormalization(vec: []const f32) !void {
    const norm = @sqrt(distance.dot(vec, vec));
    if (@abs(norm - 1.0) > 1e-4) {
        return error.NotNormalized;
    }
}

pub const FakeEmbedder = struct {
    dim: usize,
    seed: u64,

    pub fn init(dim: usize, seed: u64) FakeEmbedder {
        return .{ .dim = dim, .seed = seed };
    }

    pub fn embedder(self: *FakeEmbedder) Embedder {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable_impl,
        };
    }

    const vtable_impl = Embedder.VTable{
        .embed = typeErasedEmbed,
    };

    fn typeErasedEmbed(ptr: *anyopaque, text: []const u8, out: []f32) anyerror!void {
        const self: *FakeEmbedder = @ptrCast(@alignCast(ptr));
        return self.generateVector(text, out);
    }

    fn generateVector(self: *FakeEmbedder, text: []const u8, out: []f32) !void {
        if (out.len != self.dim) return error.DimensionMismatch;
        const hash = std.hash.Wyhash.hash(self.seed, text);
        var prng = std.Random.DefaultPrng.init(hash);
        const random = prng.random();

        for (out) |*x| x.* = random.float(f32) * 2.0 - 1.0;

        // Normalize to unit length.
        var sum: f32 = 0;
        for (out) |x| sum += x * x;
        const len = @sqrt(sum);
        if (len > 0) {
            for (out) |*x| x.* /= len;
        }
    }
};



const testing = std.testing;

test "FakeEmbedder returns unit-normalized vectors" {
    var fake = FakeEmbedder.init(1024, 42);
    const emb = fake.embedder();

    var vec: [1024]f32 = undefined;
    try emb.embed("hello world", &vec);
    try verifyNormalization(&vec);
}

test "FakeEmbedder is deterministic" {
    var fake = FakeEmbedder.init(128, 99);
    const emb = fake.embedder();

    var v1: [128]f32 = undefined;
    var v2: [128]f32 = undefined;
    try emb.embed("same text", &v1);
    try emb.embed("same text", &v2);

    try testing.expectEqualSlices(f32, &v1, &v2);
}

test "FakeEmbedder different texts produce different vectors" {
    var fake = FakeEmbedder.init(128, 99);
    const emb = fake.embedder();

    var v1: [128]f32 = undefined;
    var v2: [128]f32 = undefined;
    try emb.embed("hello", &v1);
    try emb.embed("goodbye", &v2);

    var differ = false;
    for (&v1, &v2) |a, b| {
        if (a != b) {
            differ = true;
            break;
        }
    }
    try testing.expect(differ);
}

test "verifyNormalization accepts unit vectors" {
    const v = [_]f32{ 0.6, 0.8 }; // 0.36 + 0.64 = 1.0
    try verifyNormalization(&v);
}

test "verifyNormalization rejects non-unit vectors" {
    const v = [_]f32{ 3.0, 4.0 }; // norm = 5.0
    try testing.expectError(error.NotNormalized, verifyNormalization(&v));
}

test "parseSingleEmbedding parses Ollama response into fixed buffer" {
    const body =
        \\{"model":"test","embeddings":[[1.0,0.5,0.25]]}
    ;
    var out: [3]f32 = undefined;
    var arena_buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&arena_buf);
    try parseSingleEmbedding(body, &out, fba.allocator());

    try testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), out[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.25), out[2], 1e-6);
}

test "parseSingleEmbedding rejects dim mismatch" {
    const body =
        \\{"embeddings":[[1.0,0.5]]}
    ;
    var out: [3]f32 = undefined;
    var arena_buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&arena_buf);
    try testing.expectError(error.DimensionMismatch, parseSingleEmbedding(body, &out, fba.allocator()));
}
