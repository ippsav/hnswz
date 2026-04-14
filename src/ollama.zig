const std = @import("std");
const distance = @import("distance.zig");

/// Runtime-polymorphic embedder interface (vtable-based).
/// Implementations must return caller-owned slices.
pub const Embedder = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        embed: *const fn (*anyopaque, []const u8, std.mem.Allocator) anyerror![]f32,
        embedBatch: *const fn (*anyopaque, []const []const u8, std.mem.Allocator) anyerror![][]f32,
    };

    pub fn embed(self: Embedder, text: []const u8, allocator: std.mem.Allocator) anyerror![]f32 {
        return self.vtable.embed(self.ptr, text, allocator);
    }

    pub fn embedBatch(self: Embedder, texts: []const []const u8, allocator: std.mem.Allocator) anyerror![][]f32 {
        return self.vtable.embedBatch(self.ptr, texts, allocator);
    }
};

// ── OllamaClient ──────────────────────────────────────────────────────

pub const OllamaClient = struct {
    http_client: std.http.Client,
    model: []const u8,
    base_url: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, model: []const u8, base_url: []const u8) OllamaClient {
        return .{
            .http_client = .{ .allocator = allocator },
            .model = model,
            .base_url = base_url,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *OllamaClient) void {
        self.http_client.deinit();
    }

    pub fn embedder(self: *OllamaClient) Embedder {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable_impl,
        };
    }

    const vtable_impl = Embedder.VTable{
        .embed = typeErasedEmbed,
        .embedBatch = typeErasedEmbedBatch,
    };

    fn typeErasedEmbed(ptr: *anyopaque, text: []const u8, allocator: std.mem.Allocator) anyerror![]f32 {
        const self: *OllamaClient = @ptrCast(@alignCast(ptr));
        return self.embedOne(text, allocator);
    }

    fn typeErasedEmbedBatch(ptr: *anyopaque, texts: []const []const u8, allocator: std.mem.Allocator) anyerror![][]f32 {
        const self: *OllamaClient = @ptrCast(@alignCast(ptr));
        return self.embedBatch_(texts, allocator);
    }

    pub fn embedOne(self: *OllamaClient, text: []const u8, allocator: std.mem.Allocator) ![]f32 {
        const texts = &[_][]const u8{text};
        const results = try self.embedBatch_(texts, allocator);
        defer allocator.free(results);
        return results[0];
    }

    pub fn embedBatch_(self: *OllamaClient, texts: []const []const u8, allocator: std.mem.Allocator) ![][]f32 {
        // Build JSON body
        const body = try buildRequestBody(self.model, texts, allocator);
        defer allocator.free(body);

        // Build URL
        const url = try std.fmt.allocPrint(allocator, "{s}/api/embed", .{self.base_url});
        defer allocator.free(url);

        // Allocating writer for the response body
        var response_storage: std.io.Writer.Allocating = .init(allocator);
        defer response_storage.deinit();

        const result = try self.http_client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = body,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
            },
            .response_writer = &response_storage.writer,
        });

        if (result.status != .ok) {
            return error.HttpRequestFailed;
        }

        const response_body = response_storage.written();
        return parseEmbeddings(response_body, texts.len, allocator);
    }

    fn buildRequestBody(model: []const u8, texts: []const []const u8, allocator: std.mem.Allocator) ![]u8 {
        const Request = struct {
            model: []const u8,
            input: []const []const u8,
        };
        return std.json.Stringify.valueAlloc(allocator, Request{
            .model = model,
            .input = texts,
        }, .{});
    }

    fn parseEmbeddings(body: []const u8, expected_count: usize, allocator: std.mem.Allocator) ![][]f32 {
        const Response = struct {
            embeddings: [][]f32,
        };

        const parsed = try std.json.parseFromSlice(Response, allocator, body, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        if (parsed.value.embeddings.len != expected_count) {
            return error.UnexpectedEmbeddingCount;
        }

        const result = try allocator.alloc([]f32, expected_count);
        errdefer {
            for (result) |r| allocator.free(r);
            allocator.free(result);
        }

        for (parsed.value.embeddings, 0..) |embedding, i| {
            result[i] = try allocator.dupe(f32, embedding);
        }

        return result;
    }
};

/// Check that a vector is unit-normalized (L2 norm ≈ 1.0).
pub fn verifyNormalization(vec: []const f32) !void {
    const norm = @sqrt(distance.dot(vec, vec));
    if (@abs(norm - 1.0) > 1e-4) {
        return error.NotNormalized;
    }
}

// ── FakeEmbedder (for tests) ──────────────────────────────────────────

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
        .embedBatch = typeErasedEmbedBatch,
    };

    fn typeErasedEmbed(ptr: *anyopaque, text: []const u8, allocator: std.mem.Allocator) anyerror![]f32 {
        const self: *FakeEmbedder = @ptrCast(@alignCast(ptr));
        return self.generateVector(text, allocator);
    }

    fn typeErasedEmbedBatch(ptr: *anyopaque, texts: []const []const u8, allocator: std.mem.Allocator) anyerror![][]f32 {
        const self: *FakeEmbedder = @ptrCast(@alignCast(ptr));
        const result = try allocator.alloc([]f32, texts.len);
        errdefer {
            for (result) |r| allocator.free(r);
            allocator.free(result);
        }
        for (texts, 0..) |text, i| {
            result[i] = try self.generateVector(text, allocator);
        }
        return result;
    }

    fn generateVector(self: *FakeEmbedder, text: []const u8, allocator: std.mem.Allocator) ![]f32 {
        // Hash text with seed for deterministic, text-dependent output
        const hash = std.hash.Wyhash.hash(self.seed, text);
        var prng = std.Random.DefaultPrng.init(hash);
        const random = prng.random();

        const vec = try allocator.alloc(f32, self.dim);
        errdefer allocator.free(vec);

        for (vec) |*x| {
            x.* = random.float(f32) * 2.0 - 1.0;
        }

        // Normalize to unit length
        var sum: f32 = 0;
        for (vec) |x| sum += x * x;
        const len = @sqrt(sum);
        if (len > 0) {
            for (vec) |*x| x.* /= len;
        }

        return vec;
    }
};

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "FakeEmbedder returns unit-normalized vectors" {
    var fake = FakeEmbedder.init(1024, 42);
    const emb = fake.embedder();

    const vec = try emb.embed("hello world", testing.allocator);
    defer testing.allocator.free(vec);

    try testing.expectEqual(@as(usize, 1024), vec.len);
    try verifyNormalization(vec);
}

test "FakeEmbedder is deterministic" {
    var fake = FakeEmbedder.init(128, 99);
    const emb = fake.embedder();

    const v1 = try emb.embed("same text", testing.allocator);
    defer testing.allocator.free(v1);

    const v2 = try emb.embed("same text", testing.allocator);
    defer testing.allocator.free(v2);

    try testing.expectEqualSlices(f32, v1, v2);
}

test "FakeEmbedder different texts produce different vectors" {
    var fake = FakeEmbedder.init(128, 99);
    const emb = fake.embedder();

    const v1 = try emb.embed("hello", testing.allocator);
    defer testing.allocator.free(v1);

    const v2 = try emb.embed("goodbye", testing.allocator);
    defer testing.allocator.free(v2);

    // At least one component should differ
    var differ = false;
    for (v1, v2) |a, b| {
        if (a != b) {
            differ = true;
            break;
        }
    }
    try testing.expect(differ);
}

test "FakeEmbedder embedBatch returns correct count" {
    var fake = FakeEmbedder.init(64, 7);
    const emb = fake.embedder();

    const texts = &[_][]const u8{ "one", "two", "three" };
    const results = try emb.embedBatch(texts, testing.allocator);
    defer {
        for (results) |r| testing.allocator.free(r);
        testing.allocator.free(results);
    }

    try testing.expectEqual(@as(usize, 3), results.len);
    for (results) |vec| {
        try testing.expectEqual(@as(usize, 64), vec.len);
        try verifyNormalization(vec);
    }
}

test "verifyNormalization accepts unit vectors" {
    const v = [_]f32{ 0.6, 0.8 }; // 0.36 + 0.64 = 1.0
    try verifyNormalization(&v);
}

test "verifyNormalization rejects non-unit vectors" {
    const v = [_]f32{ 3.0, 4.0 }; // norm = 5.0
    try testing.expectError(error.NotNormalized, verifyNormalization(&v));
}

test "parseEmbeddings parses Ollama response" {
    const body =
        \\{"model":"test","embeddings":[[1.0,0.0,0.0],[0.0,1.0,0.0]]}
    ;
    const result = try OllamaClient.parseEmbeddings(body, 2, testing.allocator);
    defer {
        for (result) |r| testing.allocator.free(r);
        testing.allocator.free(result);
    }

    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expectEqual(@as(usize, 3), result[0].len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0][0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result[0][1], 1e-6);
}

test "buildRequestBody produces valid JSON" {
    const body = try OllamaClient.buildRequestBody("test-model", &.{ "hello", "world" }, testing.allocator);
    defer testing.allocator.free(body);

    // Parse it back to verify it's valid JSON
    const parsed = try std.json.parseFromSlice(struct {
        model: []const u8,
        input: []const []const u8,
    }, testing.allocator, body, .{});
    defer parsed.deinit();

    try testing.expectEqualStrings("test-model", parsed.value.model);
    try testing.expectEqual(@as(usize, 2), parsed.value.input.len);
    try testing.expectEqualStrings("hello", parsed.value.input[0]);
    try testing.expectEqualStrings("world", parsed.value.input[1]);
}
