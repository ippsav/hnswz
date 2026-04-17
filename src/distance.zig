const std = @import("std");
const assert = std.debug.assert;

const VecLen = std.simd.suggestVectorLength(f32) orelse 4;
const Vec4 = @Vector(VecLen, f32);

pub fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);

    var acc: Vec4 = @splat(0);

    var sum: f32 = 0;
    var i: usize = 0;

    while (i + VecLen <= a.len) : (i += VecLen) {
        const va: Vec4 = a[i..][0..VecLen].*;
        const vb: Vec4 = b[i..][0..VecLen].*;

        acc += va * vb;
    }

    sum = @reduce(.Add, acc);

    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

fn norm(a: []const f32) f32 {
    return @sqrt(dot(a, a));
}

pub fn cosine(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);

    return 1 - dot(a, b) / (norm(a) * norm(b));
}

pub fn cosineNormalized(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);

    return 1 - dot(a, b);
}

test "dot product of len 4" {
    const a: [4]f32 = .{ 1, 2, 3, 4 };
    const b: [4]f32 = .{ 5, 6, 7, 8 };

    const res = dot(&a, &b);

    try std.testing.expectEqual(res, 70);
}

test "dot product of len 5 (remainder case)" {
    const a: [5]f32 = .{ 1, 2, 3, 4, 6 };
    const b: [5]f32 = .{ 7, 8, 9, 10, 11 };

    const res = dot(&a, &b);

    try std.testing.expectEqual(res, 156);
}

test "dot product of zeroes" {
    const a: [2]f32 = .{ 0, 0 };
    const b: [2]f32 = .{ 0, 0 };

    const res = dot(&a, &b);

    try std.testing.expectEqual(res, 0);
}

test "cosine distance of identical vectors" {
    const a: [2]f32 = .{ 1, 0 };
    const b: [2]f32 = .{ 1, 0 };

    try std.testing.expectApproxEqAbs(cosine(&a, &b), 0, 1e-6);
}

test "cosine distance of orthogonal vectors" {
    const a: [2]f32 = .{ 1, 0 };
    const b: [2]f32 = .{ 0, 1 };

    try std.testing.expectApproxEqAbs(cosine(&a, &b), 1, 1e-6);
}

test "cosine distance of opposite vectors" {
    const a: [2]f32 = .{ 1, 0 };
    const b: [2]f32 = .{ -1, 0 };

    try std.testing.expectApproxEqAbs(cosine(&a, &b), 2, 1e-6);
}

test "cosine distance of non-unit vectors" {
    const a: [2]f32 = .{ 3, 4 };
    const b: [2]f32 = .{ 3, 4 };

    try std.testing.expectApproxEqAbs(cosine(&a, &b), 0, 1e-6);
}

test "cosineNormalized of identical unit vectors" {
    const a: [2]f32 = .{ 1, 0 };
    const b: [2]f32 = .{ 1, 0 };

    try std.testing.expectApproxEqAbs(cosineNormalized(&a, &b), 0, 1e-6);
}

test "cosineNormalized of orthogonal unit vectors" {
    const a: [2]f32 = .{ 1, 0 };
    const b: [2]f32 = .{ 0, 1 };

    try std.testing.expectApproxEqAbs(cosineNormalized(&a, &b), 1, 1e-6);
}

test "cosineNormalized of opposite unit vectors" {
    const a: [2]f32 = .{ 1, 0 };
    const b: [2]f32 = .{ -1, 0 };

    try std.testing.expectApproxEqAbs(cosineNormalized(&a, &b), 2, 1e-6);
}
