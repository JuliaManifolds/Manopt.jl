using Manopt
using LinearAlgebra
using RecursiveArrayTools
using Test

@testset "BlockNonzeroMatrix and BlockNonzeroVector" begin
    bv = BlockNonzeroVector(6, (2, 5), ([1.0, 2.0], [3.0, 4.0]))
    @test size(bv) == (6,)
    @test length(bv) == 6
    @test bv[1] == 0.0
    @test bv[2] == 1.0
    @test bv[6] == 4.0
    @test Vector(bv) == [0.0, 1.0, 2.0, 0.0, 3.0, 4.0]
    @test repr(bv) == "BlockNonzeroVector(6, (2, 5), ([1.0, 2.0], [3.0, 4.0]))"
    @test eval(Meta.parse(repr(bv))) == bv
    @test_throws ArgumentError BlockNonzeroVector(3, (3,), ([1.0, 2.0],))
    bv2 = copy(bv)
    bv2 .*= 2
    @test Vector(bv2) == 2 .* Vector(bv)
    @test Vector(bv) == [0.0, 1.0, 2.0, 0.0, 3.0, 4.0]

    Cv = fill(2.0, 6, 6)
    Cv_expected = copy(Cv)
    bv_dense = Vector(bv)
    mul!(Cv_expected, bv_dense, bv_dense', 1.7, 0.3)
    mul!(Cv, bv, bv', 1.7, 0.3)
    @test isapprox(Cv, Cv_expected)

    Cv2 = similar(Cv)
    mul!(Cv2, bv, bv')
    @test isapprox(Cv2, bv_dense * bv_dense')

    @testset "Reshaped BlockNonzeroVector in-place add" begin
        bv_reshape = BlockNonzeroVector(
            12,
            (2, 6, 11),
            ([1.0, 2.0], [3.0], [4.0, 5.0]),
        )
        Y_reshape = reshape(bv_reshape, 2, 2, 3)

        X = reshape(collect(1.0:12.0), 2, 2, 3)
        X_expected = copy(X)
        X_expected .+= reshape(Vector(bv_reshape), 2, 2, 3)
        X .+= Y_reshape
        @test X == X_expected

        X2 = fill(10.0, 2, 2, 3)
        X2_expected = copy(X2)
        X2_expected .+= reshape(Vector(bv_reshape), 2, 2, 3)
        X2 .= X2 .+ Y_reshape
        @test X2 == X2_expected

        Y_view_reshape = reshape(view(bv_reshape, 2:11), 2, 5)

        X3 = fill(7.0, 2, 5)
        X3_expected = copy(X3)
        X3_expected .+= reshape(Vector(view(bv_reshape, 2:11)), 2, 5)
        X3 .+= Y_view_reshape
        @test X3 == X3_expected

        X4 = fill(-3.0, 2, 5)
        X4_expected = copy(X4)
        X4_expected .+= reshape(Vector(view(bv_reshape, 2:11)), 2, 5)
        X4 .= Y_view_reshape .+ X4
        @test X4 == X4_expected
    end

    B = [1.0 2.0; 3.0 4.0]
    J = BlockNonzeroMatrix(5, 6, (2,), (3,), (B,))

    @test size(J) == (5, 6)
    @test J[1, 1] == 0.0
    @test J[2, 3] == 1.0
    @test J[3, 4] == 4.0
    @test repr(J) == "BlockNonzeroMatrix(5, 6, (2,), (3,), ([1.0 2.0; 3.0 4.0],))"
    @test eval(Meta.parse(repr(J))) == J

    @test Matrix(J) == [
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 2.0 0.0 0.0
        0.0 0.0 3.0 4.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
    ]

    v = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    y_expected = zeros(5)
    y_expected[2:3] .= B * v[3:4]
    y = J * v
    @test y isa BlockNonzeroVector
    @test Vector(y) == y_expected
    @test y == y_expected
    yt = J' * y_expected
    @test yt isa BlockNonzeroVector
    @test Vector(yt) == Matrix(J)' * y_expected
    yt_mul = fill(2.0, size(J, 2))
    yt_mul_expected = copy(yt_mul)
    mul!(yt_mul_expected, Matrix(J)', y_expected, 1.7, 0.3)
    mul!(yt_mul, J', y_expected, 1.7, 0.3)
    @test yt_mul == yt_mul_expected
    yt_mul2 = similar(yt_mul)
    mul!(yt_mul2, J', y_expected)
    @test yt_mul2 == Matrix(J)' * y_expected
    @test transpose(J) * y_expected == transpose(Matrix(J)) * y_expected

    M = reshape(collect(1.0:18.0), 6, 3)
    JM_expected = zeros(5, 3)
    JM_expected[2:3, :] .= B * M[3:4, :]
    @test J * M == JM_expected

    L = reshape(collect(1.0:20.0), 4, 5)
    LJ_expected = zeros(4, 6)
    LJ_expected[:, 3:4] .= L[:, 2:3] * B
    @test L * J == LJ_expected
    @test y_expected' * J == y_expected' * Matrix(J)
    @test y_expected' * J isa Adjoint
    @test transpose(y_expected) * J == transpose(y_expected) * Matrix(J)
    @test transpose(y_expected) * J isa Transpose

    J2 = BlockNonzeroMatrix(5, 6, (2,), (3,), (fill(1.0, 2, 2),))
    S = J + J2
    @test S isa BlockNonzeroMatrix
    @test Matrix(S) == Matrix(J) + Matrix(J2)

    A = ones(5, 6)
    @test J + A == Matrix(J) + A
    @test A + J == A + Matrix(J)

    @test Matrix(2.0 * J) == 2.0 * Matrix(J)
    @test Matrix(J * 2.0) == Matrix(J) * 2.0

    Jcopy = copy(J)
    Jcopy .*= 3
    @test Matrix(Jcopy) == 3 .* Matrix(J)
    @test Matrix(J) == [
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 2.0 0.0 0.0
        0.0 0.0 3.0 4.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
    ]

    J3 = BlockNonzeroMatrix(6, 4, (4,), (2,), ([5.0 6.0; 7.0 8.0],))
    @test Matrix(J * J3) == Matrix(J) * Matrix(J3)

    C = fill(2.0, 6, 6)
    C_expected = copy(C)
    mul!(C_expected, Matrix(J)', Matrix(J), 3.0, 0.5)
    mul!(C, J', J, 3.0, 0.5)
    @test C == C_expected

    C2 = similar(C)
    mul!(C2, J', J)
    @test C2 == Matrix(J)' * Matrix(J)

    @test_throws ArgumentError BlockNonzeroMatrix(3, 3, (3,), (3,), (ones(2, 2),))

    Bc = ComplexF64[1 + 2im 2 - im; -3im 4 + 5im]
    Jc = BlockNonzeroMatrix(5, 6, (2,), (3,), (Bc,))
    xc = ComplexF64[1 - im, 2 + 3im, -1 + im, 4 - 2im, 0.5 + 0.5im]
    yct = Jc' * xc
    @test yct isa BlockNonzeroVector
    @test Vector(yct) == Matrix(Jc)' * xc
    @test transpose(Jc) * xc == transpose(Matrix(Jc)) * xc
    @test xc' * Jc == xc' * Matrix(Jc)
    @test xc' * Jc isa Adjoint
    @test transpose(xc) * Jc == transpose(xc) * Matrix(Jc)
    @test transpose(xc) * Jc isa Transpose

    B4 = [10.0 20.0]
    Jmulti = BlockNonzeroMatrix(5, 6, (2, 5), (3, 1), (B, B4))
    Mmulti = [
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 2.0 0.0 0.0
        0.0 0.0 3.0 4.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        10.0 20.0 0.0 0.0 0.0 0.0
    ]
    @test Matrix(Jmulti) == Mmulti
    @test repr(Jmulti) ==
        "BlockNonzeroMatrix(5, 6, (2, 5), (3, 1), ([1.0 2.0; 3.0 4.0], [10.0 20.0]))"
    @test eval(Meta.parse(repr(Jmulti))) == Jmulti
    @test Jmulti[5, 1] == 10.0
    @test Jmulti[5, 2] == 20.0
    @test Jmulti[4, 1] == 0.0
    ymulti = Jmulti * v
    @test ymulti isa BlockNonzeroVector
    @test Vector(ymulti) == Mmulti * v
    ymulti_t = fill(1.0, size(Jmulti, 2))
    ymulti_t_expected = copy(ymulti_t)
    mul!(ymulti_t_expected, Matrix(Jmulti)', Vector(ymulti), 2.5, 0.25)
    mul!(ymulti_t, Jmulti', Vector(ymulti), 2.5, 0.25)
    @test ymulti_t == ymulti_t_expected
    @test Jmulti * M == Mmulti * M
    @test L * Jmulti == L * Mmulti

    Cmulti = fill(3.0, 6, 6)
    Cmulti_expected = copy(Cmulti)
    mul!(Cmulti_expected, Matrix(Jmulti)', Matrix(Jmulti), 2.5, 0.25)
    mul!(Cmulti, Jmulti', Jmulti, 2.5, 0.25)
    @test Cmulti == Cmulti_expected

    @test_throws DimensionMismatch mul!(zeros(size(J, 2) - 1), J', y_expected, 1.0, 0.0)
    @test_throws DimensionMismatch mul!(zeros(size(J, 2)), J', y_expected[1:(end - 1)], 1.0, 0.0)

    @testset "ArrayPartition in-place add fast-path" begin
        X = ArrayPartition(
            reshape(collect(1.0:12.0), 2, 2, 3),
            reshape(collect(13.0:18.0), 2, 3),
            reshape(collect(19.0:24.0), 2, 3),
        )
        X_expected = deepcopy(X)

        bv_fast = BlockNonzeroVector(
            12,
            (2, 6, 11),
            ([1.0, 2.0], [3.0], [4.0, 5.0]),
        )
        Y = ArrayPartition(
            zeros(2, 2, 3),
            reshape(view(bv_fast, 1:6), 2, 3),
            reshape(view(bv_fast, 7:12), 2, 3),
        )

        X_expected.x[1] .+= Y.x[1]
        X_expected.x[2] .+= Y.x[2]
        X_expected.x[3] .+= Y.x[3]

        X .+= Y

        @test X == X_expected
        @test Vector(reshape(Y.x[2], :)) == Vector(view(bv_fast, 1:6))
        @test Vector(reshape(Y.x[3], :)) == Vector(view(bv_fast, 7:12))
    end
end
