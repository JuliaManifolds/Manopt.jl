using Manopt
using LinearAlgebra
using Test

@testset "BlockNonzeroMatrix and BlockNonzeroVector" begin
    bv = BlockNonzeroVector(6, (2, 5), ([1.0, 2.0], [3.0, 4.0]))
    @test size(bv) == (6,)
    @test length(bv) == 6
    @test bv[1] == 0.0
    @test bv[2] == 1.0
    @test bv[6] == 4.0
    @test Vector(bv) == [0.0, 1.0, 2.0, 0.0, 3.0, 4.0]
    @test_throws ArgumentError BlockNonzeroVector(3, (3,), ([1.0, 2.0],))

    Cv = fill(2.0, 6, 6)
    Cv_expected = copy(Cv)
    bv_dense = Vector(bv)
    mul!(Cv_expected, bv_dense, bv_dense', 1.7, 0.3)
    mul!(Cv, bv, bv', 1.7, 0.3)
    @test Cv == Cv_expected

    Cv2 = similar(Cv)
    mul!(Cv2, bv, bv')
    @test Cv2 == bv_dense * bv_dense'

    B = [1.0 2.0; 3.0 4.0]
    J = BlockNonzeroMatrix(5, 6, (2,), (3,), (B,))

    @test size(J) == (5, 6)
    @test J[1, 1] == 0.0
    @test J[2, 3] == 1.0
    @test J[3, 4] == 4.0

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
    @test transpose(J) * y_expected == transpose(Matrix(J)) * y_expected

    M = reshape(collect(1.0:18.0), 6, 3)
    JM_expected = zeros(5, 3)
    JM_expected[2:3, :] .= B * M[3:4, :]
    @test J * M == JM_expected

    L = reshape(collect(1.0:20.0), 4, 5)
    LJ_expected = zeros(4, 6)
    LJ_expected[:, 3:4] .= L[:, 2:3] * B
    @test L * J == LJ_expected

    J2 = BlockNonzeroMatrix(5, 6, (2,), (3,), (fill(1.0, 2, 2),))
    S = J + J2
    @test S isa BlockNonzeroMatrix
    @test Matrix(S) == Matrix(J) + Matrix(J2)

    A = ones(5, 6)
    @test J + A == Matrix(J) + A
    @test A + J == A + Matrix(J)

    @test Matrix(2.0 * J) == 2.0 * Matrix(J)
    @test Matrix(J * 2.0) == Matrix(J) * 2.0

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
    @test Jmulti[5, 1] == 10.0
    @test Jmulti[5, 2] == 20.0
    @test Jmulti[4, 1] == 0.0
    ymulti = Jmulti * v
    @test ymulti isa BlockNonzeroVector
    @test Vector(ymulti) == Mmulti * v
    @test Jmulti * M == Mmulti * M
    @test L * Jmulti == L * Mmulti

    Cmulti = fill(3.0, 6, 6)
    Cmulti_expected = copy(Cmulti)
    mul!(Cmulti_expected, Matrix(Jmulti)', Matrix(Jmulti), 2.5, 0.25)
    mul!(Cmulti, Jmulti', Jmulti, 2.5, 0.25)
    @test Cmulti == Cmulti_expected
end
