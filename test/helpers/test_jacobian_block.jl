using Manopt
using LinearAlgebra
using Test

@testset "JacobianBlock" begin
    B = [1.0 2.0; 3.0 4.0]
    J = JacobianBlock(5, 6, (2,), (3,), (B,))

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
    @test J * v == y_expected

    M = reshape(collect(1.0:18.0), 6, 3)
    JM_expected = zeros(5, 3)
    JM_expected[2:3, :] .= B * M[3:4, :]
    @test J * M == JM_expected

    L = reshape(collect(1.0:20.0), 4, 5)
    LJ_expected = zeros(4, 6)
    LJ_expected[:, 3:4] .= L[:, 2:3] * B
    @test L * J == LJ_expected

    J2 = JacobianBlock(5, 6, (2,), (3,), (fill(1.0, 2, 2),))
    S = J + J2
    @test S isa JacobianBlock
    @test Matrix(S) == Matrix(J) + Matrix(J2)

    A = ones(5, 6)
    @test J + A == Matrix(J) + A
    @test A + J == A + Matrix(J)

    @test Matrix(2.0 * J) == 2.0 * Matrix(J)
    @test Matrix(J * 2.0) == Matrix(J) * 2.0

    J3 = JacobianBlock(6, 4, (4,), (2,), ([5.0 6.0; 7.0 8.0],))
    @test Matrix(J * J3) == Matrix(J) * Matrix(J3)

    C = fill(2.0, 6, 6)
    C_expected = copy(C)
    mul!(C_expected, Matrix(J)', Matrix(J), 3.0, 0.5)
    mul!(C, J', J, 3.0, 0.5)
    @test C == C_expected

    C2 = similar(C)
    mul!(C2, J', J)
    @test C2 == Matrix(J)' * Matrix(J)

    @test_throws ArgumentError JacobianBlock(3, 3, (3,), (3,), (ones(2, 2),))

    B4 = [10.0 20.0]
    Jmulti = JacobianBlock(5, 6, (2, 5), (3, 1), (B, B4))
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
    @test Jmulti * v == Mmulti * v
    @test Jmulti * M == Mmulti * M
    @test L * Jmulti == L * Mmulti

    Cmulti = fill(3.0, 6, 6)
    Cmulti_expected = copy(Cmulti)
    mul!(Cmulti_expected, Matrix(Jmulti)', Matrix(Jmulti), 2.5, 0.25)
    mul!(Cmulti, Jmulti', Jmulti, 2.5, 0.25)
    @test Cmulti == Cmulti_expected
end
