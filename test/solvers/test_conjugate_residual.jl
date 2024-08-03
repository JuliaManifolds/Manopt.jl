using Manifolds, Manopt, Test

@testset "Conjugate Residual" begin
    M = ℝ^2
    p = [1.0, 1.0]
    TpM = TangentSpace(M, p)

    Am = [2.0 1.0; 1.0 4.0]
    bv = [1.0, 2.0]
    ps = Am \ (-bv)
    X0 = [3.0, 4.0]
    A(M, X, V) = Am * V
    b(M, p) = bv

    slso = SymmetricLinearSystemObjective(A, b)
    pT = conjugate_residual(TpM, slso, X0)
    pT2 = conjugate_residual(TpM, A, b, X0)
    @test norm(ps - pT) < 3e-15
    @test norm(pT2 - pT) < 3e-15
    @test get_cost(TpM, slso, pT) < 5e-15
end
