using Manifolds, Manopt, ManifoldsBase, Test

@testset "Truncated Conjugate Gradient Descent" begin
    M = Grassmann(3, 2)
    p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    η = zero_vector(M, p)
    s = TruncatedConjugateGradientState(M, p, η)
    @test startswith(
        repr(s), "# Solver state for `Manopt.jl`s Truncated Conjugate Gradient Descent\n"
    )
end
