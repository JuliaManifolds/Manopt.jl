using Manifolds, Manopt, ManifoldsBase, Test

@testset "Truncated Conjugate Gradient Descent" begin
    M = Grassmann(3, 2)
    p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    η = zero_vector(M, p)
    s = TruncatedConjugateGradientState(TangentSpace(M, p); X = η)
    @test startswith(
        Manopt.status_summary(s; context = :default),
        "# Solver state for `Manopt.jl`s Truncated Conjugate Gradient Descent\n"
    )
    @test get_iterate(s) == η
    # the default radius comes from the base manifold (finite here), not the flat tangent space
    @test s.trust_region_radius ≈ injectivity_radius(M) / 4
    # standalone solve with a negative-curvature model stays finite with the default radius
    A = [2.0 1.0 0.0; 1.0 -3.0 1.0; 0.0 1.0 4.0]
    fq(TpM, X) = 0.5 * sum(X .* (A * X)) - X[1, 1]
    trmo = TrustRegionModelObjective(
        ManifoldHessianObjective((M, q) -> 0.0, (M, q) -> project(M, p, -A[:, 1:2]), (M, q, X) -> project(M, p, A * X))
    )
    Yfin = truncated_conjugate_gradient_descent(TangentSpace(M, p), trmo, p, η)
    @test all(isfinite, Yfin)
    srr = StopWhenResidualIsReducedByFactorOrPower()
    ssr1 = Manopt.status_summary(srr)
    @test startswith(ssr1, "A stopping criterion used within tCG to check whether the residual is reduced by factor")
    @test repr(srr) == "StopWhenResidualIsReducedByFactorOrPower(0.1, 1.0)"
    str = StopWhenTrustRegionIsExceeded()
    str1 = Manopt.status_summary(str)
    @test str1 == "A stopping criterion to stop when the trust region radius (0.0) is exceeded.\n$(Manopt._MANOPT_INDENT)not reached"
    @test repr(str) == "StopWhenTrustRegionIsExceeded()"
    @test get_reason(str) == ""
    # Trigger manually
    str.at_iteration = 1
    @test length(get_reason(str)) > 0
    scn = StopWhenCurvatureIsNegative()
    scn1 = Manopt.status_summary(scn)
    @test scn1 == "A stopping criterion to stop when the curvature is negative\n$(Manopt._MANOPT_INDENT)not reached"
    @test repr(scn) == "StopWhenCurvatureIsNegative()"
    smi = StopWhenModelIncreased()
    smi1 = Manopt.status_summary(smi)
    @test startswith(smi1, "A stopping criterion to indicate when the model increased.")
    @test repr(smi) == "StopWhenModelIncreased()"
end
