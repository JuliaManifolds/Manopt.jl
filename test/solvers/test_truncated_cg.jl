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
    @test scn1 == "A stopping criterion to stop when the is negative\n$(Manopt._MANOPT_INDENT)not reached"
    @test repr(scn) == "StopWhenCurvatureIsNegative()"
    smi = StopWhenModelIncreased()
    smi1 = Manopt.status_summary(smi)
    @test smi1 == "Model Increased:$(Manopt._MANOPT_INDENT)not reached"
    @test repr(smi) == "StopWhenModelIncreased()"
end
