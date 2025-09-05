using Manifolds, Manopt, ManifoldsBase, Test

@testset "Truncated Conjugate Gradient Descent" begin
    M = Grassmann(3, 2)
    p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    η = zero_vector(M, p)
    s = TruncatedConjugateGradientState(TangentSpace(M, p); X = η)
    @test startswith(
        repr(s), "# Solver state for `Manopt.jl`s Truncated Conjugate Gradient Descent\n"
    )
    @test get_iterate(s) == η
    srr = StopWhenResidualIsReducedByFactorOrPower()
    ssr1 = Manopt.status_summary(srr)
    @test ssr1 == "Residual reduced by factor 0.1 or power 1.0:\tnot reached"
    @test repr(srr) == "StopWhenResidualIsReducedByFactorOrPower(0.1, 1.0)\n    $(ssr1)"
    str = StopWhenTrustRegionIsExceeded()
    str1 = Manopt.status_summary(str)
    @test str1 == "Trust region exceeded:\tnot reached"
    @test repr(str) == "StopWhenTrustRegionIsExceeded()\n    $(str1)"
    @test get_reason(str) == ""
    # Trigger manually
    str.at_iteration = 1
    @test length(get_reason(str)) > 0
    scn = StopWhenCurvatureIsNegative()
    scn1 = Manopt.status_summary(scn)
    @test scn1 == "Curvature is negative:\tnot reached"
    @test repr(scn) == "StopWhenCurvatureIsNegative()\n    $(scn1)"
    smi = StopWhenModelIncreased()
    smi1 = Manopt.status_summary(smi)
    @test smi1 == "Model Increased:\tnot reached"
    @test repr(smi) == "StopWhenModelIncreased()\n    $(smi1)"
end
