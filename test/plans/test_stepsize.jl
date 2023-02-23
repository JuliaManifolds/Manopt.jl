using Manopt, Manifolds, Test

@testset "Stepsize" begin
    s = ArmijoLinesearch()
    @test startswith(repr(s), "ArmijoLineseach() with keyword parameters\n")
    s_stat = Manopt.status_summary(s)
    @test startswith(s_stat, "ArmijoLineseach() with keyword parameters\n")
    @test endswith(s_stat, "of 1.0")

    s2 = NonmonotoneLinesearch()
    @test startswith(repr(s2), "NonmonotoneLinesearch() with keyword arguments\n")
    s2b = NonmonotoneLinesearch(Euclidean(2)) # with manifold -> faster storage
    @test startswith(repr(s2b), "NonmonotoneLinesearch() with keyword arguments\n")
    s3 = WolfePowellBinaryLinesearch()
    @test startswith(repr(s3), "WolfePowellBinaryLinesearch(DefaultManifold(), ")
    # no stepsize yet so repr and summary are the same
    @test repr(s3) == Manopt.status_summary(s3)
    s4 = WolfePowellLinesearch()
    @test startswith(repr(s4), "WolfePowellLinesearch(DefaultManifold(), ")
    # no stepsize yet so repr and summary are the same
    @test repr(s4) == Manopt.status_summary(s4)
end
