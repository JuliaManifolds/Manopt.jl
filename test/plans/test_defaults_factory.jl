using Manopt, Manifolds, Test

# A rule taht does not need a manifold but has defaults
struct FactoryDummyRule{R<:Real}
    t::R
    FactoryDummyRule(; t::R=1.0) where {R<:Real} = new{R}(t)
end

@testset "ManifoldsDefaultFactory" begin
    fdr = Manopt.ManifoldDefaultsFactory(
        FactoryDummyRule, Sphere(2); requires_manifold=false, t=2.0
    )
    @test fdr().t == 2.0
    @test fdr(Euclidean(2)).t == 2.0
    @test startswith(repr(fdr), "ManifoldDefaultsFactory(FactoryDummyRule)")
end
