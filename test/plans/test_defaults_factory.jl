using Manopt, Manifolds, Test

# A rule that does not need a manifold but has defaults
struct FactoryDummyRule{R <: Real}
    t::R
    FactoryDummyRule(; t::R = 1.0) where {R <: Real} = new{R}(t)
end

# A rule that requires a point but not a manifold
struct FactoryDummyPointRule{P, R <: Real}
    p::P
    t::R
end
FactoryDummyPointRule(p; t::R = 1.0) where {R <: Real} = FactoryDummyPointRule(allocate(p), t)

# A rule that requires both a manifold and a point
struct FactoryDummyManifoldPointRule{M, P, R <: Real}
    M::M
    p::P
    t::R
end
FactoryDummyManifoldPointRule(M::TM, p; t::R = 1.0) where {TM <: AbstractManifold, R <: Real} =
    FactoryDummyManifoldPointRule(M, allocate(p), t)

# A rule that requires a manifold but no point
struct FactoryDummyManifoldRule{M, R <: Real}
    M::M
    t::R
end
FactoryDummyManifoldRule(M::TM; t::R = 1.0) where {TM <: AbstractManifold, R <: Real} =
    FactoryDummyManifoldRule(M, t)

@testset "ManifoldsDefaultFactory" begin
    fdr = Manopt.ManifoldDefaultsFactory(
        FactoryDummyRule, Sphere(2); requires_manifold = false, t = 2.0
    )
    @test fdr().t == 2.0
    @test fdr(Euclidean(2)).t == 2.0
    @test startswith(repr(fdr), "ManifoldDefaultsFactory(FactoryDummyRule)")
    # A case without a manifold and with keywords instead.
    fdr2 = Manopt.ManifoldDefaultsFactory(FactoryDummyRule, 1)
    @test startswith(repr(fdr2), "ManifoldDefaultsFactory(FactoryDummyRule)")

    M = Sphere(2)
    p = Float32.(rand(M))
    fdrp = Manopt.ManifoldDefaultsFactory(
        FactoryDummyPointRule, M; requires_manifold = false, requires_point = true, t = 3.0
    )
    @test fdrp(M, p).p isa Vector{Float32}

    fdrmp = Manopt.ManifoldDefaultsFactory(
        FactoryDummyManifoldPointRule, M; requires_manifold = true, requires_point = true, t = 4.0
    )
    obj = fdrmp(M, p)
    @test obj.p isa Vector{Float32}

    fdrm = Manopt.ManifoldDefaultsFactory(
        FactoryDummyManifoldRule, M; requires_manifold = true, requires_point = false, t = 5.0
    )
    obj_m = fdrm(M)
    @test obj_m.M === M
end
