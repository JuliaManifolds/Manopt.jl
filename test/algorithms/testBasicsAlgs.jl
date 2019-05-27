@testset "Median, Mean, and Variance (on Sphere(2))" begin
    M = Sphere(2)
    x = SnPoint([1., 0, 0.])
    ξ = SnTVector([0., 1., 1.])
    η = SnTVector([0., 1., -1.])
    y = exp(M,x,0.2*ξ)
    z = exp(M,x,0.2*η)
    # CPPA not that exact
    @test distance(M, mean(M, [x,y]; method = :GradientDescent), midPoint(M,x,y) ) ≈ 0
    @test distance(M, mean(M, [x,y]; method = :CyclicProximalPoint), midPoint(M,x,y) ) ≈ 0 atol=8*10^(-5)
    @test distance(M,
        mean(M, [x,y,z]; method = :GradientDescent),
        mean(M, [x,y,z]; method = :CyclicProximalPoint)
    ) ≈ 0 atol=10^(-4)
    # Variance
    m = mean(M, [x,y,z])
    v = 1/(2 * manifoldDimension(M)) * sum( distance.(Ref(M),Ref(m),[x,y,z]).^2 )
    @test variance(M,[x,y,z]) ≈ v atol=10^-8
    # Median
    p = SnPoint([0.960266, 0.197344, 0.197344])
    @test distance(M,median(M,[x,y]),p) ≈ 0
end