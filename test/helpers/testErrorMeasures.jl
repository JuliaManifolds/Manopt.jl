@testset "Manopt.jl Error Measures" begin
    M = Sphere(2)
    N = PowerManifold(M,(2,))
    using Random: seed!
    seed!(42)
    w = rand(M)
    x = rand(M)
    y = rand(M)
    z = rand(M)
    a = [w,x]
    b = [y,z]
    @test meanSquaredError(M,x,y) == distance(M,x,y)^2
    @test meanSquaredError(N,a,b) == 1/2*distance(N,a,b)^2
    @test meanAverageError(M,x,y) == distance(M,x,y)
    @test meanAverageError(N,a,b) == 1/2*sum( distance.(Ref(M), a, b) )
end