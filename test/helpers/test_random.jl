@testset "random" begin
    M = Euclidean(3, 5)
    p = random_point(M, :Gaussian, 3.6)
    v = random_tangent(M, p)
    @test size(p) == (3,5)
    @test size(v) == (3,5)


    M = Circle()
    p = random_point(M)
    v = random_tangent(M, p)
    @test size(p) == ()
    @test size(v) == ()


    M = FixedRankMatrices(3, 5, 2)
    p = random_point(M)
    v = random_tangent(M, p)
    @test (size(p.U, 1), size(p.Vt, 2)) == (3,5)
    @test (size(v.U, 1), size(v.Vt, 2)) == (3,5)


    M = Grassmann(5, 3)
    p = random_point(M, :Gaussian, 3.6)
    v = random_tangent(M, p)
    @test size(p) == (5,3)
    @test size(v) == (5,3)


    M = Rotations(5)
    p = random_point(M, :Gaussian, 3.6)
    v = random_tangent(M, p)
    @test size(p) == (5,5)
    @test size(v) == (5,5)


    M = SymmetricPositiveDefinite(5)
    p = random_point(M, :Gaussian, 3.6)
    v = random_tangent(M, p)
    @test size(p) == (5,5)
    @test size(v) == (5,5)

    M = Stiefel(5, 3)
    p = random_point(M, :Gaussian, 3.6)
    v = random_tangent(M, p)
    @test size(p) == (5,3)
    @test size(v) == (5,3)

    M = Sphere(5)
    p = random_point(M, :Gaussian, 3.6)
    v = random_tangent(M, p)
    @test size(p) == (6,)
    @test size(v) == (6,)

    M = TangentBundle(Rotations(3))
    p = random_point(M)
    v = random_tangent(M, p)
    @test size(p[M, :point]) == (3,3)
    @test size(p[M, :vector]) == (3,3)
    @test size(v[M, :point]) == (3,3)
    @test size(v[M, :vector]) == (3,3)
end
