using Test, Manopt, ManifoldsBase

@testset "StoreStateAction" begin
    M = ManifoldsBase.DefaultManifold(2)
    p = [4.0, 2.0]
    st = GradientDescentState(
        M, p; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
    )
    f(M, q) = distance(M, q, p) .^ 2
    grad_f(M, q) = -2 * log(M, q, p)
    mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

    a = StoreStateAction(M; store_fields=[:p, :X])

    @test !has_storage(a, Manopt.PointStorageKey(:p))
    @test !has_storage(a, Manopt.TangentStorageKey(:X))
    update_storage!(a, mp, st)
    @test has_storage(a, Manopt.PointStorageKey(:p))
    @test has_storage(a, Manopt.TangentStorageKey(:X))
    @test get_storage(a, Manopt.PointStorageKey(:p)) == p
    @test get_storage(a, Manopt.TangentStorageKey(:X)) == [0.0, 0.0]

    a2 = StoreStateAction(M; store_points=[:p], store_vectors=[:X])
    @test !has_storage(a2, Manopt.PointStorageKey(:p))
    @test !has_storage(a2, Manopt.TangentStorageKey(:X))
    update_storage!(a2, mp, st)
    @test has_storage(a2, Manopt.PointStorageKey(:p))
    @test has_storage(a2, Manopt.TangentStorageKey(:X))
    @test get_storage(a2, Manopt.PointStorageKey(:p)) == p
    @test get_storage(a2, Manopt.TangentStorageKey(:X)) == [0.0, 0.0]
    a2b = StoreStateAction(M; store_points=Tuple{:p}, store_vectors=Tuple{:X})
    @test keys(a2.point_values) == keys(a2b.point_values)
    @test keys(a2.tangent_values) == keys(a2b.tangent_values)
    @test keys(a2.keys) == keys(a2b.keys)
end
