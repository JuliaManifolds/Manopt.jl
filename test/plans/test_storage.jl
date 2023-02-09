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

    a = StoreStateAction(M, [:p, :X], Tuple{}, Tuple{})

    @test !has_storage(a, Manopt.PointStorageKey(:p))
    @test !has_storage(a, Manopt.TangentStorageKey(:X))
    update_storage!(a, mp, st)
    @test has_storage(a, Manopt.PointStorageKey(:p))
    @test has_storage(a, Manopt.TangentStorageKey(:X))
    @test get_storage(a, Manopt.PointStorageKey(:p)) == p
    @test get_storage(a, Manopt.PointStorageKey(:X)) == [0.0, 0.0]
end
