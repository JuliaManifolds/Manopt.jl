using Test, Manopt, ManifoldsBase, Manifolds

@testset "StoreStateAction" begin
    @testset "manifold $M" for M in [ManifoldsBase.DefaultManifold(2), Circle()]
        if M isa Circle
            p = 0.4
            p_fast = fill(p)
            X_zero = 0.0
            X_zero_fast = fill(0.0)
        else
            p = [4.0, 2.0]
            p_fast = p
            X_zero = [0.0, 0.0]
            X_zero_fast = X_zero
        end

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
        @test get_storage(a, Manopt.TangentStorageKey(:X)) == X_zero

        a2 = StoreStateAction(M; store_points=Tuple{:p}, store_vectors=Tuple{:X})
        @test !has_storage(a2, Manopt.PointStorageKey(:p))
        @test !has_storage(a2, Manopt.TangentStorageKey(:X))
        update_storage!(a2, mp, st)
        @test has_storage(a2, Manopt.PointStorageKey(:p))
        @test has_storage(a2, Manopt.TangentStorageKey(:X))
        @test get_storage(a2, Manopt.PointStorageKey(:p)) == p_fast
        @test get_storage(a2, Manopt.TangentStorageKey(:X)) == X_zero_fast

        a3 = StoreStateAction(M; store_points=[:p], store_vectors=[:X])
        @test !has_storage(a3, Manopt.PointStorageKey(:p))
        @test !has_storage(a3, Manopt.TangentStorageKey(:X))
        update_storage!(a3, mp, st)
        @test has_storage(a3, Manopt.PointStorageKey(:p))
        @test has_storage(a3, Manopt.TangentStorageKey(:X))
        @test get_storage(a3, Manopt.PointStorageKey(:p)) == p_fast
        @test get_storage(a3, Manopt.TangentStorageKey(:X)) == X_zero_fast
    end
end
