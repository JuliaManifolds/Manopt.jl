using Test, Manopt, ManifoldsBase, Manifolds

@testset "StoreStateAction" begin
    @testset "manifold $M" for M in [ManifoldsBase.DefaultManifold(2), Circle()]
        if M isa Circle
            p = 0.4
            X_zero = 0.0
        else
            p = [4.0, 2.0]
            X_zero = [0.0, 0.0]
        end

        st = GradientDescentState(
            M, p; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
        )
        f(M, q) = distance(M, q, p) .^ 2
        grad_f(M, q) = -2 * log(M, q, p)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

        a = StoreStateAction(M; store_fields=[:p, :X])

        @test !has_storage(a, Manopt.PointStorageKey(:p))
        @test !has_storage(a, Manopt.VectorStorageKey(:X))
        update_storage!(a, mp, st)
        @test has_storage(a, Manopt.PointStorageKey(:p))
        @test has_storage(a, Manopt.VectorStorageKey(:X))
        @test get_storage(a, Manopt.PointStorageKey(:p)) == p
        @test get_storage(a, Manopt.VectorStorageKey(:X)) == X_zero

        a2 = StoreStateAction(M; store_points=Tuple{:p}, store_vectors=Tuple{:X})
        @test !has_storage(a2, Manopt.PointStorageKey(:p))
        @test !has_storage(a2, Manopt.VectorStorageKey(:X))
        update_storage!(a2, mp, st)
        @test has_storage(a2, Manopt.PointStorageKey(:p))
        @test has_storage(a2, Manopt.VectorStorageKey(:X))
        @test get_storage(a2, Manopt.PointStorageKey(:p)) == p
        @test get_storage(a2, Manopt.VectorStorageKey(:X)) == X_zero
        a2b = StoreStateAction(M; store_points=Tuple{:p}, store_vectors=Tuple{:X})
        @test keys(a2.point_values) == keys(a2b.point_values)
        @test keys(a2.vector_values) == keys(a2b.vector_values)
        @test keys(a2.keys) == keys(a2b.keys)

        # make sure fast storage is actually fast
        @test (@allocated update_storage!(a2, mp, st)) == 0
        @test (@allocated has_storage(a2, Manopt.PointStorageKey(:p))) == 0
        if M isa ManifoldsBase.DefaultManifold
            @test (@allocated get_storage(a2, Manopt.PointStorageKey(:p))) == 0
        end

        a3 = StoreStateAction(M; store_points=[:p], store_vectors=[:X])
        @test !has_storage(a3, Manopt.PointStorageKey(:p))
        @test !has_storage(a3, Manopt.VectorStorageKey(:X))
        update_storage!(a3, mp, st)
        @test has_storage(a3, Manopt.PointStorageKey(:p))
        @test has_storage(a3, Manopt.VectorStorageKey(:X))
        @test get_storage(a3, Manopt.PointStorageKey(:p)) == p
        @test get_storage(a3, Manopt.VectorStorageKey(:X)) == X_zero

        # make sure fast storage is actually fast
        @test (@allocated update_storage!(a3, mp, st)) == 0
        @test (@allocated has_storage(a3, Manopt.PointStorageKey(:p))) == 0
        if M isa ManifoldsBase.DefaultManifold
            @test (@allocated get_storage(a3, Manopt.PointStorageKey(:p))) == 0
        end
    end

    @test Manopt.extract_type_from_namedtuple(typeof((; a=10, b='a')), Val(:c)) === Any
end
